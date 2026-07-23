"""Trailing-stop optimization (test-only): reconstructs, bar-by-bar and
strictly without look-ahead, what would have happened if a trailing stop
had been layered on top of each trade's EXISTING entry/stop-loss/exit.

Never touches original rPnL / initalSL / entry logic -- see
trailing-stop-optimization-prompt.md. This module only produces a
hypothetical, alternative EXIT for the same 789 trades.

Definition used (confirmed with the user): trailing_pct is a percentage
GIVEBACK of open profit, not a fixed price distance. Once a trade's running
peak favorable price implies a gain of at least TRAILING_ACTIVATION_MFE_R
over entry, the hypothetical stop trails at

    trail_price = peak - trailing_pct * (peak - entry)      [buy]
    trail_price = peak + trailing_pct * (entry - peak)      [sell]

and the effective stop at any moment is max(initalSL, trail_price) for a
buy / min(initalSL, trail_price) for a sell -- i.e. the trailing stop can
only ever tighten the exit, never loosen the original stop-loss (scope:
"do not change stop-loss levels").

No-look-ahead / conservative fill, per candle, in order:
  1. Compute the effective stop using the peak known BEFORE this candle.
  2. If the candle's low (buy) / high (sell) crosses the effective stop,
     the trade exits this candle. Fill = the stop price itself, unless the
     candle's open already gapped past it, in which case fill = open (the
     worse, conservative price) -- we cannot know intra-candle high/low
     ordering from 1-minute OHLC, so this ordering (check-stop-before-
     extending-peak) is the conservative assumption throughout.
  3. Only if not exited, extend the peak using this candle's high (buy) /
     low (sell).

Two scenarios are produced (per the prompt's critical note):
  - conservative: capped at the trade's own dateEnd_utc. If the trailing
    stop never triggers by then, the hypothetical exit falls back to the
    ORIGINAL actual exit (no extension, no changed duration).
  - aggressive: allowed to keep walking up to
    TRAILING_AGGRESSIVE_MAX_EXTENSION_MINUTES past dateEnd_utc looking for
    a trigger. If still not triggered (or candle data runs out first), it
    also falls back to the original exit, flagged as "unresolved".

Both scenarios share the same underlying peak-price path, so each trade is
walked ONCE per trailing_pct over the (larger, aggressive) window; the
conservative result is simply "was the first trigger inside the original
window."
"""
import numpy as np
import pandas as pd

from src import config, data_loading


def _simulate_one_trade(entry: float, sl: float, side: str, risk: float,
                         opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                         pct_grid: list, activation_r: float, orig_len: int):
    """Walks one trade's candle window (already sliced to [start, extended_end))
    and returns, for each pct in pct_grid, (trigger_idx or None, fill_price or None).
    orig_len = number of candles in the ORIGINAL (non-extended) window, so
    callers can tell whether a trigger fell inside or after it.
    """
    is_buy = side == "buy"
    peak = entry
    pending = {pct: None for pct in pct_grid}  # None until resolved
    n = len(opens)

    for idx in range(n):
        o, h, l = opens[idx], highs[idx], lows[idx]

        peak_gain_r = (peak - entry) / risk if is_buy else (entry - peak) / risk
        activated = peak_gain_r >= activation_r

        for pct in pct_grid:
            if pending[pct] is not None:
                continue
            if activated:
                trail_price = peak - pct * (peak - entry) if is_buy else peak + pct * (entry - peak)
                eff_stop = max(sl, trail_price) if is_buy else min(sl, trail_price)
            else:
                eff_stop = sl

            if is_buy:
                if l <= eff_stop:
                    fill = eff_stop if o >= eff_stop else o
                    pending[pct] = (idx, fill)
            else:
                if h >= eff_stop:
                    fill = eff_stop if o <= eff_stop else o
                    pending[pct] = (idx, fill)

        if all(v is not None for v in pending.values()):
            break

        peak = max(peak, h) if is_buy else min(peak, l)

    return pending


def simulate_trailing_stops(trades: pd.DataFrame, candles: pd.DataFrame,
                             pct_grid: list = None,
                             activation_r: float = config.TRAILING_ACTIVATION_MFE_R,
                             max_extension_minutes: int = config.TRAILING_AGGRESSIVE_MAX_EXTENSION_MINUTES
                             ) -> pd.DataFrame:
    """Returns a long-format DataFrame, one row per (trade id, trailing_pct,
    scenario in {conservative, aggressive}), with the hypothetical exit.
    """
    pct_grid = pct_grid or config.TRAILING_STOP_GRID

    ts = candles["datetime_utc"].values
    opens_all = candles["open"].values
    highs_all = candles["high"].values
    lows_all = candles["low"].values

    starts = trades["dateStart_utc"].values
    ends = trades["dateEnd_utc"].values
    ext = np.timedelta64(max_extension_minutes, "m")

    start_idx = np.searchsorted(ts, starts, side="left")
    end_idx = np.searchsorted(ts, ends, side="right")
    ext_end_idx = np.searchsorted(ts, ends + ext, side="right")
    ext_end_idx = np.minimum(ext_end_idx, len(ts))

    entry = trades["entryPrice"].values
    sl = trades["initalSL"].values
    side = trades["side"].values
    risk = np.abs(entry - sl)
    risk = np.where(risk == 0, np.nan, risk)

    rows = []
    for i in range(len(trades)):
        lo, hi, ext_hi = start_idx[i], end_idx[i], ext_end_idx[i]
        orig_len = hi - lo
        r_i = risk[i]

        trade_id = trades["id"].iloc[i]
        orig_exit_price = trades["avgClosePrice"].iloc[i]
        orig_r = trades["avgRiskReward"].iloc[i]

        if ext_hi <= lo or np.isnan(r_i) or orig_len <= 0:
            # no in-window candle data at all, or degenerate risk: trailing
            # can't be simulated: both scenarios fall back to the original exit
            for pct in pct_grid:
                for scenario in ("conservative", "aggressive"):
                    rows.append(dict(id=trade_id, pct=pct, scenario=scenario,
                                      trail_exit_price=orig_exit_price, trail_r=orig_r,
                                      triggered=False, unresolved=True))
            continue

        pending = _simulate_one_trade(
            entry[i], sl[i], side[i], r_i,
            opens_all[lo:ext_hi], highs_all[lo:ext_hi], lows_all[lo:ext_hi],
            pct_grid, activation_r, orig_len,
        )

        is_buy = side[i] == "buy"
        for pct in pct_grid:
            result = pending[pct]

            # conservative: only counts if the trigger fell within the ORIGINAL window
            if result is not None and result[0] < orig_len:
                idx, fill = result
                trail_r = (fill - entry[i]) / r_i if is_buy else (entry[i] - fill) / r_i
                rows.append(dict(id=trade_id, pct=pct, scenario="conservative",
                                  trail_exit_price=fill, trail_r=trail_r,
                                  triggered=True, unresolved=False))
            else:
                rows.append(dict(id=trade_id, pct=pct, scenario="conservative",
                                  trail_exit_price=orig_exit_price, trail_r=orig_r,
                                  triggered=False, unresolved=False))

            # aggressive: counts any trigger found within the extended window
            if result is not None:
                idx, fill = result
                trail_r = (fill - entry[i]) / r_i if is_buy else (entry[i] - fill) / r_i
                rows.append(dict(id=trade_id, pct=pct, scenario="aggressive",
                                  trail_exit_price=fill, trail_r=trail_r,
                                  triggered=True, unresolved=False))
            else:
                rows.append(dict(id=trade_id, pct=pct, scenario="aggressive",
                                  trail_exit_price=orig_exit_price, trail_r=orig_r,
                                  triggered=False, unresolved=True))

    out = pd.DataFrame(rows)
    meta = trades[["id", "dateStart_utc", "dateEnd_utc", "side", "amount", "avgRiskReward"]].rename(
        columns={"avgRiskReward": "orig_r"})
    meta["risk_price"] = np.abs(trades["entryPrice"] - trades["initalSL"])
    out = out.merge(meta, on="id", how="left")
    out["trail_pnl_usd"] = out["trail_r"] * out["risk_price"] * out["amount"]
    out["orig_pnl_usd"] = out["orig_r"] * out["risk_price"] * out["amount"]
    out["month"] = out["dateStart_utc"].dt.to_period("M").astype(str)
    return out


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = simulate_trailing_stops(trades, candles)

    print(f"rows: {len(sim)}  (expect {len(trades)} trades x {len(config.TRAILING_STOP_GRID)} pcts x 2 scenarios "
          f"= {len(trades) * len(config.TRAILING_STOP_GRID) * 2})")
    for scenario in ("conservative", "aggressive"):
        s = sim[sim["scenario"] == scenario]
        print(f"\n=== {scenario} ===")
        summary = s.groupby("pct").agg(
            n=("id", "count"),
            pct_triggered=("triggered", "mean"),
            pct_unresolved=("unresolved", "mean"),
            mean_trail_r=("trail_r", "mean"),
            mean_orig_r=("orig_r", "mean"),
        )
        print(summary)
