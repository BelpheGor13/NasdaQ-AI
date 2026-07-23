"""Phase 3: unified, no-look-ahead bar-by-bar simulator for every exit
strategy in the spec (test-only; see hidden-patterns-exit-optimization-
prompt.md): Fixed TP (idealTP price, 2R/3R/4R), trailing stop (grid reused
from the trailing-stop-optimization analysis), time-based, partial profit-
taking, and MFE-aware adaptive trailing.

Shared invariant across every non-baseline strategy: the original initalSL
always remains an active floor -- these strategies test alternative PROFIT-
TAKING/exit-timing rules layered on top of the existing risk management,
never a looser stop (same invariant as trailing_stop_simulation.py).

Same two scenarios as the trailing-stop analysis, for the same reason
(Critical Note 2 in the prompt): conservative (capped at the trade's own
dateEnd_utc, falling back to the original actual exit if nothing triggers)
and aggressive (allowed to look up to
config.EXIT_STRATEGY_MAX_EXTENSION_MINUTES past dateEnd_utc).
"""
import numpy as np
import pandas as pd

from src import config, data_loading

FIXED_TP_KEYS = ["fixed_tp_idealTP"] + [f"fixed_tp_{int(m)}R" for m in config.FIXED_TP_R_MULTIPLES]
TRAILING_KEYS = [f"trailing_{int(p*100)}pct" for p in config.TRAILING_STOP_GRID]
TIME_KEYS = [f"time_{h}h" for h in config.TIME_BASED_EXIT_HOURS]
SINGLE_TRIGGER_STRATEGIES = FIXED_TP_KEYS + TRAILING_KEYS + TIME_KEYS + ["mfe_aware"]
ALL_STRATEGIES = ["baseline"] + SINGLE_TRIGGER_STRATEGIES + ["partial_profit"]


def _stop_fill(o: float, level: float, is_buy: bool) -> float:
    """Conservative fill for a stop-style order: the level itself, unless
    the candle's open already gapped past it (then the worse open price)."""
    if is_buy:
        return level if o >= level else o
    return level if o <= level else o


def _simulate_one_trade(entry, sl, ideal_tp, side, risk, opens, highs, lows, closes, ts, orig_r):
    is_buy = side == "buy"
    sign = 1.0 if is_buy else -1.0

    fixed_tp_targets = {"fixed_tp_idealTP": ideal_tp}
    for m in config.FIXED_TP_R_MULTIPLES:
        fixed_tp_targets[f"fixed_tp_{int(m)}R"] = entry + sign * m * risk

    time_targets = {f"time_{h}h": ts[0] + np.timedelta64(h, "h") if len(ts) else None
                     for h in config.TIME_BASED_EXIT_HOURS}

    partial_targets = []
    for frac, mult in config.PARTIAL_PROFIT_LEGS:
        level = ideal_tp if mult is None else entry + sign * mult * risk
        partial_targets.append([frac, level, False, None, None])  # frac, level, done, idx, r

    pending = {k: None for k in fixed_tp_targets}
    pending.update({k: None for k in TRAILING_KEYS})
    pending.update({k: None for k in time_targets})
    pending["mfe_aware"] = None

    peak = entry
    partial_remaining = 1.0
    partial_events = []  # (fraction, idx_or_None, r)

    n = len(opens)
    for idx in range(n):
        o, h, l, c, t = opens[idx], highs[idx], lows[idx], closes[idx], ts[idx]
        sl_breach = (l <= sl) if is_buy else (h >= sl)
        sl_fill = _stop_fill(o, sl, is_buy) if sl_breach else None

        # --- fixed TP legs ---
        for key, level in fixed_tp_targets.items():
            if pending[key] is not None:
                continue
            touched = (h >= level) if is_buy else (l <= level)
            if sl_breach and not touched:
                pending[key] = (idx, sl_fill)
            elif touched:
                pending[key] = (idx, level)

        # --- time-based legs ---
        for key, threshold in time_targets.items():
            if pending[key] is not None or threshold is None:
                continue
            if sl_breach:
                pending[key] = (idx, sl_fill)
            elif t >= threshold:
                pending[key] = (idx, c)

        # --- trailing stop grid (peak-based, mirrors trailing_stop_simulation.py) ---
        peak_gain_r = (peak - entry) / risk if is_buy else (entry - peak) / risk
        activated = peak_gain_r >= config.TRAILING_ACTIVATION_MFE_R
        for pct, key in zip(config.TRAILING_STOP_GRID, TRAILING_KEYS):
            if pending[key] is not None:
                continue
            if activated:
                trail_price = peak - pct * (peak - entry) if is_buy else peak + pct * (entry - peak)
                eff_stop = max(sl, trail_price) if is_buy else min(sl, trail_price)
            else:
                eff_stop = sl
            touched = (l <= eff_stop) if is_buy else (h >= eff_stop)
            if touched:
                pending[key] = (idx, _stop_fill(o, eff_stop, is_buy))

        # --- MFE-aware adaptive trailing ---
        if pending["mfe_aware"] is None:
            if peak_gain_r >= config.MFE_AWARE_HIGH_R:
                dyn_pct = config.MFE_AWARE_TIGHT_TRAIL_PCT
            elif peak_gain_r >= config.MFE_AWARE_LOW_R:
                dyn_pct = config.MFE_AWARE_MID_TRAIL_PCT
            else:
                dyn_pct = None
            if dyn_pct is None:
                eff_stop = sl
            else:
                trail_price = peak - dyn_pct * (peak - entry) if is_buy else peak + dyn_pct * (entry - peak)
                eff_stop = max(sl, trail_price) if is_buy else min(sl, trail_price)
            touched = (l <= eff_stop) if is_buy else (h >= eff_stop)
            if touched:
                pending["mfe_aware"] = (idx, _stop_fill(o, eff_stop, is_buy))

        # --- partial profit-taking ---
        if partial_remaining > 1e-9:
            if sl_breach:
                r = (sl_fill - entry) / risk if is_buy else (entry - sl_fill) / risk
                partial_events.append((partial_remaining, idx, r))
                partial_remaining = 0.0
            else:
                for leg in partial_targets:
                    frac, level, done, _, _ = leg
                    if done:
                        continue
                    touched = (h >= level) if is_buy else (l <= level)
                    if touched:
                        r = (level - entry) / risk if is_buy else (entry - level) / risk
                        partial_events.append((frac, idx, r))
                        leg[2] = True
                        partial_remaining -= frac

        # --- extend the peak using this candle (after all checks, no look-ahead) ---
        peak = max(peak, h) if is_buy else min(peak, l)

        if partial_remaining <= 1e-9 and all(v is not None for v in pending.values()):
            break

    if partial_remaining > 1e-9:
        partial_events.append((partial_remaining, None, orig_r))

    return pending, partial_events


def simulate_exit_strategies(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    ts = candles["datetime_utc"].values
    opens_all = candles["open"].values
    highs_all = candles["high"].values
    lows_all = candles["low"].values
    closes_all = candles["close"].values

    starts = trades["dateStart_utc"].values
    ends = trades["dateEnd_utc"].values
    ext = np.timedelta64(config.EXIT_STRATEGY_MAX_EXTENSION_MINUTES, "m")

    start_idx = np.searchsorted(ts, starts, side="left")
    end_idx = np.searchsorted(ts, ends, side="right")
    ext_end_idx = np.minimum(np.searchsorted(ts, ends + ext, side="right"), len(ts))

    entry_arr = trades["entryPrice"].values
    sl_arr = trades["initalSL"].values
    ideal_tp_arr = trades["idealTP"].values
    side_arr = trades["side"].values
    risk_arr = np.abs(entry_arr - sl_arr)
    risk_arr = np.where(risk_arr == 0, np.nan, risk_arr)

    rows = []
    for i in range(len(trades)):
        lo, hi, ext_hi = start_idx[i], end_idx[i], ext_end_idx[i]
        orig_len = hi - lo
        r_i = risk_arr[i]
        trade_id = trades["id"].iloc[i]
        orig_exit_price = trades["avgClosePrice"].iloc[i]
        orig_r = trades["avgRiskReward"].iloc[i]

        for scenario in ("conservative", "aggressive"):
            rows.append(dict(id=trade_id, strategy="baseline", scenario=scenario,
                              exit_r=orig_r, triggered=False, unresolved=False))

        if ext_hi <= lo or pd.isna(r_i) or pd.isna(ideal_tp_arr[i]) or orig_len <= 0:
            for strat in SINGLE_TRIGGER_STRATEGIES + ["partial_profit"]:
                for scenario in ("conservative", "aggressive"):
                    rows.append(dict(id=trade_id, strategy=strat, scenario=scenario,
                                      exit_r=orig_r, triggered=False, unresolved=True))
            continue

        pending, partial_events = _simulate_one_trade(
            entry_arr[i], sl_arr[i], ideal_tp_arr[i], side_arr[i], r_i,
            opens_all[lo:ext_hi], highs_all[lo:ext_hi], lows_all[lo:ext_hi], closes_all[lo:ext_hi],
            ts[lo:ext_hi], orig_r,
        )

        is_buy = side_arr[i] == "buy"
        for key in SINGLE_TRIGGER_STRATEGIES:
            result = pending[key]
            if result is not None and result[0] < orig_len:
                idx, fill = result
                r = (fill - entry_arr[i]) / r_i if is_buy else (entry_arr[i] - fill) / r_i
                rows.append(dict(id=trade_id, strategy=key, scenario="conservative",
                                  exit_r=r, triggered=True, unresolved=False))
            else:
                rows.append(dict(id=trade_id, strategy=key, scenario="conservative",
                                  exit_r=orig_r, triggered=False, unresolved=False))
            if result is not None:
                idx, fill = result
                r = (fill - entry_arr[i]) / r_i if is_buy else (entry_arr[i] - fill) / r_i
                rows.append(dict(id=trade_id, strategy=key, scenario="aggressive",
                                  exit_r=r, triggered=True, unresolved=False))
            else:
                rows.append(dict(id=trade_id, strategy=key, scenario="aggressive",
                                  exit_r=orig_r, triggered=False, unresolved=True))

        for scenario, len_cap in (("conservative", orig_len), ("aggressive", None)):
            r_total = 0.0
            any_triggered = False
            for frac, idx, r in partial_events:
                if idx is not None and (len_cap is None or idx < len_cap):
                    r_total += frac * r
                    any_triggered = True
                else:
                    r_total += frac * orig_r
            rows.append(dict(id=trade_id, strategy="partial_profit", scenario=scenario,
                              exit_r=r_total, triggered=any_triggered,
                              unresolved=not any_triggered and scenario == "aggressive"))

    out = pd.DataFrame(rows)
    meta = trades[["id", "dateStart_utc", "dateEnd_utc", "side", "amount", "avgRiskReward"]].rename(
        columns={"avgRiskReward": "orig_r"})
    meta["risk_price"] = np.abs(trades["entryPrice"] - trades["initalSL"])
    out = out.merge(meta, on="id", how="left")
    out["exit_pnl_usd"] = out["exit_r"] * out["risk_price"] * out["amount"]
    out["month"] = out["dateStart_utc"].dt.to_period("M").astype(str)
    out["year"] = out["dateStart_utc"].dt.year
    return out


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = simulate_exit_strategies(trades, candles)

    print(f"rows: {len(sim)}, strategies: {sim['strategy'].nunique()}")
    print(sim["strategy"].unique())

    cons = sim[sim["scenario"] == "conservative"]
    summary = cons.groupby("strategy").agg(
        n=("id", "count"), mean_r=("exit_r", "mean"), pct_triggered=("triggered", "mean"),
    ).sort_values("mean_r", ascending=False)
    print(summary)
