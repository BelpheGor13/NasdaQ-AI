"""Mechanical "which came first" reconstruction, straight from the 1-minute
candles -- the calculation the user asked for, and the one that does NOT
depend on any of the unreliable trade-log columns (idealTP / maxTP /
maxRiskReward are all null-or-zero for exactly the losing and breakeven
trades we care about; see idealtp_data_quality_check.py).

For every trade, we walk forward from dateStart_utc and ask a single
purely mechanical question: starting from entryPrice, with the ORIGINAL
initalSL and a target placed at a fixed R multiple, which level does price
touch first?

This answers "what would each trade have done if it had simply been left
alone -- no moving the stop to breakeven, no early manual close" and lets
us price that policy honestly across ALL 789 trades, including the 465
full-loss and 93 breakeven ones the trade log can't describe.

Conservative-fill conventions (same as the rest of this project):
  - If a single candle's range spans BOTH the stop and the target, we
    charge the STOP. We cannot see intrabar order from OHLC, so we always
    resolve the ambiguity against the trade rather than for it.
  - A gap through a level fills at the candle open (the worse price),
    never at the level itself.
  - We look forward up to config.EXIT_STRATEGY_MAX_EXTENSION_MINUTES past
    the trade's own dateEnd_utc. If neither level is hit in that whole
    window, the trade is reported as unresolved and excluded from the
    headline numbers rather than being silently scored.
"""
import numpy as np
import pandas as pd

from src import config, data_loading, stats_utils

# Target multiples to race against the original stop. 2.0 first: the user
# states a >=1:2 R:R house rule, so that is the policy-relevant number.
TARGET_R_MULTIPLES = [1.0, 1.5, 2.0, 3.0, 4.0]


def _race_one_trade(entry, sl, side, risk, target_r, opens, highs, lows, orig_len):
    """Returns (outcome, r_result, bar_index, hit_within_original_window).

    outcome is one of: 'target', 'stop', 'unresolved'.
    """
    is_buy = side == "buy"
    target = entry + target_r * risk if is_buy else entry - target_r * risk

    for idx in range(len(opens)):
        o, h, l = opens[idx], highs[idx], lows[idx]

        if is_buy:
            hit_stop = l <= sl
            hit_target = h >= target
        else:
            hit_stop = h >= sl
            hit_target = l <= target

        if hit_stop and hit_target:
            # Ambiguous bar: charge the stop (conservative).
            fill = sl if (o >= sl if is_buy else o <= sl) else o
            r = (fill - entry) / risk if is_buy else (entry - fill) / risk
            return "stop", r, idx, idx < orig_len
        if hit_stop:
            fill = sl if (o >= sl if is_buy else o <= sl) else o
            r = (fill - entry) / risk if is_buy else (entry - fill) / risk
            return "stop", r, idx, idx < orig_len
        if hit_target:
            fill = target if (o <= target if is_buy else o >= target) else o
            r = (fill - entry) / risk if is_buy else (entry - fill) / risk
            return "target", r, idx, idx < orig_len

    return "unresolved", np.nan, -1, False


def run_race(trades: pd.DataFrame, candles: pd.DataFrame,
             target_multiples=TARGET_R_MULTIPLES) -> pd.DataFrame:
    ts = candles["datetime_utc"].values
    opens_all = candles["open"].values
    highs_all = candles["high"].values
    lows_all = candles["low"].values

    starts = trades["dateStart_utc"].values
    ends = trades["dateEnd_utc"].values
    ext = np.timedelta64(config.EXIT_STRATEGY_MAX_EXTENSION_MINUTES, "m")

    start_idx = np.searchsorted(ts, starts, side="left")
    end_idx = np.searchsorted(ts, ends, side="right")
    ext_end_idx = np.minimum(np.searchsorted(ts, ends + ext, side="right"), len(ts))

    entry_arr = trades["entryPrice"].values
    sl_arr = trades["initalSL"].values
    side_arr = trades["side"].values
    risk_arr = np.abs(entry_arr - sl_arr)
    risk_arr = np.where(risk_arr == 0, np.nan, risk_arr)

    rows = []
    for i in range(len(trades)):
        lo, hi, ext_hi = start_idx[i], end_idx[i], ext_end_idx[i]
        orig_len = hi - lo
        r_i = risk_arr[i]
        trade_id = trades["id"].iloc[i]
        actual_r = trades["avgRiskReward"].iloc[i]

        if ext_hi <= lo or np.isnan(r_i):
            for tr in target_multiples:
                rows.append(dict(id=trade_id, target_r=tr, outcome="unresolved", race_r=np.nan,
                                 bars_to_resolution=np.nan, resolved_within_original_window=False,
                                 actual_r=actual_r))
            continue

        for tr in target_multiples:
            outcome, r, idx, within = _race_one_trade(
                entry_arr[i], sl_arr[i], side_arr[i], r_i, tr,
                opens_all[lo:ext_hi], highs_all[lo:ext_hi], lows_all[lo:ext_hi], orig_len,
            )
            rows.append(dict(id=trade_id, target_r=tr, outcome=outcome, race_r=r,
                             bars_to_resolution=idx if idx >= 0 else np.nan,
                             resolved_within_original_window=within, actual_r=actual_r))

    out = pd.DataFrame(rows)
    meta = trades[["id", "dateStart_utc", "side", "amount", "avgRiskReward"]].copy()
    meta["risk_price"] = np.abs(trades["entryPrice"] - trades["initalSL"])
    out = out.merge(meta.drop(columns=["avgRiskReward"]), on="id", how="left")
    out["race_pnl_usd"] = out["race_r"] * out["risk_price"] * out["amount"]
    out["actual_pnl_usd"] = out["actual_r"] * out["risk_price"] * out["amount"]
    out["month"] = out["dateStart_utc"].dt.to_period("M").astype(str)
    out["year"] = out["dateStart_utc"].dt.year
    return out


def summarize(race: pd.DataFrame) -> pd.DataFrame:
    """Headline table: for each target multiple, what would leaving every
    trade completely alone have produced, vs what actually happened."""
    rows = []
    for tr, group in race.groupby("target_r"):
        resolved = group[group["outcome"] != "unresolved"]
        r_race = resolved["race_r"].dropna().values
        r_actual = resolved["actual_r"].dropna().values
        rows.append({
            "target_r": tr,
            "n_resolved": len(resolved),
            "n_unresolved": int((group["outcome"] == "unresolved").sum()),
            "pct_hit_target": float((resolved["outcome"] == "target").mean()),
            "mechanical_win_rate": float((r_race > 0).mean()),
            "mechanical_expectancy_r": stats_utils.expectancy(r_race),
            "mechanical_pf": stats_utils.profit_factor(r_race),
            "mechanical_total_r": float(r_race.sum()),
            "actual_total_r": float(r_actual.sum()),
            "difference_r": float(r_race.sum() - r_actual.sum()),
            "mechanical_total_usd": float(resolved["race_pnl_usd"].sum()),
            "actual_total_usd": float(resolved["actual_pnl_usd"].sum()),
        })
    return pd.DataFrame(rows)


def by_actual_outcome_group(race: pd.DataFrame, trades: pd.DataFrame, target_r: float = 2.0) -> pd.DataFrame:
    """The part the user actually cares about: split by what the trade
    REALLY did, and show what leaving it alone would have done instead.
    The breakeven and full-loss groups are the whole point -- those are
    the ones the trade log cannot describe on its own."""
    sub = race[(race["target_r"] == target_r) & (race["outcome"] != "unresolved")].copy()
    actual = trades.set_index("id")["avgRiskReward"]
    sub["actual_group"] = np.select(
        [sub["id"].map(actual) == -1.0, sub["id"].map(actual) == 0.0, sub["id"].map(actual) > 0],
        ["full_loss", "breakeven", "winner"], default="partial")

    rows = []
    for grp, g in sub.groupby("actual_group"):
        rows.append({
            "actual_group": grp, "n": len(g),
            "pct_would_hit_target_first": float((g["outcome"] == "target").mean()),
            "actual_total_r": float(g["actual_r"].sum()),
            "mechanical_total_r": float(g["race_r"].sum()),
            "difference_r": float(g["race_r"].sum() - g["actual_r"].sum()),
            "difference_usd": float(g["race_pnl_usd"].sum() - g["actual_pnl_usd"].sum()),
        })
    return pd.DataFrame(rows).sort_values("difference_r")


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()

    race = run_race(trades, candles)

    print("=== leave-it-alone: stop vs target race, per target multiple ===")
    print(summarize(race).to_string(index=False))

    print("\n=== at 1:2 (the stated house rule), split by what the trade ACTUALLY did ===")
    print(by_actual_outcome_group(race, trades, target_r=2.0).to_string(index=False))
