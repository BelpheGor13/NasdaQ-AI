"""Focused recomputation requested directly by the user: of the 93 trades
closed at exactly breakeven (avgRiskReward == 0.0), ALL 93 show a
profitable idealTP reading (idealTP is MFE-before-stop, confirmed in
idealtp_data_quality_check.py -- these trades were genuinely in profit at
some point before being brought back to exactly $0).

For each, what would have happened if the stop had never been moved to
breakeven -- i.e. left at the original initalSL, racing mechanically
against a target of 2.1R (the user's own figure, since the real original
target isn't recorded for a trade that never reached it and maxTP is
null for exactly this group).

Same conservative-fill conventions as every other mechanical race in this
project: an ambiguous bar (spans both stop and target) is charged to the
stop, and a gap fills at the bar's open, never at the level itself.
"""
import numpy as np
import pandas as pd

from src import config, data_loading, stats_utils


def recompute(trades: pd.DataFrame, candles: pd.DataFrame, target_r: float = 2.1) -> pd.DataFrame:
    entry_all = trades["entryPrice"].values
    sl_all = trades["initalSL"].values
    idealtp_all = trades["idealTP"].values
    side_all = trades["side"].values
    risk_all = np.abs(entry_all - sl_all)
    idealtp_r = np.where(side_all == "buy", (idealtp_all - entry_all) / risk_all,
                          (entry_all - idealtp_all) / risk_all)

    be_mask = trades["avgRiskReward"].values == 0.0
    profitable_mask = be_mask & (idealtp_r > 0)
    subset = trades[profitable_mask].reset_index(drop=True)

    ts = candles["datetime_utc"].values
    opens_all = candles["open"].values
    highs_all = candles["high"].values
    lows_all = candles["low"].values
    ext = np.timedelta64(config.EXIT_STRATEGY_MAX_EXTENSION_MINUTES, "m")

    starts = subset["dateStart_utc"].values
    ends = subset["dateEnd_utc"].values
    start_idx = np.searchsorted(ts, starts, side="left")
    ext_end_idx = np.minimum(np.searchsorted(ts, ends + ext, side="right"), len(ts))

    rows = []
    for i in range(len(subset)):
        t = subset.iloc[i]
        entry, sl, side = t["entryPrice"], t["initalSL"], t["side"]
        is_buy = side == "buy"
        risk = abs(entry - sl)
        target = entry + target_r * risk if is_buy else entry - target_r * risk
        lo, hi = start_idx[i], ext_end_idx[i]

        outcome, r = "unresolved", np.nan
        for k in range(lo, hi):
            o, h, l = opens_all[k], highs_all[k], lows_all[k]
            hit_sl = (l <= sl) if is_buy else (h >= sl)
            hit_tp = (h >= target) if is_buy else (l <= target)
            if hit_sl:
                fill = sl if (o >= sl if is_buy else o <= sl) else o
                outcome = "stop"
            elif hit_tp:
                fill = target if (o <= target if is_buy else o >= target) else o
                outcome = "target"
            else:
                continue
            r = (fill - entry) / risk if is_buy else (entry - fill) / risk
            break

        rows.append(dict(id=t["id"], dateStart_utc=t["dateStart_utc"], side=side,
                         entryPrice=entry, initalSL=sl, target_price=target, target_r=target_r,
                         idealtp_r=idealtp_r[trades["id"].values == t["id"]][0],
                         amount=t["amount"], risk_price=risk,
                         actual_r=0.0, new_outcome=outcome, new_r=r))

    out = pd.DataFrame(rows)
    out["actual_pnl_usd"] = 0.0
    out["new_pnl_usd"] = out["new_r"] * out["risk_price"] * out["amount"]
    return out


def max_drawdown_r(series: np.ndarray) -> float:
    if len(series) == 0:
        return np.nan
    equity = np.cumsum(series)
    return float((np.maximum.accumulate(equity) - equity).max())


def summarize(result: pd.DataFrame) -> dict:
    resolved = result[result["new_outcome"] != "unresolved"]
    r = resolved["new_r"].values
    return {
        "n_total": len(result), "n_resolved": len(resolved),
        "n_unresolved": len(result) - len(resolved),
        "n_would_hit_target": int((resolved["new_outcome"] == "target").sum()),
        "n_would_hit_stop": int((resolved["new_outcome"] == "stop").sum()),
        "actual_total_r": 0.0, "actual_total_usd": 0.0,
        "new_total_r": float(r.sum()), "new_total_usd": float(resolved["new_pnl_usd"].sum()),
        "new_win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "new_expectancy": stats_utils.expectancy(r),
        "new_profit_factor": stats_utils.profit_factor(r),
        "actual_max_dd_r": 0.0,
        "new_max_dd_r": max_drawdown_r(r),
    }


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()

    result = recompute(trades, candles, target_r=2.1)
    result.to_csv("outputs/data/breakeven_2_1r_recompute.csv", index=False)

    print(f"breakeven trades with a profitable idealTP: {len(result)} / 93")
    print()
    for k, v in summarize(result).items():
        print(f"  {k}: {v}")
    print()
    print(result[["id", "dateStart_utc", "side", "idealtp_r", "new_outcome", "new_r"]].to_string(index=False))
