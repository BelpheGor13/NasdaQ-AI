"""Comparison metrics between the real, stop-protected outcomes and the
hypothetical no-stop-loss ride. Both mean AND median are reported
throughout -- with outcomes ranging up to -400R, the mean is dominated by
a handful of extreme trades (real market crashes hit during the 30-day
window) and would be misleading on its own.
"""
import numpy as np
import pandas as pd

from src import config, stats_utils


def _years_spanned(dates: pd.Series) -> float:
    span_days = (dates.max() - dates.min()).total_seconds() / 86400
    return max(span_days / 365.25, 1e-9)


def _max_drawdown_r(r_values_chronological: np.ndarray) -> float:
    if len(r_values_chronological) == 0:
        return np.nan
    equity = np.cumsum(r_values_chronological)
    running_peak = np.maximum.accumulate(equity)
    return float((running_peak - equity).max())


def summary_comparison(sim: pd.DataFrame) -> pd.DataFrame:
    s = sim.dropna(subset=["orig_r", "no_sl_final_r"]).sort_values("dateStart_utc")
    trades_per_year = len(s) / _years_spanned(s["dateStart_utc"])

    rows = []
    for label, col in [("baseline (with stop-loss)", "orig_r"), ("no stop-loss (ride to 30d deadline)", "no_sl_final_r")]:
        r = s[col].values
        rows.append({
            "config": label,
            "n": len(r),
            "mean": float(np.mean(r)),
            "median": float(np.median(r)),
            "win_rate": float((r > 0).mean()),
            "profit_factor": stats_utils.profit_factor(r),
            "max_drawdown_r": _max_drawdown_r(r),
            "worst_single_trade_r": float(np.min(r)),
        })
    return pd.DataFrame(rows)


def tail_risk_summary(sim: pd.DataFrame) -> dict:
    s = sim.dropna(subset=["no_sl_worst_r_reached"])
    worst = s["no_sl_worst_r_reached"].values

    n_catastrophic = int(s["catastrophic"].sum())
    n_recovered_after_catastrophic = int(((s["catastrophic"]) & (s["no_sl_final_r"] > 0)).sum())

    return {
        "n": len(s),
        "pct_catastrophic": n_catastrophic / len(s) if len(s) else np.nan,
        "n_catastrophic": n_catastrophic,
        "n_recovered_after_catastrophic": n_recovered_after_catastrophic,
        "median_worst_r_reached": float(np.median(worst)),
        "p5_worst_r_reached": float(np.percentile(worst, 5)),
        "p1_worst_r_reached": float(np.percentile(worst, 1)),
        "min_worst_r_reached": float(np.min(worst)),
        "threshold_used": config.NO_STOP_CATASTROPHIC_R_THRESHOLD,
        "n_data_censored": int(sim["data_censored"].sum()),
    }


def stop_outs_that_would_have_recovered(sim: pd.DataFrame) -> dict:
    """Of the trades that were ACTUAL -1R stop-outs, how many would have
    ended up profitable if simply given more room/time (no stop)? This is
    the "counter-argument" half of the story -- removing the stop isn't
    purely bad on every individual trade, which is exactly why the tail
    risk below matters so much (it's not that every trade suffers; it's
    that a few trades suffer catastrophically).
    """
    stopped = sim[(sim["orig_r"] <= -0.99) & sim["no_sl_final_r"].notna()]
    would_recover = stopped[stopped["no_sl_final_r"] > 0]
    return {
        "n_original_stop_outs": len(stopped),
        "n_would_have_recovered": len(would_recover),
        "pct_would_have_recovered": len(would_recover) / len(stopped) if len(stopped) else np.nan,
        "median_recovery_r": float(would_recover["no_sl_final_r"].median()) if len(would_recover) else np.nan,
    }


if __name__ == "__main__":
    from src import data_loading, no_stop_simulation

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = no_stop_simulation.simulate_no_stop(trades, candles)

    print(summary_comparison(sim).to_string(index=False))
    print()
    for k, v in tail_risk_summary(sim).items():
        print(f"{k}: {v}")
    print()
    for k, v in stop_outs_that_would_have_recovered(sim).items():
        print(f"{k}: {v}")
