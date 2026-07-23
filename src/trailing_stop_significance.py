"""Paired significance test (deliverable 3): H0 = mean(r_original) =
mean(r_trailing) for the single best trailing configuration, on the SAME
789 trades (paired, not independent samples) -- a paired t-test is the
correct test here rather than Welch's (used elsewhere in this project for
independent pattern-vs-population comparisons).
"""
import numpy as np
import pandas as pd
from scipy import stats


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if diff.std(ddof=1) == 0:
        return 0.0
    return float(diff.mean() / diff.std(ddof=1))


def paired_ttest(orig_r: np.ndarray, trail_r: np.ndarray) -> dict:
    t_stat, p_value = stats.ttest_rel(trail_r, orig_r)
    d = cohens_d_paired(trail_r, orig_r)

    if p_value < 0.05 and abs(d) >= 0.2:
        interpretation = "significant_and_meaningful"
    elif p_value < 0.05:
        interpretation = "significant_but_tiny_effect"
    else:
        interpretation = "not_significant"

    return {
        "n": len(orig_r),
        "mean_orig_r": float(np.mean(orig_r)),
        "mean_trail_r": float(np.mean(trail_r)),
        "mean_diff": float(np.mean(trail_r) - np.mean(orig_r)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": d,
        "interpretation": interpretation,
    }


def significance_for_best_config(sim: pd.DataFrame, best_pct: float, scenario: str = "conservative") -> dict:
    g = sim[(sim["scenario"] == scenario) & (sim["pct"] == best_pct)].sort_values("id")
    result = paired_ttest(g["orig_r"].values, g["trail_r"].values)
    result["pct"] = best_pct
    return result


if __name__ == "__main__":
    from src import data_loading, trailing_stop_simulation, trailing_stop_metrics

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = trailing_stop_simulation.simulate_trailing_stops(trades, candles)
    summary = trailing_stop_metrics.build_summary_table(sim, scenario="conservative")

    best_pct = summary[summary["config"] != "baseline (no trailing)"].sort_values(
        "profit_factor", ascending=False).iloc[0]["pct"]

    result = significance_for_best_config(sim, best_pct)
    for k, v in result.items():
        print(f"{k}: {v}")
