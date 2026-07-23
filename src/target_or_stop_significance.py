"""Paired significance test: does letting a trade run untouched to its
original stop-loss or a fixed target (no early manual exit, no trailing)
perform differently from what actually happened (the real, partly
discretionary exit)? Same 789 trades, paired t-test + Wilcoxon (the latter
as a robustness cross-check, consistent with this project's convention of
never relying on a single test).
"""
import numpy as np
import pandas as pd
from scipy import stats


def paired_tests(baseline_r: np.ndarray, target_r: np.ndarray) -> dict:
    t_stat, t_p = stats.ttest_rel(target_r, baseline_r)
    w_stat, w_p = stats.wilcoxon(target_r, baseline_r)
    diff = target_r - baseline_r
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) != 0 else 0.0

    return {
        "n": len(baseline_r),
        "mean_baseline_r": float(np.mean(baseline_r)),
        "mean_target_r": float(np.mean(target_r)),
        "mean_diff": float(np.mean(diff)),
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "cohens_d": d,
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p_value": float(w_p),
    }


def significance_for_scenario(sim: pd.DataFrame, scenario: str = "conservative",
                               strategy: str = "fixed_tp_idealTP") -> dict:
    base = sim[(sim["scenario"] == scenario) & (sim["strategy"] == "baseline")].sort_values("id")
    targ = sim[(sim["scenario"] == scenario) & (sim["strategy"] == strategy)].sort_values("id")
    assert list(base["id"]) == list(targ["id"]), "trade ids must align for a paired test"
    result = paired_tests(base["exit_r"].values, targ["exit_r"].values)
    result["scenario"] = scenario
    result["strategy"] = strategy
    return result


if __name__ == "__main__":
    from src import data_loading, exit_strategy_simulation

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)

    for scenario in ("conservative", "aggressive"):
        result = significance_for_scenario(sim, scenario=scenario)
        print(f"=== {scenario} ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
