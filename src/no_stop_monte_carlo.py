"""Statistical tests and Monte Carlo robustness checks for the no-stop-loss
scenario. Both a paired t-test and a paired Wilcoxon signed-rank test are
reported (not just one) -- with outcomes ranging to -400R, the per-trade
difference distribution is heavily skewed by a handful of extreme trades,
so the rank-based Wilcoxon is the more trustworthy of the two, but the
t-test is kept alongside for direct comparability with the rest of this
project's convention.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src import config


def paired_tests(orig_r: np.ndarray, no_sl_r: np.ndarray) -> dict:
    t_stat, t_p = stats.ttest_rel(no_sl_r, orig_r)
    w_stat, w_p = stats.wilcoxon(no_sl_r, orig_r)
    diff = no_sl_r - orig_r
    d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) != 0 else 0.0
    return {
        "n": len(orig_r),
        "mean_orig_r": float(np.mean(orig_r)),
        "mean_no_sl_r": float(np.mean(no_sl_r)),
        "median_orig_r": float(np.median(orig_r)),
        "median_no_sl_r": float(np.median(no_sl_r)),
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "cohens_d": d,
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p_value": float(w_p),
    }


def bootstrap_equity_difference(orig_r: np.ndarray, no_sl_r: np.ndarray, n_boot: int = config.MC_ITERATIONS,
                                 seed: int = config.RANDOM_SEED) -> dict:
    """Paired bootstrap (same resample indices for both series): for each
    resample, total cumulative R with vs without the stop, and the
    difference. Reports how often removing the stop would have left the
    account worse off in total, not just per-trade.
    """
    n = len(orig_r)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))

    orig_totals = orig_r[idx].sum(axis=1)
    no_sl_totals = no_sl_r[idx].sum(axis=1)
    diff = no_sl_totals - orig_totals

    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {
        "observed_total_orig_r": float(orig_r.sum()),
        "observed_total_no_sl_r": float(no_sl_r.sum()),
        "mean_bootstrap_diff": float(diff.mean()),
        "diff_ci_lo_2.5pct": float(lo),
        "diff_ci_hi_97.5pct": float(hi),
        "pct_boot_iters_no_sl_worse": float((diff < 0).mean()),
    }


def reshuffled_drawdown_sensitivity(no_sl_r: np.ndarray, n_shuffles: int = 2000,
                                     seed: int = config.RANDOM_SEED) -> dict:
    """Total cumulative R doesn't depend on trade order, but max drawdown
    DOES (path-dependent). Shuffles the same 789 no-stop-loss outcomes into
    random orders to see whether the actual chronological sequence was
    unusually unlucky, or whether severe drawdown is typical regardless of
    ordering.
    """
    rng = np.random.default_rng(seed)
    n = len(no_sl_r)
    drawdowns = np.empty(n_shuffles)
    for i in range(n_shuffles):
        shuffled = rng.permutation(no_sl_r)
        equity = np.cumsum(shuffled)
        running_peak = np.maximum.accumulate(equity)
        drawdowns[i] = (running_peak - equity).max()

    actual_equity = np.cumsum(no_sl_r)
    actual_dd = float((np.maximum.accumulate(actual_equity) - actual_equity).max())

    return {
        "actual_chronological_max_drawdown_r": actual_dd,
        "shuffled_median_max_drawdown_r": float(np.median(drawdowns)),
        "shuffled_p5_max_drawdown_r": float(np.percentile(drawdowns, 5)),
        "shuffled_p95_max_drawdown_r": float(np.percentile(drawdowns, 95)),
        "pct_shuffles_worse_than_actual": float((drawdowns > actual_dd).mean()),
    }


if __name__ == "__main__":
    from src import data_loading, no_stop_simulation

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = no_stop_simulation.simulate_no_stop(trades, candles).dropna(subset=["orig_r", "no_sl_final_r"])
    sim = sim.sort_values("dateStart_utc")

    print("=== paired tests ===")
    for k, v in paired_tests(sim["orig_r"].values, sim["no_sl_final_r"].values).items():
        print(f"{k}: {v}")

    print("\n=== bootstrap equity difference ===")
    for k, v in bootstrap_equity_difference(sim["orig_r"].values, sim["no_sl_final_r"].values).items():
        print(f"{k}: {v}")

    print("\n=== reshuffled drawdown sensitivity ===")
    for k, v in reshuffled_drawdown_sensitivity(sim["no_sl_final_r"].values).items():
        print(f"{k}: {v}")
