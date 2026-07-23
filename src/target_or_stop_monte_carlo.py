"""Paired bootstrap robustness check (same convention as
trailing_stop_monte_carlo.py): resample the SAME 789 trades with
replacement so the baseline and fixed_tp_idealTP profit factors are always
compared trade-for-trade, not as two independent distributions.
"""
import numpy as np
import pandas as pd

from src import config, stats_utils


def bootstrap_pf_ci(target_r: np.ndarray, baseline_r: np.ndarray, n_boot: int = 2000,
                     seed: int = config.RANDOM_SEED) -> dict:
    n = len(target_r)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))

    target_samples = target_r[idx]
    baseline_samples = baseline_r[idx]

    def pf_batch(samples):
        wins = np.where(samples > 0, samples, 0).sum(axis=1)
        losses = np.where(samples < 0, -samples, 0).sum(axis=1)
        return np.divide(wins, losses, out=np.full(n_boot, np.inf), where=losses > 0)

    target_pf_dist = pf_batch(target_samples)
    baseline_pf_dist = pf_batch(baseline_samples)

    observed_baseline_pf = stats_utils.profit_factor(baseline_r)
    finite = np.isfinite(target_pf_dist)
    lo, hi = np.percentile(target_pf_dist[finite], [5, 95])
    pct_beats_baseline = float((target_pf_dist > baseline_pf_dist).mean())

    return {
        "mean_pf": float(np.mean(target_pf_dist[finite])),
        "ci_lo_5th": float(lo),
        "ci_hi_95th": float(hi),
        "observed_baseline_pf": float(observed_baseline_pf),
        "ci_crosses_baseline": bool(lo <= observed_baseline_pf <= hi),
        "pct_boot_iters_beats_baseline": pct_beats_baseline,
    }


if __name__ == "__main__":
    from src import data_loading, exit_strategy_simulation

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)

    for scenario in ("conservative", "aggressive"):
        base = sim[(sim["scenario"] == scenario) & (sim["strategy"] == "baseline")].sort_values("id")
        targ = sim[(sim["scenario"] == scenario) & (sim["strategy"] == "fixed_tp_idealTP")].sort_values("id")
        result = bootstrap_pf_ci(targ["exit_r"].values, base["exit_r"].values)
        print(f"=== {scenario} ===")
        for k, v in result.items():
            print(f"  {k}: {v}")
