"""Monte Carlo robustness check (deliverable 2) for the top-3 trailing-stop
configurations (ranked by profit factor): bootstrap resample the SAME 789
trades 1,000 times with replacement, recompute PF each time, and report the
5th/95th percentile CI. The baseline (no-trailing) PF is bootstrapped with
the SAME resample indices each iteration so the comparison is paired trade-
for-trade, not just two independent distributions.
"""
import numpy as np
import pandas as pd

from src import config, stats_utils


def select_top_configs(summary_table: pd.DataFrame, n: int = 3, rank_by: str = "profit_factor") -> list:
    non_baseline = summary_table[summary_table["config"] != "baseline (no trailing)"]
    return non_baseline.sort_values(rank_by, ascending=False).head(n)["pct"].tolist()


def bootstrap_pf_ci(trail_r: np.ndarray, orig_r: np.ndarray, n_boot: int = 1000,
                     seed: int = config.RANDOM_SEED) -> dict:
    """trail_r and orig_r must be aligned (same trade order)."""
    n = len(trail_r)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))

    trail_samples = trail_r[idx]
    orig_samples = orig_r[idx]

    def pf_batch(samples):
        wins = np.where(samples > 0, samples, 0).sum(axis=1)
        losses = np.where(samples < 0, -samples, 0).sum(axis=1)
        return np.divide(wins, losses, out=np.full(n_boot, np.inf), where=losses > 0)

    trail_pf_dist = pf_batch(trail_samples)
    orig_pf_dist = pf_batch(orig_samples)

    observed_baseline_pf = stats_utils.profit_factor(orig_r)
    lo, hi = np.percentile(trail_pf_dist[np.isfinite(trail_pf_dist)], [5, 95])
    crosses_baseline = bool(lo <= observed_baseline_pf <= hi)
    pct_beats_baseline = float((trail_pf_dist > orig_pf_dist).mean())

    return {
        "mean_pf": float(np.mean(trail_pf_dist[np.isfinite(trail_pf_dist)])),
        "ci_lo_5th": float(lo),
        "ci_hi_95th": float(hi),
        "observed_baseline_pf": float(observed_baseline_pf),
        "ci_crosses_baseline": crosses_baseline,
        "pct_boot_iters_config_beats_baseline": pct_beats_baseline,
        "robust_verdict": "not_robust_ci_crosses_baseline" if crosses_baseline else "robust_ci_does_not_cross_baseline",
    }


def run_mc_for_top_configs(sim: pd.DataFrame, summary_table: pd.DataFrame, scenario: str = "conservative",
                            n: int = 3, n_boot: int = 1000) -> pd.DataFrame:
    top_pcts = select_top_configs(summary_table, n=n)
    s = sim[sim["scenario"] == scenario]

    rows = []
    for pct in top_pcts:
        g = s[s["pct"] == pct].sort_values("id")
        result = bootstrap_pf_ci(g["trail_r"].values, g["orig_r"].values, n_boot=n_boot)
        result["pct"] = pct
        rows.append(result)

    out = pd.DataFrame(rows)
    cols = ["pct", "mean_pf", "ci_lo_5th", "ci_hi_95th", "observed_baseline_pf",
            "ci_crosses_baseline", "pct_boot_iters_config_beats_baseline", "robust_verdict"]
    return out[cols]


if __name__ == "__main__":
    from src import data_loading, trailing_stop_simulation, trailing_stop_metrics

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = trailing_stop_simulation.simulate_trailing_stops(trades, candles)
    summary = trailing_stop_metrics.build_summary_table(sim, scenario="conservative")

    mc = run_mc_for_top_configs(sim, summary)
    print(mc.to_string(index=False))
