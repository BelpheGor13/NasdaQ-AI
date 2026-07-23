"""Phase 4: Monte Carlo validation of each cluster's best-performing exit
strategy -- bootstrap resampling for a PF confidence interval, plus a
trade-order shuffle stress test.

Note on the shuffle test: profit factor and expectancy are SUMS/MEANS over
a fixed set of per-trade R-values, so reordering the same 789 (or per-
cluster) trades can never change them -- the best strategy's aggregate
edge over baseline is mathematically invariant to order. What genuinely
IS order-dependent is the DRAWDOWN path, so that's what the shuffle test
reports: across 1,000 random orderings of the same trades, how bad could
the max drawdown have looked, purely from sequence luck.
"""
import numpy as np
import pandas as pd

from src import config, stats_utils


def bootstrap_pf_ci(strategy_r: np.ndarray, baseline_r: np.ndarray, n_boot: int = 1000,
                     seed: int = config.RANDOM_SEED) -> dict:
    n = len(strategy_r)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (n_boot, n))

    def pf_batch(values, idx_matrix):
        samples = values[idx_matrix]
        wins = np.where(samples > 0, samples, 0).sum(axis=1)
        losses = np.where(samples < 0, -samples, 0).sum(axis=1)
        return np.divide(wins, losses, out=np.full(n_boot, np.inf), where=losses > 0)

    strat_pf_dist = pf_batch(strategy_r, idx)
    baseline_pf = stats_utils.profit_factor(baseline_r)

    finite = strat_pf_dist[np.isfinite(strat_pf_dist)]
    lo, hi = np.percentile(finite, [5, 95]) if len(finite) else (np.nan, np.nan)

    if lo > baseline_pf and hi > baseline_pf:
        classification = "robust"
    elif lo <= baseline_pf <= hi:
        classification = "borderline" if (hi - baseline_pf) / max(hi - lo, 1e-9) < 0.9 else "not_robust"
    else:
        classification = "not_robust"

    return {
        "mean_pf": float(np.mean(finite)) if len(finite) else np.nan,
        "ci_lo_5th": float(lo), "ci_hi_95th": float(hi),
        "baseline_pf": float(baseline_pf),
        "classification": classification,
    }


def shuffle_drawdown_stress_test(strategy_r: np.ndarray, baseline_r: np.ndarray, n_shuffle: int = 1000,
                                  seed: int = config.RANDOM_SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(strategy_r)

    def dd_dist(values):
        dds = np.empty(n_shuffle)
        for k in range(n_shuffle):
            perm = rng.permutation(n)
            equity = np.cumsum(values[perm])
            running_peak = np.maximum.accumulate(equity)
            dds[k] = (running_peak - equity).max()
        return dds

    strat_dd = dd_dist(strategy_r)
    base_dd = dd_dist(baseline_r)

    return {
        "strategy_total_r_invariant_to_order": float(strategy_r.sum()),
        "baseline_total_r_invariant_to_order": float(baseline_r.sum()),
        "strategy_dd_median": float(np.median(strat_dd)),
        "strategy_dd_95th_pct": float(np.percentile(strat_dd, 95)),
        "baseline_dd_median": float(np.median(base_dd)),
        "baseline_dd_95th_pct": float(np.percentile(base_dd, 95)),
        "pct_shuffles_strategy_dd_smaller": float((strat_dd < base_dd).mean()),
    }


def run_mc_for_best_per_cluster(sim: pd.DataFrame, cluster_map: pd.Series, best_strategy: pd.Series,
                                 scenario: str = "conservative") -> pd.DataFrame:
    s = sim[sim["scenario"] == scenario].copy()
    s["cluster"] = s["id"].map(cluster_map)
    s = s.dropna(subset=["cluster"])

    rows = []
    for cluster, strategy in best_strategy.items():
        group_strategy = s[(s["cluster"] == cluster) & (s["strategy"] == strategy)].sort_values("id")
        group_baseline = s[(s["cluster"] == cluster) & (s["strategy"] == "baseline")].sort_values("id")

        boot = bootstrap_pf_ci(group_strategy["exit_r"].values, group_baseline["exit_r"].values)
        shuffle = shuffle_drawdown_stress_test(group_strategy["exit_r"].values, group_baseline["exit_r"].values)
        rows.append({"cluster": cluster, "best_strategy": strategy, "n": len(group_strategy), **boot, **shuffle})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src import data_loading, exit_strategy_simulation, exit_strategy_metrics, hidden_pattern_features as hpf, \
        hidden_pattern_clustering as hpc

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)

    feats = hpf.build_hidden_pattern_features(trades, candles)
    X = hpc.build_cluster_matrix(feats)
    fit = hpc.fit_clusters(X)
    cluster_map = pd.Series(fit["labels"], index=feats.loc[X.index, "id"].values)

    table = exit_strategy_metrics.build_cluster_strategy_table(sim, cluster_map)
    best = exit_strategy_metrics.best_strategy_per_cluster(table)
    print("best per cluster:", best.to_dict())

    mc = run_mc_for_best_per_cluster(sim, cluster_map, best)
    print(mc.to_string(index=False))
