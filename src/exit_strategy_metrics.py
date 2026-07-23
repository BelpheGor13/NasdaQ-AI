"""Phase 3 deliverable: per-cluster x per-strategy performance table (win
rate, expected value, PF, Sharpe), with the best strategy per cluster
flagged. Reuses trailing_stop_metrics' Sharpe/max-drawdown helpers (same
R-multiple-based definitions) rather than redefining them.
"""
import numpy as np
import pandas as pd

from src import stats_utils, trailing_stop_metrics as tsm


def build_cluster_strategy_table(sim: pd.DataFrame, cluster_map: pd.Series, scenario: str = "conservative",
                                  min_cluster_size: int = 30) -> pd.DataFrame:
    """cluster_map: id -> cluster label (trades not in cluster_map's index
    are excluded -- e.g. rows dropped for NaN clustering features)."""
    s = sim[sim["scenario"] == scenario].copy()
    s["cluster"] = s["id"].map(cluster_map)
    s = s.dropna(subset=["cluster"])
    s = s.sort_values("dateStart_utc")

    rows = []
    for (cluster, strategy), group in s.groupby(["cluster", "strategy"]):
        r = group["exit_r"].dropna().values
        trades_per_year = len(group) / max(tsm._years_spanned(group["dateStart_utc"]), 1e-9)
        rows.append({
            "cluster": cluster,
            "strategy": strategy,
            "n": len(r),
            "low_confidence": len(group["id"].unique()) < min_cluster_size,
            "win_rate": float((r > 0).mean()) if len(r) else np.nan,
            "expectancy": stats_utils.expectancy(r),
            "profit_factor": stats_utils.profit_factor(r),
            "sharpe": tsm._sharpe(r, trades_per_year),
            "max_drawdown_r": tsm._max_drawdown_r(r),
        })
    out = pd.DataFrame(rows)

    out["best_in_cluster"] = False
    for cluster in out["cluster"].unique():
        mask = out["cluster"] == cluster
        best_idx = out.loc[mask, "profit_factor"].idxmax()
        out.loc[best_idx, "best_in_cluster"] = True
    return out.sort_values(["cluster", "profit_factor"], ascending=[True, False])


def build_global_strategy_table(sim: pd.DataFrame, scenario: str = "conservative") -> pd.DataFrame:
    """Same table but ignoring clusters entirely -- used for Phase 5's
    'best single global exit' comparison."""
    s = sim[sim["scenario"] == scenario]
    rows = []
    for strategy, group in s.groupby("strategy"):
        r = group["exit_r"].dropna().values
        trades_per_year = len(group) / max(tsm._years_spanned(group["dateStart_utc"]), 1e-9)
        rows.append({
            "strategy": strategy, "n": len(r),
            "win_rate": float((r > 0).mean()) if len(r) else np.nan,
            "expectancy": stats_utils.expectancy(r),
            "profit_factor": stats_utils.profit_factor(r),
            "sharpe": tsm._sharpe(r, trades_per_year),
            "max_drawdown_r": tsm._max_drawdown_r(r),
        })
    return pd.DataFrame(rows).sort_values("profit_factor", ascending=False)


def best_strategy_per_cluster(cluster_strategy_table: pd.DataFrame) -> pd.Series:
    best = cluster_strategy_table[cluster_strategy_table["best_in_cluster"]]
    return best.set_index("cluster")["strategy"]


if __name__ == "__main__":
    from src import data_loading, exit_strategy_simulation, hidden_pattern_features as hpf, hidden_pattern_clustering as hpc

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)

    feats = hpf.build_hidden_pattern_features(trades, candles)
    X = hpc.build_cluster_matrix(feats)
    fit = hpc.fit_clusters(X)
    cluster_map = pd.Series(fit["labels"], index=feats.loc[X.index, "id"].values)

    table = build_cluster_strategy_table(sim, cluster_map)
    for cluster in sorted(table["cluster"].unique()):
        print(f"\n=== cluster {cluster} ===")
        sub = table[table["cluster"] == cluster].head(6)
        print(sub[["strategy", "n", "win_rate", "expectancy", "profit_factor", "best_in_cluster"]].to_string(index=False))

    print("\n=== global (no clustering) ===")
    global_table = build_global_strategy_table(sim)
    print(global_table.head(6)[["strategy", "n", "expectancy", "profit_factor"]].to_string(index=False))
