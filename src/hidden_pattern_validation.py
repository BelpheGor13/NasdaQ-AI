"""Phase 7: chronological train/test validation. Clusters are FIT on the
first config.TRAILING_TRAIN_FRACTION of trades only (2020-2022-ish) and
test trades are assigned to the NEAREST EXISTING centroid (hidden_pattern_
clustering.assign_clusters) -- never re-fit on test data, which would leak
the held-out period into the "discovered" pattern definitions themselves.
The best exit strategy per cluster is likewise chosen using ONLY train
trades, then applied as a fixed rule to test trades.
"""
import pandas as pd

from src import (config, exit_strategy_metrics, hidden_pattern_clustering as hpc,
                  hidden_pattern_features as hpf, stats_utils)


def chronological_split(feats: pd.DataFrame, train_fraction: float = config.TRAILING_TRAIN_FRACTION):
    ordered = feats.sort_values("dateStart_utc")
    split_at = int(len(ordered) * train_fraction)
    train_ids = set(ordered.iloc[:split_at]["id"])
    test_ids = set(ordered.iloc[split_at:]["id"])
    return train_ids, test_ids


def run_validation(sim: pd.DataFrame, feats: pd.DataFrame, scenario: str = "conservative") -> dict:
    train_ids, test_ids = chronological_split(feats)
    train_feats = feats[feats["id"].isin(train_ids)]
    test_feats = feats[feats["id"].isin(test_ids)]

    X_train = hpc.build_cluster_matrix(train_feats)
    fit = hpc.fit_clusters(X_train)
    train_cluster_ids = train_feats.loc[X_train.index, "id"].values
    train_cluster_map = pd.Series(fit["labels"], index=train_cluster_ids)

    X_test = hpc.build_cluster_matrix(test_feats)
    test_labels = hpc.assign_clusters(fit, X_test)
    test_cluster_ids = test_feats.loc[X_test.index, "id"].values
    test_cluster_map = pd.Series(test_labels, index=test_cluster_ids)

    train_table = exit_strategy_metrics.build_cluster_strategy_table(sim, train_cluster_map, scenario=scenario)
    best_per_cluster = exit_strategy_metrics.best_strategy_per_cluster(train_table)

    s = sim[sim["scenario"] == scenario].copy()

    def score_split(cluster_map, label):
        rows = []
        s_local = s.copy()
        s_local["cluster"] = s_local["id"].map(cluster_map)
        s_local = s_local.dropna(subset=["cluster"])
        for cluster, strategy in best_per_cluster.items():
            g = s_local[(s_local["cluster"] == cluster) & (s_local["strategy"] == strategy)]
            g_base = s_local[(s_local["cluster"] == cluster) & (s_local["strategy"] == "baseline")]
            r = g["exit_r"].dropna().values
            r_base = g_base["exit_r"].dropna().values
            rows.append({
                "split": label, "cluster": cluster, "strategy": strategy, "n": len(r),
                "expectancy": stats_utils.expectancy(r), "profit_factor": stats_utils.profit_factor(r),
                "baseline_expectancy": stats_utils.expectancy(r_base),
                "baseline_profit_factor": stats_utils.profit_factor(r_base),
            })
        return pd.DataFrame(rows)

    train_scored = score_split(train_cluster_map, "train")
    test_scored = score_split(test_cluster_map, "test")
    result = pd.concat([train_scored, test_scored], ignore_index=True)

    return {
        "k_train": fit["k"], "silhouette_train": fit["silhouette_score"],
        "n_train": len(train_ids), "n_test": len(test_ids),
        "best_per_cluster_train": best_per_cluster,
        "train_test_table": result,
    }


if __name__ == "__main__":
    from src import data_loading, exit_strategy_simulation

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)
    feats = hpf.build_hidden_pattern_features(trades, candles)

    result = run_validation(sim, feats)
    print(f"k (train)={result['k_train']}, silhouette={result['silhouette_train']:.3f}")
    print(f"n_train={result['n_train']}, n_test={result['n_test']}")
    print()
    print(result["train_test_table"].to_string(index=False))
