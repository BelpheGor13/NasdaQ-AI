"""Phase 6: monthly PnL/win-rate comparison, yearly stability of each
cluster's best exit, and a drawdown profile comparison (original vs
hybrid).
"""
import numpy as np
import pandas as pd

from src import stats_utils, trailing_stop_metrics as tsm


def monthly_comparison(hybrid_df: pd.DataFrame) -> pd.DataFrame:
    monthly = hybrid_df.groupby("month").agg(
        original_pnl_usd=("baseline_pnl_usd", "sum"),
        hybrid_pnl_usd=("hybrid_pnl_usd", "sum"),
        n_trades=("id", "count"),
        win_rate_original=("baseline_r", lambda r: float((r > 0).mean())),
        win_rate_hybrid=("hybrid_r", lambda r: float((r > 0).mean())),
    ).reset_index().sort_values("month")
    monthly["difference_usd"] = monthly["hybrid_pnl_usd"] - monthly["original_pnl_usd"]
    monthly["direction"] = monthly["difference_usd"].apply(
        lambda d: "improved" if d > 0 else ("degraded" if d < 0 else "unchanged"))
    return monthly


def yearly_stability(sim: pd.DataFrame, cluster_map: pd.Series, best_strategy: pd.Series,
                      scenario: str = "conservative") -> pd.DataFrame:
    """For each cluster, is the SAME strategy that's globally 'best' also
    the best when each year (2020-2024) is scored on its own? A strategy
    whose ranking flips year to year isn't a stable rule (Critical Note 4)."""
    s = sim[sim["scenario"] == scenario].copy()
    s["cluster"] = s["id"].map(cluster_map)
    s = s.dropna(subset=["cluster"])

    rows = []
    for cluster, best in best_strategy.items():
        cluster_data = s[s["cluster"] == cluster]
        for year, year_group in cluster_data.groupby("year"):
            pf_by_strategy = year_group.groupby("strategy")["exit_r"].apply(
                lambda r: stats_utils.profit_factor(r.dropna().values))
            if len(pf_by_strategy) == 0:
                continue
            best_this_year = pf_by_strategy.idxmax()
            rows.append({
                "cluster": cluster, "year": year, "overall_best_strategy": best,
                "pf_of_overall_best_this_year": pf_by_strategy.get(best, np.nan),
                "best_strategy_this_year": best_this_year,
                "pf_of_best_this_year": pf_by_strategy[best_this_year],
                "consistent": best_this_year == best,
                "n": len(year_group["id"].unique()),
            })
    return pd.DataFrame(rows)


def drawdown_profile(hybrid_df: pd.DataFrame) -> dict:
    ordered = hybrid_df.sort_values("dateStart_utc")
    return {
        "baseline_max_dd_r": tsm._max_drawdown_r(ordered["baseline_r"].values),
        "hybrid_max_dd_r": tsm._max_drawdown_r(ordered["hybrid_r"].values),
    }


if __name__ == "__main__":
    from src import data_loading, exit_strategy_simulation, exit_strategy_metrics, hybrid_strategy, \
        hidden_pattern_features as hpf, hidden_pattern_clustering as hpc

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)

    feats = hpf.build_hidden_pattern_features(trades, candles)
    X = hpc.build_cluster_matrix(feats)
    fit = hpc.fit_clusters(X)
    cluster_map = pd.Series(fit["labels"], index=feats.loc[X.index, "id"].values)

    table = exit_strategy_metrics.build_cluster_strategy_table(sim, cluster_map)
    best = exit_strategy_metrics.best_strategy_per_cluster(table)
    hybrid_df = hybrid_strategy.build_hybrid_exit_r(sim, cluster_map, best)

    monthly = monthly_comparison(hybrid_df)
    print(monthly.to_string(index=False))
    print()
    print(monthly["direction"].value_counts())

    print("\n=== yearly stability ===")
    yearly = yearly_stability(sim, cluster_map, best)
    print(yearly.to_string(index=False))

    print("\n=== drawdown profile ===")
    print(drawdown_profile(hybrid_df))
