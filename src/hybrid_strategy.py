"""Phase 5: the regime-conditional hybrid strategy -- apply each trade's
OWN cluster's best exit strategy, and compare the result against (a) the
original baseline and (b) the single best exit strategy applied uniformly
to every trade regardless of cluster. If cluster-conditional exiting adds
nothing beyond the best global exit, the numbers here will show it plainly
(same invariant as everywhere else in this project: don't force a
narrative -- see Critical Note 5).
"""
import numpy as np
import pandas as pd

from src import stats_utils, trailing_stop_metrics as tsm, trailing_stop_significance as tss


def build_hybrid_exit_r(sim: pd.DataFrame, cluster_map: pd.Series, best_strategy: pd.Series,
                         scenario: str = "conservative") -> pd.DataFrame:
    """One row per clustered trade: id, cluster, hybrid_r (this trade's own
    cluster's best-strategy result), baseline_r, plus dateStart_utc/month/year
    for the downstream monthly/yearly breakdowns."""
    s = sim[sim["scenario"] == scenario].copy()
    s["cluster"] = s["id"].map(cluster_map)
    s = s.dropna(subset=["cluster"])

    baseline = s[s["strategy"] == "baseline"].set_index("id")

    pieces = []
    for cluster, strategy in best_strategy.items():
        chosen = s[(s["cluster"] == cluster) & (s["strategy"] == strategy)]
        pieces.append(chosen)
    hybrid = pd.concat(pieces).sort_values("id")

    out = hybrid[["id", "cluster", "exit_r", "exit_pnl_usd", "dateStart_utc", "month", "year",
                  "risk_price", "amount"]].rename(columns={"exit_r": "hybrid_r", "exit_pnl_usd": "hybrid_pnl_usd"})
    out["hybrid_strategy"] = hybrid["strategy"].values
    out["baseline_r"] = baseline.loc[out["id"], "orig_r"].values
    out["baseline_pnl_usd"] = baseline.loc[out["id"], "exit_pnl_usd"].values
    return out.reset_index(drop=True)


def _metrics_row(r_values: np.ndarray, label: str, trades_per_year: float) -> dict:
    r = np.asarray(r_values, dtype=float)
    return {
        "metric_set": label, "n": len(r),
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "expectancy": stats_utils.expectancy(r),
        "profit_factor": stats_utils.profit_factor(r),
        "sharpe": tsm._sharpe(r, trades_per_year),
        "max_drawdown_r": tsm._max_drawdown_r(r),
        "total_pnl_r": float(r.sum()),
    }


def compare_hybrid_vs_global(sim: pd.DataFrame, hybrid_df: pd.DataFrame, global_table: pd.DataFrame,
                              scenario: str = "conservative") -> tuple:
    """Returns (comparison_table, best_global_strategy_name)."""
    trades_per_year = len(hybrid_df) / max(tsm._years_spanned(hybrid_df["dateStart_utc"]), 1e-9)
    best_global_name = global_table[global_table["strategy"] != "baseline"].iloc[0]["strategy"]

    s = sim[sim["scenario"] == scenario]
    global_r = s[(s["strategy"] == best_global_name) & (s["id"].isin(hybrid_df["id"]))] \
        .set_index("id").loc[hybrid_df["id"], "exit_r"].values

    rows = [
        {**_metrics_row(hybrid_df["baseline_r"].values, "baseline", trades_per_year), "exit_strategy_used": None},
        {**_metrics_row(global_r, "best_single_global_exit", trades_per_year), "exit_strategy_used": best_global_name},
        {**_metrics_row(hybrid_df["hybrid_r"].values, "hybrid_cluster_conditional", trades_per_year),
         "exit_strategy_used": "varies per cluster (see cluster table)"},
    ]
    out = pd.DataFrame(rows)

    baseline_pf = out.loc[out["metric_set"] == "baseline", "profit_factor"].values[0]
    baseline_exp = out.loc[out["metric_set"] == "baseline", "expectancy"].values[0]
    out["pf_improvement_pct"] = (out["profit_factor"] - baseline_pf) / baseline_pf * 100
    out["expectancy_improvement_pct"] = (out["expectancy"] - baseline_exp) / abs(baseline_exp) * 100
    return out, best_global_name, global_r


def hybrid_vs_baseline_significance(hybrid_df: pd.DataFrame) -> dict:
    return tss.paired_ttest(hybrid_df["baseline_r"].values, hybrid_df["hybrid_r"].values)


def hybrid_vs_global_significance(hybrid_df: pd.DataFrame, global_r: np.ndarray) -> dict:
    return tss.paired_ttest(global_r, hybrid_df["hybrid_r"].values)


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
    global_table = exit_strategy_metrics.build_global_strategy_table(sim)

    hybrid_df = build_hybrid_exit_r(sim, cluster_map, best)
    comparison, best_global_name, global_r = compare_hybrid_vs_global(sim, hybrid_df, global_table)
    print(f"best single global exit: {best_global_name}")
    print(comparison.to_string(index=False))

    print("\n=== hybrid vs baseline (paired t-test) ===")
    sig1 = hybrid_vs_baseline_significance(hybrid_df)
    for k, v in sig1.items():
        print(f"  {k}: {v}")

    print("\n=== hybrid vs best-single-global-exit (paired t-test) ===")
    sig2 = hybrid_vs_global_significance(hybrid_df, global_r)
    for k, v in sig2.items():
        print(f"  {k}: {v}")
