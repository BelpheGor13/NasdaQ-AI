"""End-to-end hidden-pattern discovery + regime-conditional exit
optimization pipeline (test-only; see hidden-patterns-exit-optimization-
prompt.md). Never touches analytics_1.csv or any original-strategy logic.

Run with: python -m src.hidden_pattern_main
"""
import time

import pandas as pd

from src import (
    config, data_loading, exit_strategy_simulation, exit_strategy_metrics,
    exit_strategy_monte_carlo, hybrid_strategy, hidden_pattern_features as hpf,
    hidden_pattern_clustering as hpc, hidden_pattern_monthly, hidden_pattern_validation,
    hidden_pattern_visualizations, hidden_pattern_reporting,
)


def main():
    t0 = time.time()
    config.DATA_OUT.mkdir(parents=True, exist_ok=True)
    config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    config.REPORTS_OUT.mkdir(parents=True, exist_ok=True)

    print("[1/8] Loading data & building Phase-1 entry-context features...")
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = hpf.build_hidden_pattern_features(trades, candles)

    print("[2/8] Phase 2: clustering entry contexts...")
    X = hpc.build_cluster_matrix(feats)
    fit = hpc.fit_clusters(X)
    cluster_map = pd.Series(fit["labels"], index=feats.loc[X.index, "id"].values)
    cluster_profile = hpc.name_and_profile_clusters(feats, fit["labels"], X.index)
    cluster_profile.to_csv(config.DATA_OUT / "hidden_pattern_cluster_profile.csv", index=False)
    print(f"  k={fit['k']}, silhouette={fit['silhouette_score']:.3f}, sizes={cluster_profile['size'].tolist()}")

    print("[3/8] Phase 3: simulating every exit strategy (no look-ahead)...")
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)
    sim.drop(columns=["dateStart_utc", "dateEnd_utc"]).to_csv(
        config.DATA_OUT / "hidden_pattern_exit_simulation.csv", index=False)

    cluster_table = exit_strategy_metrics.build_cluster_strategy_table(sim, cluster_map)
    cluster_table.to_csv(config.DATA_OUT / "hidden_pattern_cluster_strategy_table.csv", index=False)
    best_per_cluster = exit_strategy_metrics.best_strategy_per_cluster(cluster_table)
    global_table = exit_strategy_metrics.build_global_strategy_table(sim)

    print("[4/8] Phase 4: Monte Carlo + shuffle stress test per cluster...")
    mc = exit_strategy_monte_carlo.run_mc_for_best_per_cluster(sim, cluster_map, best_per_cluster)
    mc.to_csv(config.DATA_OUT / "hidden_pattern_monte_carlo.csv", index=False)

    print("[5/8] Phase 5: hybrid strategy vs baseline vs best global exit...")
    hybrid_df = hybrid_strategy.build_hybrid_exit_r(sim, cluster_map, best_per_cluster)
    comparison, best_global_name, global_r = hybrid_strategy.compare_hybrid_vs_global(sim, hybrid_df, global_table)
    sig_baseline = hybrid_strategy.hybrid_vs_baseline_significance(hybrid_df)
    sig_global = hybrid_strategy.hybrid_vs_global_significance(hybrid_df, global_r)
    comparison.to_csv(config.DATA_OUT / "hidden_pattern_hybrid_comparison.csv", index=False)

    print("[6/8] Phase 6: monthly/yearly breakdown & drawdown...")
    monthly = hidden_pattern_monthly.monthly_comparison(hybrid_df)
    yearly = hidden_pattern_monthly.yearly_stability(sim, cluster_map, best_per_cluster)
    dd = hidden_pattern_monthly.drawdown_profile(hybrid_df)
    monthly.to_csv(config.DATA_OUT / "hidden_pattern_monthly.csv", index=False)
    yearly.to_csv(config.DATA_OUT / "hidden_pattern_yearly.csv", index=False)

    print("[7/8] Phase 7: train/test holdout validation...")
    validation = hidden_pattern_validation.run_validation(sim, feats)
    validation["train_test_table"].to_csv(config.DATA_OUT / "hidden_pattern_validation.csv", index=False)

    print("[8/8] Visualizations & Arabic report...")
    hidden_pattern_visualizations.plot_cluster_strategy_pf(cluster_table)
    hidden_pattern_visualizations.plot_monthly_comparison(monthly)
    hidden_pattern_visualizations.plot_bootstrap_ci(mc)
    hidden_pattern_visualizations.plot_yearly_stability(yearly)
    hidden_pattern_visualizations.plot_drawdown_comparison(dd)

    results = {
        "cluster_profile": cluster_profile,
        "cluster_table": cluster_table,
        "monte_carlo": mc,
        "comparison": comparison,
        "best_global_name": best_global_name,
        "sig_vs_baseline": sig_baseline,
        "sig_vs_global": sig_global,
        "monthly": monthly,
        "yearly": yearly,
        "drawdown": dd,
        "validation": validation,
        "silhouette": fit["silhouette_score"],
        "k": fit["k"],
    }

    report_text = hidden_pattern_reporting.build_report(results)
    config.HIDDEN_PATTERN_REPORT.write_text(report_text, encoding="utf-8")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Report: {config.HIDDEN_PATTERN_REPORT}")
    print(f"Data: {config.DATA_OUT}")
    print(f"Figures: {config.FIGURES_OUT}")


if __name__ == "__main__":
    main()
