"""End-to-end trailing-stop optimization & Monte Carlo validation pipeline
(test-only; see trailing-stop-optimization-prompt.md). Never touches
analytics_1.csv or any original-strategy logic.

Run with: python -m src.trailing_stop_main
"""
import time

from src import (
    config, data_loading, trailing_stop_simulation, trailing_stop_metrics,
    trailing_stop_monte_carlo, trailing_stop_significance, trailing_stop_monthly,
    trailing_stop_validation, trailing_stop_regime, trailing_stop_visualizations,
    trailing_stop_reporting,
)


def main():
    t0 = time.time()
    config.DATA_OUT.mkdir(parents=True, exist_ok=True)
    config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    config.REPORTS_OUT.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading data & simulating trailing stops (no look-ahead)...")
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = trailing_stop_simulation.simulate_trailing_stops(trades, candles)
    sim.drop(columns=["dateStart_utc", "dateEnd_utc"]).to_csv(
        config.DATA_OUT / "trailing_stop_simulation.csv", index=False)

    print("[2/7] Building per-configuration metrics (grid sweep)...")
    summary_cons = trailing_stop_metrics.build_summary_table(sim, scenario="conservative")
    summary_agg = trailing_stop_metrics.build_summary_table(sim, scenario="aggressive")
    summary_cons.to_csv(config.DATA_OUT / "trailing_stop_summary_conservative.csv", index=False)
    summary_agg.to_csv(config.DATA_OUT / "trailing_stop_summary_aggressive.csv", index=False)

    best_pct = summary_cons[summary_cons["config"] != "baseline (no trailing)"].sort_values(
        "profit_factor", ascending=False).iloc[0]["pct"]
    print(f"  best config by profit factor (conservative): {best_pct*100:.0f}%")

    print("[3/7] Monte Carlo bootstrap (top 3 configs)...")
    mc = trailing_stop_monte_carlo.run_mc_for_top_configs(sim, summary_cons)
    mc.to_csv(config.DATA_OUT / "trailing_stop_monte_carlo.csv", index=False)

    print("[4/7] Paired significance test (best config vs original)...")
    sig = trailing_stop_significance.significance_for_best_config(sim, best_pct)

    print("[5/7] Monthly PnL comparison...")
    monthly = trailing_stop_monthly.monthly_comparison(sim, best_pct)
    monthly.to_csv(config.DATA_OUT / "trailing_stop_monthly.csv", index=False)

    print("[6/7] Train/test overfitting check & regime dependency...")
    overfit = trailing_stop_validation.run_overfitting_check(sim)
    sim_regime = trailing_stop_regime.attach_regime(sim, candles, trades)
    regime = trailing_stop_regime.performance_by_regime(sim_regime, best_pct)
    regime.to_csv(config.DATA_OUT / "trailing_stop_regime.csv", index=False)

    print("[7/7] Visualizations & Arabic report...")
    trailing_stop_visualizations.plot_grid_sweep(summary_cons, "conservative")
    trailing_stop_visualizations.plot_grid_sweep(summary_agg, "aggressive")
    trailing_stop_visualizations.plot_bootstrap_ci(mc)
    trailing_stop_visualizations.plot_monthly_comparison(monthly, best_pct)

    results = {
        "summary_conservative": summary_cons,
        "summary_aggressive": summary_agg,
        "monte_carlo": mc,
        "significance": sig,
        "monthly": monthly,
        "overfitting": overfit,
        "regime": regime,
        "best_pct": best_pct,
    }

    report_text = trailing_stop_reporting.build_report(results)
    config.TRAILING_STOP_REPORT.write_text(report_text, encoding="utf-8")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Report: {config.TRAILING_STOP_REPORT}")
    print(f"Data: {config.DATA_OUT}")
    print(f"Figures: {config.FIGURES_OUT}")


if __name__ == "__main__":
    main()
