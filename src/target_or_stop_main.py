"""Target-or-stop scenario test (test-only; direct answer to a user
question): what if every trade were left alone to hit its original stop
or a fixed target, with no early manual exit and no trailing? Reuses
exit_strategy_simulation.py's existing simulation -- no new simulator.

Run with: python -m src.target_or_stop_main
"""
import time

from src import (
    config, data_loading, exit_strategy_simulation, exit_strategy_metrics,
    target_or_stop_significance, target_or_stop_monte_carlo,
    target_or_stop_visualizations, target_or_stop_reporting,
)


def main():
    t0 = time.time()
    config.DATA_OUT.mkdir(parents=True, exist_ok=True)
    config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    config.REPORTS_OUT.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading data & simulating exit strategies (reused, no look-ahead)...")
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)

    tables, sig, mc, pct_unresolved = {}, {}, {}, {}
    for scenario in ("conservative", "aggressive"):
        print(f"[2/4] Metrics, significance & Monte Carlo ({scenario})...")
        tables[scenario] = exit_strategy_metrics.build_global_strategy_table(sim, scenario=scenario)
        sig[scenario] = target_or_stop_significance.significance_for_scenario(sim, scenario=scenario)

        base = sim[(sim["scenario"] == scenario) & (sim["strategy"] == "baseline")].sort_values("id")
        targ = sim[(sim["scenario"] == scenario) & (sim["strategy"] == "fixed_tp_idealTP")].sort_values("id")
        mc[scenario] = target_or_stop_monte_carlo.bootstrap_pf_ci(targ["exit_r"].values, base["exit_r"].values)
        pct_unresolved[scenario] = float(
            sim[(sim["scenario"] == scenario) & (sim["strategy"] == "fixed_tp_idealTP")]["unresolved"].mean())

        tables[scenario].to_csv(config.DATA_OUT / f"target_or_stop_table_{scenario}.csv", index=False)

    print("[3/4] Visualizations...")
    for scenario in ("conservative", "aggressive"):
        target_or_stop_visualizations.plot_equity_comparison(sim, scenario)
        target_or_stop_visualizations.plot_strategy_comparison(tables[scenario], scenario)
    base_cons = sim[(sim["scenario"] == "conservative") & (sim["strategy"] == "baseline")].sort_values("id")
    targ_cons = sim[(sim["scenario"] == "conservative") & (sim["strategy"] == "fixed_tp_idealTP")].sort_values("id")
    target_or_stop_visualizations.plot_bootstrap_pf(targ_cons["exit_r"].values, base_cons["exit_r"].values)

    print("[4/4] Arabic report...")
    results = {"tables": tables, "significance": sig, "monte_carlo": mc, "pct_unresolved": pct_unresolved}
    report_text = target_or_stop_reporting.build_report(results)
    config.TARGET_OR_STOP_REPORT.write_text(report_text, encoding="utf-8")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Report: {config.TARGET_OR_STOP_REPORT}")
    print(f"Data: {config.DATA_OUT}")
    print(f"Figures: {config.FIGURES_OUT}")


if __name__ == "__main__":
    main()
