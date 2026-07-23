"""No-stop-loss scenario test (test-only; see no-stop-loss-prompt.md /
direct user question): what happens if the original initalSL is removed
entirely and each trade rides real candles to a documented deadline?

Run with: python -m src.no_stop_main
"""
import time

from src import (
    config, data_loading, no_stop_simulation, no_stop_metrics,
    no_stop_monte_carlo, no_stop_visualizations, no_stop_reporting,
)


def main():
    t0 = time.time()
    config.DATA_OUT.mkdir(parents=True, exist_ok=True)
    config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    config.REPORTS_OUT.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading data & simulating the no-stop-loss ride (no look-ahead)...")
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = no_stop_simulation.simulate_no_stop(trades, candles)
    sim.drop(columns=["dateStart_utc", "dateEnd_utc"]).to_csv(
        config.DATA_OUT / "no_stop_simulation.csv", index=False)

    s = sim.dropna(subset=["orig_r", "no_sl_final_r"]).sort_values("dateStart_utc")

    print("[2/5] Building comparison & tail-risk metrics...")
    summary = no_stop_metrics.summary_comparison(sim)
    tail = no_stop_metrics.tail_risk_summary(sim)
    recovery = no_stop_metrics.stop_outs_that_would_have_recovered(sim)
    summary.to_csv(config.DATA_OUT / "no_stop_summary.csv", index=False)

    print("[3/5] Paired significance test & Monte Carlo...")
    sig = no_stop_monte_carlo.paired_tests(s["orig_r"].values, s["no_sl_final_r"].values)
    boot = no_stop_monte_carlo.bootstrap_equity_difference(s["orig_r"].values, s["no_sl_final_r"].values)
    shuffle = no_stop_monte_carlo.reshuffled_drawdown_sensitivity(s["no_sl_final_r"].values)

    print("[4/5] Visualizations...")
    no_stop_visualizations.plot_equity_comparison(sim)
    no_stop_visualizations.plot_worst_r_tail(sim)
    no_stop_visualizations.plot_final_vs_worst(sim)
    no_stop_visualizations.plot_bootstrap_equity_diff(s["orig_r"].values, s["no_sl_final_r"].values)

    print("[5/5] Arabic report...")
    results = {
        "summary": summary,
        "tail_risk": tail,
        "recovery": recovery,
        "significance": sig,
        "bootstrap": boot,
        "shuffle": shuffle,
    }
    report_text = no_stop_reporting.build_report(results)
    config.NO_STOP_REPORT.write_text(report_text, encoding="utf-8")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Report: {config.NO_STOP_REPORT}")
    print(f"Data: {config.DATA_OUT}")
    print(f"Figures: {config.FIGURES_OUT}")


if __name__ == "__main__":
    main()
