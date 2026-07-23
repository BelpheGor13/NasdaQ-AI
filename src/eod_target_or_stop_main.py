"""End-of-day (no overnight holding) target-or-stop test -- direct answer
to a user-specified rule. Never touches the real trade log.

Run with: python -m src.eod_target_or_stop_main
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import (
    config, data_loading, eod_target_or_stop_simulation as eod_sim,
    target_or_stop_significance as sig, target_or_stop_monte_carlo as mc,
    stats_utils, eod_target_or_stop_reporting as reporting,
)

REPORT_PATH = config.REPORTS_OUT / "eod_target_or_stop_report_arabic.md"
DEADLINE_HOUR_NY = 16


def _max_drawdown(r_values_chronological: np.ndarray) -> float:
    equity = np.cumsum(r_values_chronological)
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max())


def plot_equity_comparison(sim):
    s = sim.sort_values("dateStart_utc")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.cumsum(s["orig_r"].values), color="#4C72B0", label="actual (manual exit)", linewidth=1.5)
    ax.plot(np.cumsum(s["eod_exit_r"].values), color="#55A868",
             label=f"target-or-stop, forced close {DEADLINE_HOUR_NY}:00 NY", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Trade sequence")
    ax.set_ylabel("Cumulative R")
    ax.set_title("Equity curve: actual vs target-or-stop with no overnight holding")
    ax.legend()
    config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_OUT / "eod_target_or_stop_equity.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def main():
    t0 = time.time()
    config.REPORTS_OUT.mkdir(parents=True, exist_ok=True)
    config.DATA_OUT.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading data & simulating end-of-day target-or-stop (no look-ahead)...")
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = eod_sim.simulate_eod_target_or_stop(trades, candles, deadline_hour_ny=DEADLINE_HOUR_NY)
    sim.drop(columns=["dateStart_utc", "dateEnd_utc"]).to_csv(
        config.DATA_OUT / "eod_target_or_stop_simulation.csv", index=False)

    s = sim.dropna(subset=["orig_r", "eod_exit_r"]).sort_values("dateStart_utc")
    orig, eodr = s["orig_r"].values, s["eod_exit_r"].values

    print("[2/4] Metrics, significance & Monte Carlo...")
    summary = {
        "n": len(s),
        "mean_orig": float(orig.mean()), "median_orig": float(np.median(orig)),
        "win_rate_orig": float((orig > 0).mean()), "pf_orig": stats_utils.profit_factor(orig),
        "max_dd_orig": _max_drawdown(orig),
        "mean_eod": float(eodr.mean()), "median_eod": float(np.median(eodr)),
        "win_rate_eod": float((eodr > 0).mean()), "pf_eod": stats_utils.profit_factor(eodr),
        "max_dd_eod": _max_drawdown(eodr),
    }
    significance = sig.paired_tests(orig, eodr)
    monte_carlo = mc.bootstrap_pf_ci(eodr, orig)
    exit_reasons = s["eod_exit_reason"].value_counts().to_dict()

    print("[3/4] Visualization...")
    plot_equity_comparison(s)

    print("[4/4] Arabic report...")
    results = {
        "summary": summary, "significance": significance, "monte_carlo": monte_carlo,
        "exit_reasons": exit_reasons, "deadline_hour_ny": DEADLINE_HOUR_NY,
    }
    report_text = reporting.build_report(results)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
