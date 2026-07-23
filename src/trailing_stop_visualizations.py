"""Figures for the trailing-stop deliverable: the grid-sweep curve (PF /
expectancy vs trailing %), the monthly PnL comparison, and the bootstrap CI
chart for the top configs. Saved into outputs/figures alongside the
existing pattern-discovery figures, distinguished by a trailing_stop_ prefix.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

config.FIGURES_OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> str:
    path = config.FIGURES_OUT / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_grid_sweep(summary_table: pd.DataFrame, scenario: str) -> str:
    grid = summary_table[summary_table["config"] != "baseline (no trailing)"].sort_values("pct")
    baseline_pf = summary_table.loc[summary_table["config"] == "baseline (no trailing)", "profit_factor"].values[0]
    baseline_exp = summary_table.loc[summary_table["config"] == "baseline (no trailing)", "expectancy"].values[0]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(grid["pct"] * 100, grid["profit_factor"], color="#4C72B0", marker="o", label="Profit Factor (trailing)")
    ax1.axhline(baseline_pf, color="#4C72B0", linestyle="--", alpha=0.6, label="Profit Factor (baseline)")
    ax1.set_xlabel("Trailing stop % (giveback of peak profit)")
    ax1.set_ylabel("Profit Factor", color="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(grid["pct"] * 100, grid["expectancy"], color="#C44E52", marker="s", label="Expectancy (trailing)")
    ax2.axhline(baseline_exp, color="#C44E52", linestyle="--", alpha=0.6, label="Expectancy (baseline)")
    ax2.set_ylabel("Expectancy (R)", color="#C44E52")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")
    ax1.set_title(f"Trailing-stop grid sweep ({scenario})")
    return _save(fig, f"trailing_stop_grid_sweep_{scenario}")


def plot_monthly_comparison(monthly: pd.DataFrame, best_pct: float) -> str:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(monthly))
    width = 0.4
    ax.bar(x - width / 2, monthly["original_pnl_usd"], width, color="#4C72B0", label="Original PnL")
    ax.bar(x + width / 2, monthly["trailing_pnl_usd"], width, color="#C44E52", label=f"Trailing {best_pct*100:.0f}% PnL")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly["month"], rotation=90, fontsize=6)
    ax.set_ylabel("PnL (USD)")
    ax.set_title(f"Monthly PnL: original vs best trailing config ({best_pct*100:.0f}%)")
    ax.legend()
    return _save(fig, "trailing_stop_monthly_comparison")


def plot_bootstrap_ci(mc_table: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(mc_table))
    yerr = np.array([
        mc_table["mean_pf"] - mc_table["ci_lo_5th"],
        mc_table["ci_hi_95th"] - mc_table["mean_pf"],
    ])
    ax.errorbar(x, mc_table["mean_pf"], yerr=yerr, fmt="o", color="#4C72B0", capsize=5, label="Bootstrap PF (5th-95th pct)")
    ax.axhline(mc_table["observed_baseline_pf"].iloc[0], color="black", linestyle="--", label="Baseline PF")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p*100:.0f}%" for p in mc_table["pct"]])
    ax.set_xlabel("Trailing stop %")
    ax.set_ylabel("Profit Factor")
    ax.set_title("Monte Carlo bootstrap CI, top 3 configs")
    ax.legend(fontsize=8)
    return _save(fig, "trailing_stop_monte_carlo_ci")


if __name__ == "__main__":
    from src import (data_loading, trailing_stop_simulation, trailing_stop_metrics,
                      trailing_stop_monte_carlo, trailing_stop_monthly)

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = trailing_stop_simulation.simulate_trailing_stops(trades, candles)
    summary = trailing_stop_metrics.build_summary_table(sim, scenario="conservative")
    best_pct = summary[summary["config"] != "baseline (no trailing)"].sort_values(
        "profit_factor", ascending=False).iloc[0]["pct"]

    print(plot_grid_sweep(summary, "conservative"))
    mc = trailing_stop_monte_carlo.run_mc_for_top_configs(sim, summary)
    print(plot_bootstrap_ci(mc))
    monthly = trailing_stop_monthly.monthly_comparison(sim, best_pct)
    print(plot_monthly_comparison(monthly, best_pct))
