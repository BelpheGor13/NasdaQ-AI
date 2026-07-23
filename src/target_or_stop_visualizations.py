"""Visualizations for the target-or-stop scenario. Separate figure
namespace (target_or_stop_*) alongside the other test-only analyses.
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


def plot_equity_comparison(sim: pd.DataFrame, scenario: str = "conservative") -> str:
    base = sim[(sim["scenario"] == scenario) & (sim["strategy"] == "baseline")].sort_values("dateStart_utc")
    targ = sim[(sim["scenario"] == scenario) & (sim["strategy"] == "fixed_tp_idealTP")].sort_values("dateStart_utc")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.cumsum(base["exit_r"].values), color="#4C72B0", label="actual (discretionary exit)", linewidth=1.5)
    ax.plot(np.cumsum(targ["exit_r"].values), color="#55A868", label="target-or-stop (no intervention)", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Trade sequence")
    ax.set_ylabel("Cumulative R")
    ax.set_title(f"Equity curve: actual exit vs target-or-stop ({scenario})")
    ax.legend()
    return _save(fig, f"target_or_stop_equity_{scenario}")


def plot_strategy_comparison(global_table: pd.DataFrame, scenario: str) -> str:
    order = ["baseline", "fixed_tp_idealTP", "fixed_tp_2R", "fixed_tp_3R", "fixed_tp_4R"]
    t = global_table[global_table["strategy"].isin(order)].set_index("strategy").loc[order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(t["strategy"], t["win_rate"], color="#4C72B0")
    axes[0].set_ylabel("Win rate")
    axes[0].set_title("Win rate by exit rule")
    axes[0].tick_params(axis="x", rotation=25)

    pf_vals = t["profit_factor"].replace(np.inf, t["profit_factor"].replace(np.inf, np.nan).max() * 1.2)
    axes[1].bar(t["strategy"], pf_vals, color="#55A868")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Profit factor")
    axes[1].set_title("Profit factor by exit rule")
    axes[1].tick_params(axis="x", rotation=25)

    fig.suptitle(f"Fixed-target-or-stop rules vs actual ({scenario})")
    return _save(fig, f"target_or_stop_strategy_comparison_{scenario}")


def plot_bootstrap_pf(target_r: np.ndarray, baseline_r: np.ndarray, n_boot: int = 3000,
                       seed: int = config.RANDOM_SEED) -> str:
    rng = np.random.default_rng(seed)
    n = len(target_r)
    idx = rng.integers(0, n, (n_boot, n))

    def pf_batch(samples):
        wins = np.where(samples > 0, samples, 0).sum(axis=1)
        losses = np.where(samples < 0, -samples, 0).sum(axis=1)
        return np.divide(wins, losses, out=np.full(n_boot, np.inf), where=losses > 0)

    target_pf = pf_batch(target_r[idx])
    target_pf = target_pf[np.isfinite(target_pf)]
    baseline_pf = float(np.where(baseline_r > 0, baseline_r, 0).sum() / -np.where(baseline_r < 0, baseline_r, 0).sum())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(target_pf, bins=50, color="#55A868", alpha=0.85)
    ax.axvline(baseline_pf, color="#C44E52", linestyle="--", label="actual baseline PF")
    ax.axvline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("Bootstrapped Profit Factor (target-or-stop)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo: target-or-stop PF distribution vs actual baseline")
    ax.legend()
    return _save(fig, "target_or_stop_bootstrap_pf")


if __name__ == "__main__":
    from src import data_loading, exit_strategy_simulation, exit_strategy_metrics

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)

    for scenario in ("conservative", "aggressive"):
        plot_equity_comparison(sim, scenario)
        table = exit_strategy_metrics.build_global_strategy_table(sim, scenario=scenario)
        plot_strategy_comparison(table, scenario)

    base = sim[(sim["scenario"] == "conservative") & (sim["strategy"] == "baseline")].sort_values("id")
    targ = sim[(sim["scenario"] == "conservative") & (sim["strategy"] == "fixed_tp_idealTP")].sort_values("id")
    print(plot_bootstrap_pf(targ["exit_r"].values, base["exit_r"].values))
