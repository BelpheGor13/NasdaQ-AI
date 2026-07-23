"""Visualizations for the no-stop-loss scenario. Separate figure namespace
(no_stop_*) so this test-only analysis never overwrites the main pipeline's
or the other test-only analyses' figures.
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


def plot_equity_comparison(sim: pd.DataFrame) -> str:
    s = sim.dropna(subset=["orig_r", "no_sl_final_r"]).sort_values("dateStart_utc")
    equity_orig = np.cumsum(s["orig_r"].values)
    equity_no_sl = np.cumsum(s["no_sl_final_r"].values)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(equity_orig, color="#55A868", label="with stop-loss (actual)", linewidth=1.5)
    ax.plot(equity_no_sl, color="#C44E52", label="no stop-loss (hypothetical)", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Trade sequence")
    ax.set_ylabel("Cumulative R")
    ax.set_title("Equity curve: with vs without stop-loss")
    ax.legend()
    return _save(fig, "no_stop_equity_comparison")


def plot_worst_r_tail(sim: pd.DataFrame) -> str:
    worst = sim["no_sl_worst_r_reached"].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(worst, bins=50, color="#4C72B0", alpha=0.85)
    ax.axvline(config.NO_STOP_CATASTROPHIC_R_THRESHOLD, color="#C44E52", linestyle="--",
               label=f"catastrophic threshold ({config.NO_STOP_CATASTROPHIC_R_THRESHOLD}R)")
    ax.set_xlabel("Worst R reached during the ride")
    ax.set_ylabel("Number of trades")
    ax.set_title("Distribution of worst intermediate drawdown per trade (no stop-loss)")
    ax.legend()
    return _save(fig, "no_stop_worst_r_tail")


def plot_final_vs_worst(sim: pd.DataFrame) -> str:
    s = sim.dropna(subset=["no_sl_final_r", "no_sl_worst_r_reached"])
    colors = np.where(s["no_sl_final_r"] > 0, "#55A868", "#C44E52")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(s["no_sl_worst_r_reached"], s["no_sl_final_r"], c=colors, alpha=0.6, s=20)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(config.NO_STOP_CATASTROPHIC_R_THRESHOLD, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Worst R reached during the ride")
    ax.set_ylabel("Final R at the 30-day deadline")
    ax.set_title("Final outcome vs worst intermediate drawdown\n(green=ended positive, red=ended negative)")
    return _save(fig, "no_stop_final_vs_worst")


def plot_bootstrap_equity_diff(orig_r: np.ndarray, no_sl_r: np.ndarray, n_boot: int = 3000,
                                seed: int = config.RANDOM_SEED) -> str:
    rng = np.random.default_rng(seed)
    n = len(orig_r)
    idx = rng.integers(0, n, (n_boot, n))
    diff = no_sl_r[idx].sum(axis=1) - orig_r[idx].sum(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(diff, bins=50, color="#4C72B0", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", label="no difference")
    ax.axvline(diff.mean(), color="#C44E52", label="mean bootstrap difference")
    ax.set_xlabel("Bootstrapped total-R difference (no-SL minus with-SL)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo: total account impact of removing the stop-loss")
    ax.legend()
    return _save(fig, "no_stop_bootstrap_equity_diff")


if __name__ == "__main__":
    from src import data_loading, no_stop_simulation

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = no_stop_simulation.simulate_no_stop(trades, candles)

    p1 = plot_equity_comparison(sim)
    p2 = plot_worst_r_tail(sim)
    p3 = plot_final_vs_worst(sim)
    s = sim.dropna(subset=["orig_r", "no_sl_final_r"])
    p4 = plot_bootstrap_equity_diff(s["orig_r"].values, s["no_sl_final_r"].values)
    print(p1, p2, p3, p4, sep="\n")
