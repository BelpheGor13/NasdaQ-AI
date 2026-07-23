"""Stage 15: the visualizations the deliverable spec calls for. Each
function saves a PNG into outputs/figures and returns its path. Kept
independent of any single figure's data source so main.py can call them
with whatever the pipeline produced.
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


def plot_exit_quality_distribution(exit_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = exit_df["exit_quality"].value_counts()
    ax.bar(counts.index, counts.values, color="#4C72B0")
    ax.set_ylabel("Number of trades")
    ax.set_title("Exit Quality Distribution")
    plt.xticks(rotation=20, ha="right")
    return _save(fig, "exit_quality_distribution")


def plot_mfe_mae_scatter(exit_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(exit_df["avgRiskReward"] > 0, "#55A868", "#C44E52")
    ax.scatter(exit_df["mae_r"], exit_df["mfe_r"], c=colors, alpha=0.5, s=18)
    ax.set_xlabel("MAE (R)")
    ax.set_ylabel("MFE (R)")
    ax.set_title("MFE vs MAE per trade (green=win, red=loss)")
    return _save(fig, "mfe_mae_scatter")


def plot_win_rate_vs_base_rate(scored: pd.DataFrame, base_rate: float, top_n: int = 10) -> str:
    top = scored.head(top_n)
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(top))
    ax.barh(y, top["win_rate"], color="#4C72B0", label="pattern win rate")
    ax.axvline(base_rate, color="black", linestyle="--", label="base rate")
    ax.set_yticks(y)
    ax.set_yticklabels([c[:45] for c in top["condition"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Win rate")
    ax.set_title("Top candidates: win rate vs base rate")
    ax.legend()
    return _save(fig, "win_rate_vs_base_rate")


def plot_walk_forward_folds(condition_label: str, fold_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = np.where(fold_df["expectancy"] > 0, "#55A868", "#C44E52")
    ax.bar(fold_df["fold"].astype(str), fold_df["expectancy"].fillna(0), color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Expectancy (R)")
    ax.set_title(f"Walk-forward fold performance:\n{condition_label[:60]}")
    return _save(fig, "walk_forward_top_pattern")


def plot_monte_carlo_distribution(r_values: np.ndarray, n_boot: int = 3000, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    n = len(r_values)
    idx = rng.integers(0, n, (n_boot, n))
    boot_means = r_values[idx].mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(boot_means, bins=40, color="#4C72B0", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--")
    ax.axvline(r_values.mean(), color="red", label="observed expectancy")
    ax.set_xlabel("Bootstrapped expectancy (R)")
    ax.set_title("Monte Carlo bootstrap distribution (top pattern)")
    ax.legend()
    return _save(fig, "monte_carlo_distribution")


def plot_yearly_edge_decay(yearly_df: pd.DataFrame) -> str:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(yearly_df["year"].astype(str), yearly_df["expectancy"], color="#4C72B0", alpha=0.7)
    ax1.set_ylabel("Expectancy (R)", color="#4C72B0")
    ax1.axhline(0, color="black", linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.plot(yearly_df["year"].astype(str), yearly_df["profit_factor"], color="#C44E52", marker="o")
    ax2.set_ylabel("Profit Factor", color="#C44E52")
    ax1.set_title("Yearly Edge Decay: Expectancy & Profit Factor")
    return _save(fig, "yearly_edge_decay")
