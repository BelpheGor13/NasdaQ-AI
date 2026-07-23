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


def plot_walk_forward_folds(condition_label: str, fold_df: pd.DataFrame, rank: int = 1) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = np.where(fold_df["expectancy"] > 0, "#55A868", "#C44E52")
    ax.bar(fold_df["fold"].astype(str), fold_df["expectancy"].fillna(0), color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Expectancy (R)")
    ax.set_title(f"Walk-forward fold performance (rank #{rank}):\n{condition_label[:60]}")
    return _save(fig, f"walk_forward_rank{rank}")


def plot_monte_carlo_distribution(r_values: np.ndarray, rank: int = 1, n_boot: int = 3000, seed: int = 42) -> str:
    rng = np.random.default_rng(seed)
    n = len(r_values)
    idx = rng.integers(0, n, (n_boot, n))
    boot_means = r_values[idx].mean(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(boot_means, bins=40, color="#4C72B0", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--")
    ax.axvline(r_values.mean(), color="red", label="observed expectancy")
    ax.set_xlabel("Bootstrapped expectancy (R)")
    ax.set_title(f"Monte Carlo bootstrap distribution (rank #{rank})")
    ax.legend()
    return _save(fig, f"monte_carlo_rank{rank}")


def plot_entry_conditions(feats: pd.DataFrame, condition_dict: dict, rank: int = 1,
                           condition_label: str = "") -> str:
    """Shows what each condition in a pattern actually restricts: a bar of
    category counts for categorical conditions, or a histogram of the raw
    continuous feature with the selected bin shaded, so a reader can see
    where the rule sits in the data rather than just reading a bin string.
    """
    n_cond = len(condition_dict)
    fig, axes = plt.subplots(1, n_cond, figsize=(5 * n_cond, 4))
    if n_cond == 1:
        axes = [axes]

    for ax, (col, val) in zip(axes, condition_dict.items()):
        if col.endswith("_bin"):
            base_col = col[:-4]
            raw = feats[base_col].dropna()
            ax.hist(raw, bins=40, color="#B0B0B0", alpha=0.8)
            lo_str, hi_str = val.strip("()[]").split(",")
            lo, hi = float(lo_str), float(hi_str)
            ax.axvspan(lo, hi, color="#C44E52", alpha=0.4, label="pattern's bin")
            ax.set_xlabel(base_col)
            ax.legend(fontsize=8)
        else:
            counts = feats[col].value_counts()
            colors = ["#C44E52" if str(k) == str(val) else "#B0B0B0" for k in counts.index]
            ax.bar([str(k) for k in counts.index], counts.values, color=colors)
            ax.set_xlabel(col)
            ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Count")

    fig.suptitle(f"Entry conditions (rank #{rank}): {condition_label[:70]}", fontsize=10)
    return _save(fig, f"entry_conditions_rank{rank}")


def plot_mfe_mae_for_pattern(exit_df: pd.DataFrame, mask: pd.Series, rank: int = 1,
                              condition_label: str = "") -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    rest = ~mask
    ax.scatter(exit_df.loc[rest, "mae_r"], exit_df.loc[rest, "mfe_r"],
               c="#D9D9D9", alpha=0.4, s=14, label="other trades")
    matched = exit_df.loc[mask]
    colors = np.where(matched["avgRiskReward"] > 0, "#55A868", "#C44E52")
    ax.scatter(matched["mae_r"], matched["mfe_r"], c=colors, alpha=0.85, s=28,
               edgecolors="black", linewidths=0.3, label="matches pattern")
    ax.set_xlabel("MAE (R)")
    ax.set_ylabel("MFE (R)")
    ax.set_title(f"MFE vs MAE, pattern rank #{rank}:\n{condition_label[:60]}")
    ax.legend(fontsize=8)
    return _save(fig, f"mfe_mae_pattern_rank{rank}")


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
