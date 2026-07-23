"""Figures for the hidden-pattern/regime-conditional-exit deliverable,
saved into outputs/figures with a hidden_pattern_ prefix.
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


def plot_cluster_strategy_pf(cluster_table: pd.DataFrame, top_n: int = 6) -> str:
    clusters = sorted(cluster_table["cluster"].unique())
    fig, axes = plt.subplots(1, len(clusters), figsize=(6 * len(clusters), 4.5), squeeze=False)
    for ax, cluster in zip(axes[0], clusters):
        sub = cluster_table[cluster_table["cluster"] == cluster].sort_values("profit_factor", ascending=False).head(top_n)
        colors = ["#C44E52" if b else "#4C72B0" for b in sub["best_in_cluster"]]
        ax.barh(sub["strategy"], sub["profit_factor"], color=colors)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("Profit Factor")
        ax.set_title(f"Cluster {int(cluster)}: top {top_n} exit strategies")
    return _save(fig, "hidden_pattern_cluster_strategy_pf")


def plot_monthly_comparison(monthly: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(monthly))
    width = 0.4
    ax.bar(x - width / 2, monthly["original_pnl_usd"], width, color="#4C72B0", label="Original PnL")
    ax.bar(x + width / 2, monthly["hybrid_pnl_usd"], width, color="#55A868", label="Hybrid PnL")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(monthly["month"], rotation=90, fontsize=6)
    ax.set_ylabel("PnL (USD)")
    ax.set_title("Monthly PnL: original vs regime-conditional hybrid strategy")
    ax.legend()
    return _save(fig, "hidden_pattern_monthly_comparison")


def plot_bootstrap_ci(mc_table: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(mc_table))
    yerr = np.array([mc_table["mean_pf"] - mc_table["ci_lo_5th"], mc_table["ci_hi_95th"] - mc_table["mean_pf"]])
    ax.errorbar(x, mc_table["mean_pf"], yerr=yerr, fmt="o", color="#4C72B0", capsize=5,
                label="Bootstrap PF (5th-95th pct)")
    ax.scatter(x, mc_table["baseline_pf"], color="black", marker="_", s=200, label="Cluster baseline PF")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {int(c)}\n{s}" for c, s in zip(mc_table["cluster"], mc_table["best_strategy"])],
                        fontsize=8)
    ax.set_ylabel("Profit Factor")
    ax.set_title("Monte Carlo bootstrap CI per cluster's best strategy")
    ax.legend(fontsize=8)
    return _save(fig, "hidden_pattern_monte_carlo_ci")


def plot_yearly_stability(yearly: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for cluster, group in yearly.groupby("cluster"):
        ax.plot(group["year"], group["pf_of_overall_best_this_year"], marker="o", label=f"Cluster {int(cluster)}")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Profit Factor (best global strategy, scored per year)")
    ax.set_title("Yearly stability of the winning exit strategy")
    ax.legend()
    return _save(fig, "hidden_pattern_yearly_stability")


def plot_drawdown_comparison(dd: dict) -> str:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels = ["Baseline", "Hybrid"]
    values = [dd["baseline_max_dd_r"], dd["hybrid_max_dd_r"]]
    ax.bar(labels, values, color=["#C44E52", "#55A868"])
    ax.set_ylabel("Max Drawdown (R)")
    ax.set_title("Max drawdown: baseline vs hybrid")
    return _save(fig, "hidden_pattern_drawdown_comparison")


if __name__ == "__main__":
    from src import (data_loading, exit_strategy_simulation, exit_strategy_metrics, exit_strategy_monte_carlo,
                      hybrid_strategy, hidden_pattern_monthly, hidden_pattern_features as hpf,
                      hidden_pattern_clustering as hpc)

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)
    feats = hpf.build_hidden_pattern_features(trades, candles)
    X = hpc.build_cluster_matrix(feats)
    fit = hpc.fit_clusters(X)
    cluster_map = pd.Series(fit["labels"], index=feats.loc[X.index, "id"].values)

    table = exit_strategy_metrics.build_cluster_strategy_table(sim, cluster_map)
    best = exit_strategy_metrics.best_strategy_per_cluster(table)
    hybrid_df = hybrid_strategy.build_hybrid_exit_r(sim, cluster_map, best)
    monthly = hidden_pattern_monthly.monthly_comparison(hybrid_df)
    yearly = hidden_pattern_monthly.yearly_stability(sim, cluster_map, best)
    dd = hidden_pattern_monthly.drawdown_profile(hybrid_df)
    mc = exit_strategy_monte_carlo.run_mc_for_best_per_cluster(sim, cluster_map, best)

    print(plot_cluster_strategy_pf(table))
    print(plot_monthly_comparison(monthly))
    print(plot_bootstrap_ci(mc))
    print(plot_yearly_stability(yearly))
    print(plot_drawdown_comparison(dd))
