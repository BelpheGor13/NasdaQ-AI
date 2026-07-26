"""Figures for the follow-up deep-dive (test-only): monthly target-stop
grid, SL/TP size buckets, seasonality-vs-target buckets. Saved with a
pattern_deepdive_ prefix into outputs/figures alongside existing figures.
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


def plot_monthly_target_stop_grid(grid_summary: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    baseline = grid_summary[grid_summary["variant"] == "baseline (no cap)"]["total_pnl"].values[0]

    for variant, color in [("profit_only", "#4C72B0"), ("target_or_stop", "#C44E52")]:
        sub = grid_summary[grid_summary["variant"] == variant]
        ax.plot(sub["target_pct"] * 100, sub["total_pnl"], marker="o", color=color, label=variant)
    ax.axhline(baseline, color="black", linestyle="--", label="baseline (no cap)")
    ax.set_xlabel("Monthly target %")
    ax.set_ylabel("Total PnL (USD)")
    ax.set_title("Monthly target-then-stop: total PnL vs baseline")
    ax.legend(fontsize=8)
    return _save(fig, "pattern_deepdive_monthly_target_stop")


def plot_size_buckets(table: pd.DataFrame, title: str, name: str) -> str:
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(table))
    colors = np.where(table["win_rate"] > table["win_rate"].median(), "#55A868", "#C44E52")
    ax1.bar(x, table["win_rate"], color=colors, alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(table["bucket"], rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("Win rate")

    ax2 = ax1.twinx()
    ax2.plot(x, table["profit_factor"].clip(upper=table["profit_factor"].replace(np.inf, np.nan).max() * 1.1),
              color="#333333", marker="o", label="Profit Factor (clipped)")
    ax2.set_ylabel("Profit Factor (clipped for display)")
    ax1.set_title(title)
    return _save(fig, name)


def plot_news_proximity(grid: pd.DataFrame) -> str:
    sub = grid[(grid["direction"] == "before") & (grid["n_near"] > 0)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sub["window_min"], sub["expectancy_r_near"], marker="o", color="#C44E52",
            label="Expectancy: trades AFTER a news release (within window)")
    ax.plot(sub["window_min"], sub["expectancy_r_far"], marker="o", color="#4C72B0",
            label="Expectancy: all other trades")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Minutes since nearest prior news release")
    ax.set_ylabel("Expectancy (R)")
    ax.set_title("Trading soon after a major release vs everything else")
    ax.legend(fontsize=8)
    return _save(fig, "pattern_deepdive_news_proximity")


def plot_threshold_scan(scan: pd.DataFrame, xlabel: str, name: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(scan["threshold"], scan["expectancy_below"], marker="o", color="#C44E52", label="Expectancy: below threshold")
    ax.plot(scan["threshold"], scan["expectancy_above"], marker="o", color="#55A868", label="Expectancy: above threshold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Expectancy (R)")
    ax.set_title("Continuous threshold scan")
    ax.legend(fontsize=8)
    return _save(fig, name)


def plot_joint_mismatch(joint: pd.DataFrame) -> str:
    joint = joint.copy()
    joint["label"] = joint["sl_category"] + " / " + joint["tp_category"]
    joint = joint.sort_values("expectancy_r")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = np.where(joint["expectancy_r"] > 0, "#55A868", "#C44E52")
    ax.barh(joint["label"], joint["expectancy_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Expectancy (R)")
    ax.set_title("Joint SL x TP mismatch (vs that month's own typical daily move)")
    return _save(fig, "pattern_deepdive_joint_mismatch")


if __name__ == "__main__":
    from src import data_loading, monthly_target_stop, sl_tp_size_analysis, seasonality_analysis, \
        news_proximity_analysis as npa
    import numpy as np

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()

    grid = monthly_target_stop.grid_summary(trades)
    print(plot_monthly_target_stop_grid(grid))

    sized_atr = sl_tp_size_analysis.add_size_atr(trades, candles)
    sl_table = sl_tp_size_analysis.bucket_performance(sized_atr, "sl_size_atr", config.SL_ATR_BUCKETS, as_pct=False)
    tp_table = sl_tp_size_analysis.bucket_performance(sized_atr, "tp_size_atr", config.TP_ATR_BUCKETS, as_pct=False)
    print(plot_size_buckets(sl_table, "Win rate / PF by SL size (multiples of ATR)", "pattern_deepdive_sl_size_buckets"))
    print(plot_size_buckets(tp_table, "Win rate / PF by TP size (multiples of ATR)", "pattern_deepdive_tp_size_buckets"))

    ratio_df, _ = seasonality_analysis.add_seasonal_ratio(trades, candles)
    season_table = seasonality_analysis.bucket_by_seasonal_ratio(ratio_df)
    print(plot_size_buckets(season_table, "Win rate / PF by target-vs-day-of-week-move ratio",
                             "pattern_deepdive_seasonal_ratio_buckets"))

    news = npa.load_news_calendar()
    with_offset = npa.nearest_event_offset(trades, news)
    news_grid = npa.run_grid(with_offset)
    print(plot_news_proximity(news_grid))
