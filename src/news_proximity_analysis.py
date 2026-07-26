"""News-proximity pattern (test-only): does trading near a major US
economic release (CPI, NFP, FOMC, PCE, GDP -- see
outputs/data/red_news_calendar_2020_2024.csv, compiled from official BLS/
Federal Reserve/BEA sources only) predict worse outcomes? Tests a grid of
proximity windows and BEFORE vs AFTER separately, per the user's specific
question ("if a red event falls ~5 minutes before or after the trade").

Event times are published in Eastern Time by the source agencies; they are
localized to America/New_York (handles EST/EDT automatically) and
converted to UTC to match dateStart_utc, exactly like data_loading.py does
for trade timestamps.
"""
import numpy as np
import pandas as pd

from src import config, data_loading, stats_utils

NEWS_CALENDAR_CSV = config.DATA_OUT / "red_news_calendar_2020_2024.csv"
PROXIMITY_WINDOWS_MIN = [5, 15, 30, 60, 120, 180, 240, 360]


def load_news_calendar(path=NEWS_CALENDAR_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    naive = pd.to_datetime(df["date"] + " " + df["time_et"], format="%Y-%m-%d %H:%M")
    localized = naive.dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
    df["event_utc"] = localized.dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def nearest_event_offset(trades: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """For each trade, the signed minutes to the NEAREST news event
    (negative = event happened before entry, positive = after)."""
    out = trades.copy()
    event_times = np.sort(news["event_utc"].values)

    starts = out["dateStart_utc"].values
    idx_right = np.searchsorted(event_times, starts, side="left")

    nearest_offset_min = np.full(len(out), np.nan)
    for i in range(len(out)):
        candidates = []
        if idx_right[i] < len(event_times):
            candidates.append(event_times[idx_right[i]])
        if idx_right[i] > 0:
            candidates.append(event_times[idx_right[i] - 1])
        if not candidates:
            continue
        diffs = [(c - starts[i]) / np.timedelta64(1, "m") for c in candidates]
        nearest_offset_min[i] = min(diffs, key=abs)

    out["nearest_news_offset_min"] = nearest_offset_min
    return out


def proximity_performance(trades_with_offset: pd.DataFrame, window_min: int, direction: str = "any") -> dict:
    """direction: 'any' (|offset| <= window), 'before' (news occurred BEFORE
    entry, i.e. offset in [-window, 0)), 'after' (offset in (0, window])."""
    offset = trades_with_offset["nearest_news_offset_min"]
    if direction == "any":
        near_mask = offset.abs() <= window_min
    elif direction == "before":
        near_mask = (offset >= -window_min) & (offset < 0)
    else:
        near_mask = (offset > 0) & (offset <= window_min)

    near = trades_with_offset[near_mask]
    far = trades_with_offset[~near_mask]

    r_near, r_far = near["avgRiskReward"].dropna().values, far["avgRiskReward"].dropna().values
    p_value = stats_utils.welch_ttest_pvalue(r_near, r_far) if len(r_near) >= 2 else np.nan

    return {
        "window_min": window_min, "direction": direction,
        "n_near": len(near), "n_far": len(far),
        "win_rate_near": float((near["rPnL"] > 0).mean()) if len(near) else np.nan,
        "win_rate_far": float((far["rPnL"] > 0).mean()) if len(far) else np.nan,
        "expectancy_r_near": stats_utils.expectancy(r_near),
        "expectancy_r_far": stats_utils.expectancy(r_far),
        "profit_factor_near": stats_utils.profit_factor(r_near),
        "profit_factor_far": stats_utils.profit_factor(r_far),
        "welch_p_value": p_value,
    }


def run_grid(trades_with_offset: pd.DataFrame, windows=PROXIMITY_WINDOWS_MIN) -> pd.DataFrame:
    rows = []
    for w in windows:
        for direction in ("any", "before", "after"):
            rows.append(proximity_performance(trades_with_offset, w, direction))
    return pd.DataFrame(rows)


def news_during_trade(trades: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Complementary check: rather than proximity to ENTRY only, did any
    news event fall anywhere between dateStart_utc and dateEnd_utc (i.e.
    the release happened while the trade was actually open)? More relevant
    than entry-proximity alone since trades in this dataset run for hours,
    well past the tight windows tested above.
    """
    out = trades.copy()
    event_times = np.sort(news["event_utc"].values)

    starts = out["dateStart_utc"].values
    ends = out["dateEnd_utc"].values
    lo_idx = np.searchsorted(event_times, starts, side="left")
    hi_idx = np.searchsorted(event_times, ends, side="right")
    out["news_during_trade"] = (hi_idx > lo_idx)
    out["n_news_during_trade"] = hi_idx - lo_idx
    return out


def during_trade_performance(trades_with_flag: pd.DataFrame) -> dict:
    with_news = trades_with_flag[trades_with_flag["news_during_trade"]]
    without_news = trades_with_flag[~trades_with_flag["news_during_trade"]]
    r_with = with_news["avgRiskReward"].dropna().values
    r_without = without_news["avgRiskReward"].dropna().values
    p_value = stats_utils.welch_ttest_pvalue(r_with, r_without) if len(r_with) >= 2 else np.nan
    return {
        "n_with_news": len(with_news), "n_without_news": len(without_news),
        "win_rate_with_news": float((with_news["rPnL"] > 0).mean()) if len(with_news) else np.nan,
        "win_rate_without_news": float((without_news["rPnL"] > 0).mean()) if len(without_news) else np.nan,
        "expectancy_r_with_news": stats_utils.expectancy(r_with),
        "expectancy_r_without_news": stats_utils.expectancy(r_without),
        "profit_factor_with_news": stats_utils.profit_factor(r_with),
        "profit_factor_without_news": stats_utils.profit_factor(r_without),
        "welch_p_value": p_value,
    }


def during_trade_by_event_type(trades: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_type, group in news.groupby("event_type"):
        flagged = news_during_trade(trades, group)
        perf = during_trade_performance(flagged)
        perf["event_type"] = event_type
        rows.append(perf)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    trades = data_loading.load_trades()
    news = load_news_calendar()
    print(f"news events loaded: {len(news)}")
    print(news["event_type"].value_counts())

    with_offset = nearest_event_offset(trades, news)
    print("\nnearest_news_offset_min describe:")
    print(with_offset["nearest_news_offset_min"].describe())

    print("\n=== proximity grid ===")
    grid = run_grid(with_offset)
    print(grid.to_string(index=False))

    print("\n=== news DURING trade (any type) ===")
    flagged = news_during_trade(trades, news)
    print(f"trades with a news event during their open window: {flagged['news_during_trade'].sum()} / {len(flagged)}")
    print(during_trade_performance(flagged))

    print("\n=== news DURING trade, by event type ===")
    by_type = during_trade_by_event_type(trades, news)
    print(by_type.to_string(index=False))
