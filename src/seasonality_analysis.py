"""Seasonality-vs-target-size (test-only), DAY-level (per explicit user
request -- monthly was rejected as too coarse). No free external source
publishes finer-than-monthly seasonality data (confirmed via research: the
Barchart/thetradingtools sites checked only expose monthly Abs-Average
tables), so this is computed from OUR OWN 1-minute candles
(nasdaq_m1_2020_2024.parquet) as day-of-week averages -- the finest "day"
granularity with enough repetition (2020-2024) to be meaningful (~250
occurrences of Monday, ~250 of Tuesday, etc.) rather than a single exact
calendar day (~1 occurrence per year, far too small a sample).

Day's move = (daily_high - daily_low) / daily_open, i.e. the day's total
range as % of that day's open -- comparable in kind to tp_size_pct.
"""
import numpy as np
import pandas as pd

from src import config, data_loading, regime_detection, sl_tp_size_analysis, stats_utils

TARGET_VS_SEASONAL_BUCKETS = [0, 0.25, 0.5, 1.0, 2.0, np.inf]
TARGET_VS_SEASONAL_LABELS = ["<0.25x", "0.25x-0.5x", "0.5x-1.0x", "1.0x-2.0x", "2.0x+"]


def compute_day_of_week_seasonality(candles: pd.DataFrame) -> pd.Series:
    daily = regime_detection.resample_daily(candles)
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["open"]
    daily["dow"] = daily["datetime_utc"].dt.dayofweek  # 0=Mon .. 4=Fri
    return daily.groupby("dow")["range_pct"].mean()


def add_seasonal_ratio(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    out = sl_tp_size_analysis.add_size_pct(trades)
    dow_seasonality = compute_day_of_week_seasonality(candles)
    out["entry_dow"] = out["dateStart_utc"].dt.dayofweek
    out["seasonal_abs_avg_move_pct"] = out["entry_dow"].map(dow_seasonality)
    out["target_vs_seasonal_ratio"] = out["tp_size_pct"] / out["seasonal_abs_avg_move_pct"]
    return out, dow_seasonality


def bucket_by_seasonal_ratio(trades_with_ratio: pd.DataFrame) -> pd.DataFrame:
    bucket = pd.cut(trades_with_ratio["target_vs_seasonal_ratio"], bins=TARGET_VS_SEASONAL_BUCKETS,
                     labels=TARGET_VS_SEASONAL_LABELS, include_lowest=True)
    rows = []
    for label, group in trades_with_ratio.groupby(bucket, observed=True):
        r = group["rPnL"].values
        r_mult = group["avgRiskReward"].dropna().values
        rows.append({
            "bucket": label, "n": len(group),
            "low_confidence": len(group) < config.MIN_SAMPLE_SIZE,
            "win_rate": float((r > 0).mean()) if len(r) else np.nan,
            "expectancy_r": stats_utils.expectancy(r_mult),
            "profit_factor": stats_utils.profit_factor(r_mult),
            "total_pnl_usd": float(r.sum()),
        })
    return pd.DataFrame(rows)


def by_day_of_week_breakdown(trades_with_ratio: pd.DataFrame) -> pd.DataFrame:
    dow_names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    rows = []
    for dow, group in trades_with_ratio.groupby("entry_dow"):
        r_mult = group["avgRiskReward"].dropna().values
        rows.append({
            "day": dow_names.get(dow, dow), "n": len(group),
            "seasonal_abs_avg_move_pct": float(group["seasonal_abs_avg_move_pct"].iloc[0] * 100),
            "mean_target_pct": float(group["tp_size_pct"].mean() * 100),
            "mean_ratio": float(group["target_vs_seasonal_ratio"].mean()),
            "win_rate": float((group["rPnL"] > 0).mean()),
            "expectancy_r": stats_utils.expectancy(r_mult),
        })
    return pd.DataFrame(rows).sort_values("day", key=lambda s: s.map({"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4}))


def compute_monthly_seasonality(candles: pd.DataFrame) -> pd.Series:
    """Monthly seasonality computed from OUR OWN candle data (not an
    external site) -- per explicit user request to use 'all seasonal
    trends available to us,' i.e. self-computed, at calendar-month
    granularity this time (the earlier version used an external monthly
    source; this one keeps the month granularity but derives it in-house,
    matching OANDA:NAS100USD exactly rather than a related index)."""
    daily = regime_detection.resample_daily(candles)
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["open"]
    daily["month"] = daily["datetime_utc"].dt.month
    return daily.groupby("month")["range_pct"].mean()


def add_monthly_ratios(trades: pd.DataFrame, candles: pd.DataFrame) -> tuple:
    """Both SL and TP measured against the SAME month-of-entry seasonal
    reference (own data) -- answers the joint question: is my stop and/or
    target mismatched (too tight or too wide) relative to how much NAS100
    typically moves in that specific calendar month, and does that
    mismatch predict a worse outcome?"""
    out = sl_tp_size_analysis.add_size_pct(trades)
    monthly_seasonality = compute_monthly_seasonality(candles)
    out["entry_month"] = out["dateStart_utc"].dt.month
    out["month_seasonal_move_pct"] = out["entry_month"].map(monthly_seasonality)
    out["sl_vs_month_ratio"] = out["sl_size_pct"] / out["month_seasonal_move_pct"]
    out["tp_vs_month_ratio"] = out["tp_size_pct"] / out["month_seasonal_move_pct"]
    return out, monthly_seasonality


def joint_sl_tp_mismatch(trades_with_ratios: pd.DataFrame,
                          sl_tight_cutoff: float = 0.05, sl_wide_cutoff: float = 0.30,
                          tp_tight_cutoff: float = 0.25) -> pd.DataFrame:
    """Categorizes each trade by whether its STOP and/or TARGET are
    mismatched (too tight/wide) relative to that month's typical daily
    move, and reports performance per combination."""
    out = trades_with_ratios.copy()
    out["sl_category"] = np.select(
        [out["sl_vs_month_ratio"] < sl_tight_cutoff, out["sl_vs_month_ratio"] > sl_wide_cutoff],
        ["sl_too_tight", "sl_too_wide"], default="sl_normal")
    out["tp_category"] = np.where(out["tp_vs_month_ratio"] < tp_tight_cutoff, "tp_too_tight", "tp_normal")

    rows = []
    for (sl_cat, tp_cat), group in out.groupby(["sl_category", "tp_category"]):
        r_mult = group["avgRiskReward"].dropna().values
        rows.append({
            "sl_category": sl_cat, "tp_category": tp_cat, "n": len(group),
            "win_rate": float((group["rPnL"] > 0).mean()),
            "expectancy_r": stats_utils.expectancy(r_mult),
            "profit_factor": stats_utils.profit_factor(r_mult),
        })
    return pd.DataFrame(rows).sort_values(["tp_category", "sl_category"])


def by_month_breakdown_own_data(trades_with_ratios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, group in trades_with_ratios.groupby("entry_month"):
        r_mult = group["avgRiskReward"].dropna().values
        rows.append({
            "month": month, "n": len(group),
            "month_seasonal_move_pct": float(group["month_seasonal_move_pct"].iloc[0] * 100),
            "mean_sl_vs_month_ratio": float(group["sl_vs_month_ratio"].mean()),
            "mean_tp_vs_month_ratio": float(group["tp_vs_month_ratio"].mean()),
            "win_rate": float((group["rPnL"] > 0).mean()),
            "expectancy_r": stats_utils.expectancy(r_mult),
        })
    return pd.DataFrame(rows).sort_values("month")


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()

    dow_seasonality = compute_day_of_week_seasonality(candles)
    print("=== day-of-week average daily range % (own candle data, 2020-2024) ===")
    print((dow_seasonality * 100).round(3))

    ratio_df, _ = add_seasonal_ratio(trades, candles)

    print("\n=== target_vs_seasonal_ratio describe ===")
    print(ratio_df["target_vs_seasonal_ratio"].describe())

    print("\n=== performance by target-vs-seasonal-move bucket ===")
    print(bucket_by_seasonal_ratio(ratio_df).to_string(index=False))

    print("\n=== by day of week ===")
    print(by_day_of_week_breakdown(ratio_df).to_string(index=False))

    monthly_seasonality = compute_monthly_seasonality(candles)
    print("\n=== monthly average daily range % (own candle data) ===")
    print((monthly_seasonality * 100).round(3))

    monthly_ratio_df, _ = add_monthly_ratios(trades, candles)
    print("\n=== by calendar month (own-data seasonality, SL+TP) ===")
    print(by_month_breakdown_own_data(monthly_ratio_df).to_string(index=False))

    print("\n=== joint SL x TP mismatch categories ===")
    print(joint_sl_tp_mismatch(monthly_ratio_df).to_string(index=False))
