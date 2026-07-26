"""SL/TP size analysis (test-only). PRIMARY measure (per user correction):
SL/TP size relative to same-day ATR(14) -- i.e. relative to how much
NAS100 actually moves, not a flat % of its own price level. Raw %-of-
entry-price is unreliable here because NAS100's price level roughly
tripled from 2020 (~9,000) to 2024 (~20,000+), so a fixed point-distance
stop reads as a shrinking % over the sample -- confounding "stop size"
with "which year the trade happened in." The ATR-relative measure is
scale-free across the whole period. Fine buckets (~0.05 ATR increments)
are used to locate the precise threshold rather than a few coarse bands.
"""
import numpy as np
import pandas as pd

from src import config, data_loading, feature_engineering, regime_detection, stats_utils


def add_size_pct(trades: pd.DataFrame) -> pd.DataFrame:
    """Secondary/reference view: raw % of entry price."""
    out = trades.copy()
    out["sl_size_pct"] = (out["entryPrice"] - out["initalSL"]).abs() / out["entryPrice"]
    out["tp_size_pct"] = (out["idealTP"] - out["entryPrice"]).abs() / out["entryPrice"]
    return out


def add_size_atr(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Primary measure: SL/TP size as a multiple of same-day ATR(14)."""
    feats = feature_engineering.build_features(trades, candles)
    out = trades.copy()
    out["sl_size_atr"] = feats["sl_pct_of_atr"].values
    out["tp_size_atr"] = (feats["idealTP"] - feats["entryPrice"]).abs().values / feats["atr14_asof_prior_day"].values
    return out


def _bucket_label(edges, i, as_pct=True):
    lo, hi = edges[i], edges[i + 1]
    if as_pct:
        hi_str = "+" if np.isinf(hi) else f"-{hi*100:.1f}%"
        return f"{lo*100:.1f}%{hi_str}"
    hi_str = "+" if np.isinf(hi) else f"-{hi:.2f}"
    return f"{lo:.2f}{hi_str}"


def bucket_performance(trades: pd.DataFrame, size_col: str, edges: list, as_pct=True) -> pd.DataFrame:
    labels = [_bucket_label(edges, i, as_pct) for i in range(len(edges) - 1)]
    bucket = pd.cut(trades[size_col], bins=edges, labels=labels, include_lowest=True)

    rows = []
    for label, group in trades.groupby(bucket, observed=True):
        r = group["rPnL"].values
        r_mult = group["avgRiskReward"].dropna().values
        rows.append({
            "bucket": label, "n": len(group),
            "low_confidence": len(group) < config.MIN_SAMPLE_SIZE,
            "win_rate": float((r > 0).mean()) if len(r) else np.nan,
            "expectancy_r": stats_utils.expectancy(r_mult),
            "expectancy_usd": float(np.mean(r)) if len(r) else np.nan,
            "profit_factor": stats_utils.profit_factor(r_mult),
            "total_pnl_usd": float(r.sum()),
        })
    return pd.DataFrame(rows)


def add_size_day_range(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Descriptive/retrospective measure (not a live-tradeable signal --
    the day's own full range isn't known until the day is over): SL/TP
    size relative to THAT SPECIFIC DAY's actual realized (high-low)/open
    range, rather than the smoothed, lagging ATR(14) used above. Answers
    "was the stop tiny compared to how much price actually moved that
    exact day" rather than "compared to a 14-day rolling average."
    """
    daily = regime_detection.resample_daily(candles)
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["open"]
    daily["date"] = daily["datetime_utc"].dt.normalize()

    out = trades.copy()
    out["entry_date"] = out["dateStart_utc"].dt.normalize()
    out = out.merge(daily[["date", "range_pct"]], left_on="entry_date", right_on="date", how="left").drop(columns=["date"])

    out["sl_size_day_range"] = (out["entryPrice"] - out["initalSL"]).abs() / out["entryPrice"] / out["range_pct"]
    out["tp_size_day_range"] = (out["idealTP"] - out["entryPrice"]).abs() / out["entryPrice"] / out["range_pct"]
    return out


def threshold_scan(trades: pd.DataFrame, size_col: str, thresholds: np.ndarray) -> pd.DataFrame:
    """For each candidate threshold t, splits trades into size<=t vs
    size>t and reports both sides' performance -- a continuous sweep
    rather than a handful of fixed bands, to locate the exact point where
    performance flips (per the user's explicit request for precision)."""
    rows = []
    for t in thresholds:
        below = trades[trades[size_col] <= t]
        above = trades[trades[size_col] > t]
        r_below = below["avgRiskReward"].dropna().values
        r_above = above["avgRiskReward"].dropna().values
        rows.append({
            "threshold": t,
            "n_below": len(below), "n_above": len(above),
            "win_rate_below": float((below["rPnL"] > 0).mean()) if len(below) else np.nan,
            "win_rate_above": float((above["rPnL"] > 0).mean()) if len(above) else np.nan,
            "expectancy_below": stats_utils.expectancy(r_below),
            "expectancy_above": stats_utils.expectancy(r_above),
            "pf_below": stats_utils.profit_factor(r_below),
            "pf_above": stats_utils.profit_factor(r_above),
        })
    out = pd.DataFrame(rows)
    out["expectancy_gap"] = out["expectancy_above"] - out["expectancy_below"]
    return out


def yearly_size_drift(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Demonstrates the exact problem the user flagged: raw % of price vs
    ATR-relative size, by year -- shows whether raw % is drifting with
    NAS100's price level while the ATR-relative measure stays flat."""
    both = add_size_pct(trades)
    atr = add_size_atr(trades, candles)
    both["sl_size_atr"] = atr["sl_size_atr"]
    both["tp_size_atr"] = atr["tp_size_atr"]
    both["year"] = both["dateStart_utc"].dt.year

    return both.groupby("year").agg(
        n=("id", "count"),
        mean_entry_price=("entryPrice", "mean"),
        mean_sl_pct_of_price=("sl_size_pct", "mean"),
        mean_sl_pct_of_atr=("sl_size_atr", "mean"),
        mean_tp_pct_of_price=("tp_size_pct", "mean"),
        mean_tp_pct_of_atr=("tp_size_atr", "mean"),
    ).reset_index()


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()

    print("=== does raw %-of-price drift with NAS100's rising price level? ===")
    drift = yearly_size_drift(trades, candles)
    print(drift.to_string(index=False))

    sized_atr = add_size_atr(trades, candles)

    print("\n=== sl_size_atr describe ===")
    print(sized_atr["sl_size_atr"].describe())
    print("\n=== tp_size_atr describe ===")
    print(sized_atr["tp_size_atr"].describe())

    print("\n=== performance by SL size bucket (ATR-relative, fine) ===")
    sl_table = bucket_performance(sized_atr, "sl_size_atr", config.SL_ATR_BUCKETS, as_pct=False)
    print(sl_table.to_string(index=False))

    print("\n=== performance by TP size bucket (ATR-relative, fine) ===")
    tp_table = bucket_performance(sized_atr, "tp_size_atr", config.TP_ATR_BUCKETS, as_pct=False)
    print(tp_table.to_string(index=False))

    sized_day = add_size_day_range(trades, candles)
    print("\n=== sl_size_day_range describe (relative to THAT day's own actual range) ===")
    print(sized_day["sl_size_day_range"].describe())

    print("\n=== SL threshold scan (ATR-relative, fine grid) ===")
    scan_sl_atr = threshold_scan(sized_atr, "sl_size_atr", np.arange(0.02, 0.32, 0.02))
    print(scan_sl_atr.to_string(index=False))

    print("\n=== SL threshold scan (same-day actual range, fine grid) ===")
    scan_sl_day = threshold_scan(sized_day, "sl_size_day_range", np.arange(0.02, 0.32, 0.02))
    print(scan_sl_day.to_string(index=False))

    print("\n=== TP threshold scan (ATR-relative, fine grid) ===")
    scan_tp_atr = threshold_scan(sized_atr, "tp_size_atr", np.arange(0.05, 1.05, 0.05))
    print(scan_tp_atr.to_string(index=False))
