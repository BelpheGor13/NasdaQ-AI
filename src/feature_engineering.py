"""Stage 5: pre-entry composite features. Every feature here must be
computable using ONLY information strictly before dateStart_utc -- this is
the boundary between "tradeable signal" and "descriptive research" that the
rest of the pipeline depends on. Nothing from excursion.py or
exit_quality.py (which use candles during/after entry) belongs here.
"""
import numpy as np
import pandas as pd

from src import data_loading, regime_detection

# UTC hour session buckets (NAS100 trades ~23.5h/day; buckets follow the
# conventional FX/index session split).
SESSION_BOUNDS = [
    (0, 7, "Asia"),
    (7, 12, "London"),
    (12, 16, "NY_Open_Overlap"),
    (16, 21, "NY_Afternoon"),
    (21, 24, "Late_NY_PreAsia"),
]

RECENT_HL_WINDOWS = (20, 50)  # trading days
MOMENTUM_LOOKBACK_MIN = (5, 15, 30)  # minutes before entry


def _session_bucket(hour: int) -> str:
    for lo, hi, name in SESSION_BOUNDS:
        if lo <= hour < hi:
            return name
    return "Unknown"


def add_time_features(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["entry_hour_utc"] = out["dateStart_utc"].dt.hour
    out["session"] = out["entry_hour_utc"].apply(_session_bucket)
    out["day_of_week"] = out["day"]  # 1=Mon..5=Fri, NY calendar (verified against dateStart)
    return out


def add_regime_features(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    daily = regime_detection.compute_regime(candles)
    out = regime_detection.attach_regime_to_trades(trades, daily)

    for w in RECENT_HL_WINDOWS:
        roll_high = daily["high"].rolling(w, min_periods=w).max().shift(1)
        roll_low = daily["low"].rolling(w, min_periods=w).min().shift(1)
        daily[f"roll_high_{w}d"] = roll_high
        daily[f"roll_low_{w}d"] = roll_low

    daily["date"] = daily["datetime_utc"].dt.normalize()
    hl_cols = [f"roll_high_{w}d" for w in RECENT_HL_WINDOWS] + [f"roll_low_{w}d" for w in RECENT_HL_WINDOWS]
    out = out.merge(daily[["date"] + hl_cols], left_on="entry_date", right_on="date", how="left").drop(columns=["date"])

    for w in RECENT_HL_WINDOWS:
        hi, lo = out[f"roll_high_{w}d"], out[f"roll_low_{w}d"]
        rng = (hi - lo).replace(0, np.nan)
        out[f"pct_dist_from_high_{w}d"] = (hi - out["entryPrice"]) / rng
        out[f"pct_dist_from_low_{w}d"] = (out["entryPrice"] - lo) / rng

    # entryPrice/initalSL are known at entry time (order parameters), unlike
    # excursion.py's risk_price which is only *derived alongside* MFE/MAE --
    # computed independently here so this module has no dependency on
    # in-trade candle data.
    risk_price = (out["entryPrice"] - out["initalSL"]).abs()
    out["sl_pct_of_atr"] = risk_price / out["atr14_asof_prior_day"]
    return out


def add_pre_entry_candle_features(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Short-horizon momentum/candle features computed strictly on 1-min
    candles with a timestamp < dateStart_utc (the entry candle itself is
    excluded to guarantee no look-ahead into the bar the trade opens on)."""
    ts = candles["datetime_utc"].values
    closes = candles["close"].values
    opens = candles["open"].values

    starts = trades["dateStart_utc"].values
    end_idx = np.searchsorted(ts, starts, side="left")  # first idx NOT before entry

    out = trades.copy()
    for lb in MOMENTUM_LOOKBACK_MIN:
        mom = np.full(len(trades), np.nan)
        n_up = np.full(len(trades), np.nan)
        for i in range(len(trades)):
            hi = end_idx[i]
            lo = hi - lb
            if lo < 0 or hi <= lo:
                continue
            window_closes = closes[lo:hi]
            window_opens = opens[lo:hi]
            mom[i] = (window_closes[-1] - window_opens[0]) / window_opens[0]
            n_up[i] = (window_closes > window_opens).mean()
        out[f"pre_entry_momentum_{lb}m"] = mom
        out[f"pre_entry_pct_up_candles_{lb}m"] = n_up

    return out


def build_features(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features(trades)
    out = add_regime_features(out, candles)
    out = add_pre_entry_candle_features(out, candles)
    return out


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = build_features(trades, candles)

    print(feats["session"].value_counts())
    print()
    print(feats[["sl_pct_of_atr", "pct_dist_from_high_20d", "pct_dist_from_low_20d",
                 "pre_entry_momentum_15m"]].describe())
    print()
    print("nulls in key features:")
    for c in ["session", "regime_trend_asof_prior_day", "regime_vol_asof_prior_day",
              "sl_pct_of_atr", "pct_dist_from_high_20d", "pre_entry_momentum_15m"]:
        print(f"  {c}: {feats[c].isnull().sum()}")
