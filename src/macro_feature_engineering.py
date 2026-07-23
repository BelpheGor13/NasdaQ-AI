"""Pre-entry macro/cross-asset features (test-only; answers a direct
question about gold/dollar/EUR/VIX/equity-market composite correlations).
Same no-look-ahead discipline as regime_detection.py: every feature is
shifted so a trade entered on day T only ever sees macro data known as of
the close of day T-1.

This is deliberately kept SEPARATE from feature_engineering.py / the core
pipeline (which the original spec required to be self-contained on
NAS100's own candles) -- these are additional, disclosed-as-external
features for one specific follow-up question, not part of the core
discovery pipeline's feature set.
"""
import numpy as np
import pandas as pd

from src import external_data, regime_detection

CHANGE_WINDOW_DAYS = 5
N_QUANTILE_BINS = 3

RAW_SERIES = ["vix", "dxy_proxy", "eurusd", "sp500", "gold"]


def build_macro_features(candles: pd.DataFrame, force_refetch: bool = False) -> pd.DataFrame:
    daily = external_data.build_macro_daily_table(force=force_refetch)

    for col in RAW_SERIES:
        daily[f"{col}_chg_{CHANGE_WINDOW_DAYS}d"] = daily[col].pct_change(CHANGE_WINDOW_DAYS)

    daily["vix_pct_rank"] = regime_detection._expanding_pct_rank(daily["vix"], min_periods=60)
    daily["vix_regime"] = np.where(daily["vix_pct_rank"] > 0.5, "High Vol", "Low Vol")
    daily.loc[daily["vix_pct_rank"].isna(), "vix_regime"] = np.nan

    nas_daily = regime_detection.resample_daily(candles)
    nas_daily["nas100_chg_5d"] = nas_daily["close"].pct_change(CHANGE_WINDOW_DAYS)
    nas_daily["date"] = nas_daily["datetime_utc"].dt.normalize()

    daily = daily.merge(nas_daily[["date", "nas100_chg_5d"]], on="date", how="left")
    daily["nas100_vs_spx_relative_5d"] = daily["nas100_chg_5d"] - daily["sp500_chg_5d"]

    shift_cols = [c for c in daily.columns if c not in ("date",)]
    for c in shift_cols:
        daily[c + "_asof_prior_day"] = daily[c].shift(1)

    return daily


def attach_macro_to_trades(trades: pd.DataFrame, macro_daily: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["entry_date"] = out["dateStart_utc"].dt.normalize()
    cols = [c for c in macro_daily.columns if c.endswith("_asof_prior_day")]
    merged = out.merge(macro_daily[["date"] + cols], left_on="entry_date", right_on="date", how="left") \
                .drop(columns=["date"])
    return merged


MACRO_CATEGORICAL_FEATURES = ["vix_regime_asof_prior_day"]
MACRO_CONTINUOUS_FEATURES = [
    "vix_asof_prior_day", f"vix_chg_{CHANGE_WINDOW_DAYS}d_asof_prior_day",
    "dxy_proxy_asof_prior_day", f"dxy_proxy_chg_{CHANGE_WINDOW_DAYS}d_asof_prior_day",
    "eurusd_asof_prior_day", f"eurusd_chg_{CHANGE_WINDOW_DAYS}d_asof_prior_day",
    "gold_asof_prior_day", f"gold_chg_{CHANGE_WINDOW_DAYS}d_asof_prior_day",
    f"sp500_chg_{CHANGE_WINDOW_DAYS}d_asof_prior_day",
    "nas100_vs_spx_relative_5d_asof_prior_day",
]


if __name__ == "__main__":
    from src import data_loading

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()

    macro_daily = build_macro_features(candles)
    merged = attach_macro_to_trades(trades, macro_daily)

    print(f"trades: {len(merged)}")
    print("nulls in macro features:")
    for c in MACRO_CATEGORICAL_FEATURES + MACRO_CONTINUOUS_FEATURES:
        print(f"  {c}: {merged[c].isnull().sum()}")
    print()
    print(merged[MACRO_CONTINUOUS_FEATURES].describe())
