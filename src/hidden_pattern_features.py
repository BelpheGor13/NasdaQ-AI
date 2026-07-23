"""Phase 1: entry-context classification features (test-only; see
hidden-patterns-exit-optimization-prompt.md).

Builds on feature_engineering.build_features (regime A, recent-high/low C,
time D, sl_pct_of_atr E, momentum) and adds the remaining Phase-1 features:
volume context (A), candle body/wick + consecutive-direction (C),
previous-trade outcome (D), and risk/reward setup (E).

Resolved ambiguity (confirmed with the user): Phase-1 section B ("speed of
profit in the first 30 minutes") is POST-entry information and conflicts
with Critical Note 1's strict no-look-ahead requirement for entry-context
classification. It is computed here but kept OUT of CLUSTERING_FEATURES --
reported only as a separate, descriptive cross-tab against the clusters
(hidden_pattern_clustering.py), never used to assign a cluster. Every name
in CLUSTERING_FEATURES is provably computable strictly before dateStart_utc.
"""
import numpy as np
import pandas as pd

from src import data_loading, excursion, feature_engineering, regime_detection

BODY_WICK_LOOKBACK = 5  # candles immediately before entry (spec: "5 candles before entry")
VOLUME_LOOKBACK_MIN = 30
VOLUME_ROLLING_DAYS = 20
SPEED_OF_PROFIT_WINDOW_MIN = 30  # descriptive only -- NOT a clustering feature

CLUSTERING_FEATURES_NUMERIC = [
    "sl_pct_of_atr", "pct_dist_from_high_20d", "pct_dist_from_low_20d",
    "pct_dist_from_high_50d", "pct_dist_from_low_50d",
    "pre_entry_momentum_5m", "pre_entry_momentum_15m", "pre_entry_momentum_30m",
    "pre_entry_pct_up_candles_15m", "pre_entry_volume_ratio",
    "body_wick_ratio_5c", "consecutive_candles_direction",
    "entry_hour_utc", "risk_reward_setup",
]
CLUSTERING_FEATURES_CATEGORICAL = [
    "regime_trend_asof_prior_day", "regime_vol_asof_prior_day", "session", "prev_trade_was_win",
]


def add_volume_context(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Recent (pre-entry) volume vs its own 20-day trailing average, both
    expressed per-minute so they're comparable. The 20-day average is
    as-of the PRIOR day (shifted), matching regime_detection's convention.
    """
    daily = regime_detection.resample_daily(candles)
    daily["avg_vol_per_min_20d"] = (
        daily["volume"].rolling(VOLUME_ROLLING_DAYS, min_periods=VOLUME_ROLLING_DAYS).mean().shift(1) / 1440
    )
    daily["date"] = daily["datetime_utc"].dt.normalize()

    out = trades.copy()
    out["entry_date"] = out["dateStart_utc"].dt.normalize()
    out = out.merge(daily[["date", "avg_vol_per_min_20d"]], left_on="entry_date", right_on="date", how="left") \
              .drop(columns=["date"])

    ts = candles["datetime_utc"].values
    vol = candles["volume"].values
    starts = out["dateStart_utc"].values
    end_idx = np.searchsorted(ts, starts, side="left")

    recent_vol_per_min = np.full(len(out), np.nan)
    for i in range(len(out)):
        hi = end_idx[i]
        lo = hi - VOLUME_LOOKBACK_MIN
        if lo < 0 or hi <= lo:
            continue
        recent_vol_per_min[i] = vol[lo:hi].mean()

    out["pre_entry_volume_ratio"] = recent_vol_per_min / out["avg_vol_per_min_20d"]
    return out


def add_candle_pattern_features(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Body/wick ratio and consecutive-same-direction-candle count, using
    only candles with timestamp < dateStart_utc (entry candle excluded)."""
    ts = candles["datetime_utc"].values
    opens = candles["open"].values
    highs = candles["high"].values
    lows = candles["low"].values
    closes = candles["close"].values

    starts = trades["dateStart_utc"].values
    end_idx = np.searchsorted(ts, starts, side="left")

    body_wick = np.full(len(trades), np.nan)
    consecutive = np.full(len(trades), np.nan)

    for i in range(len(trades)):
        hi = end_idx[i]
        lo5 = hi - BODY_WICK_LOOKBACK
        if lo5 < 0 or hi <= lo5:
            continue
        o, h, l, c = opens[lo5:hi], highs[lo5:hi], lows[lo5:hi], closes[lo5:hi]
        total_range = (h - l)
        body = np.abs(c - o)
        valid = total_range > 0
        body_wick[i] = (body[valid] / total_range[valid]).mean() if valid.any() else np.nan

        # walk backward from the candle immediately before entry, counting
        # consecutive candles in the same up/down direction
        directions = np.sign(c - o)
        last_dir = directions[-1]
        if last_dir == 0:
            consecutive[i] = 0
        else:
            count = 0
            for d in directions[::-1]:
                if d == last_dir:
                    count += 1
                else:
                    break
            consecutive[i] = count

    out = trades.copy()
    out["body_wick_ratio_5c"] = body_wick
    out["consecutive_candles_direction"] = consecutive
    return out


def add_previous_trade_outcome(trades: pd.DataFrame) -> pd.DataFrame:
    """Outcome of the most recently CLOSED other trade strictly before this
    trade's dateStart_utc -- using dateEnd_utc (not entry order) so an
    overlapping-but-still-open prior trade is never used as if its outcome
    were already known.
    """
    out = trades.sort_values("dateStart_utc").reset_index(drop=True)
    closed = out[["dateEnd_utc", "is_win"]].sort_values("dateEnd_utc").reset_index(drop=True)

    merged = pd.merge_asof(
        out.sort_values("dateStart_utc"), closed.rename(columns={"is_win": "prev_trade_was_win"}),
        left_on="dateStart_utc", right_on="dateEnd_utc", direction="backward",
        allow_exact_matches=False, suffixes=("", "_prev"),
    )
    out["prev_trade_was_win"] = merged["prev_trade_was_win"].values
    out["prev_trade_was_win"] = out["prev_trade_was_win"].map({True: "prev_win", False: "prev_loss"})
    out["prev_trade_was_win"] = out["prev_trade_was_win"].fillna("no_prior_trade")
    return out


def add_risk_reward_setup(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    risk = (out["entryPrice"] - out["initalSL"]).abs()
    reward = np.where(out["side"] == "buy", out["idealTP"] - out["entryPrice"], out["entryPrice"] - out["idealTP"])
    out["risk_reward_setup"] = reward / risk.replace(0, np.nan)
    return out


def add_speed_of_profit(trades: pd.DataFrame, candles: pd.DataFrame,
                         window_min: int = SPEED_OF_PROFIT_WINDOW_MIN) -> pd.DataFrame:
    """Descriptive-only (post-entry) feature: R gained in the first
    window_min minutes after entry. NOT part of CLUSTERING_FEATURES.
    """
    ts = candles["datetime_utc"].values
    highs = candles["high"].values
    lows = candles["low"].values

    starts = trades["dateStart_utc"].values
    ends = trades["dateEnd_utc"].values
    window_end = starts + np.timedelta64(window_min, "m")
    capped_end = np.minimum(window_end, ends)

    start_idx = np.searchsorted(ts, starts, side="left")
    end_idx = np.searchsorted(ts, capped_end, side="right")

    entry = trades["entryPrice"].values
    sl = trades["initalSL"].values
    side = trades["side"].values
    risk = np.abs(entry - sl)
    risk = np.where(risk == 0, np.nan, risk)

    speed_r = np.full(len(trades), np.nan)
    for i in range(len(trades)):
        lo, hi = start_idx[i], end_idx[i]
        if hi <= lo:
            continue
        if side[i] == "buy":
            speed_r[i] = (highs[lo:hi].max() - entry[i]) / risk[i]
        else:
            speed_r[i] = (entry[i] - lows[lo:hi].min()) / risk[i]

    out = trades.copy()
    out["speed_of_profit_r_30m"] = speed_r
    bins = [-np.inf, 0.5, 1.5, 3.0, np.inf]
    labels = ["<0.5R", "0.5-1.5R", "1.5-3R", ">3R"]
    out["speed_of_profit_bucket"] = pd.cut(out["speed_of_profit_r_30m"], bins=bins, labels=labels)
    return out


def build_hidden_pattern_features(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    out = feature_engineering.build_features(trades, candles)
    out = add_volume_context(out, candles)
    out = add_candle_pattern_features(out, candles)
    out = add_previous_trade_outcome(out)
    out = add_risk_reward_setup(out)
    out = add_speed_of_profit(out, candles)  # descriptive only, see module docstring
    return out


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = build_hidden_pattern_features(trades, candles)

    print("=== nulls in clustering features ===")
    for c in CLUSTERING_FEATURES_NUMERIC + CLUSTERING_FEATURES_CATEGORICAL:
        print(f"  {c}: {feats[c].isnull().sum()} / {len(feats)}")

    print()
    print("=== speed_of_profit_bucket (descriptive only) ===")
    print(feats["speed_of_profit_bucket"].value_counts())

    print()
    print("=== prev_trade_was_win ===")
    print(feats["prev_trade_was_win"].value_counts())

    print()
    print(feats[["body_wick_ratio_5c", "consecutive_candles_direction",
                 "pre_entry_volume_ratio", "risk_reward_setup"]].describe())
