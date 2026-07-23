"""Stage 4: market regime classification and change-point detection.

Regime is computed on daily bars resampled from the 1-min candles (a
"regime" is a slow-moving concept; running ADX/ATR on 1-min data is mostly
noise). Every rolling statistic uses an EXPANDING window (or a trailing
window shifted by one full day) so that a day's regime label only reflects
information available strictly before that day -- required because regime
is later used as a pre-entry feature.

Thresholds (documented, not silently assumed):
  - ADX(14) > 25  => Trend, else Range          (standard textbook cutoff)
  - ATR(14) expanding percentile > 50           => High Vol, else Low Vol
  - Bollinger Band(20,2) width expanding pct>50 => Expansion, else Compression
"""
import numpy as np
import pandas as pd

from src import data_loading

ADX_TREND_THRESHOLD = 25.0
ATR_WINDOW = 14
ADX_WINDOW = 14
BB_WINDOW = 20
BB_STD = 2.0
MIN_EXPANDING_PERIODS = 60  # ~3 months of daily bars before percentile ranks are trusted


def resample_daily(candles: pd.DataFrame) -> pd.DataFrame:
    c = candles.set_index("datetime_utc")
    daily = c.resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    daily = daily.dropna(subset=["open", "high", "low", "close"])
    return daily.reset_index()


def _atr(daily: pd.DataFrame, window: int) -> pd.Series:
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _adx(daily: pd.DataFrame, window: int) -> pd.Series:
    up_move = daily["high"].diff()
    down_move = -daily["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _atr(daily, window)
    plus_di = 100 * pd.Series(plus_dm, index=daily.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=daily.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window
    ).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return adx


def _bb_width(daily: pd.DataFrame, window: int, n_std: float) -> pd.Series:
    mid = daily["close"].rolling(window, min_periods=window).mean()
    std = daily["close"].rolling(window, min_periods=window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return (upper - lower) / mid


def _expanding_pct_rank(series: pd.Series, min_periods: int) -> pd.Series:
    """Percentile rank of each value against ALL prior values only (no look-ahead)."""
    def pct_of_last(x):
        if len(x) < min_periods:
            return np.nan
        return (x.iloc[:-1] < x.iloc[-1]).mean()

    return series.expanding(min_periods=min_periods).apply(pct_of_last, raw=False)


def compute_regime(candles: pd.DataFrame) -> pd.DataFrame:
    daily = resample_daily(candles)

    daily["atr14"] = _atr(daily, ATR_WINDOW)
    daily["adx14"] = _adx(daily, ADX_WINDOW)
    daily["bb_width"] = _bb_width(daily, BB_WINDOW, BB_STD)
    daily["realized_vol_20d"] = daily["close"].pct_change().rolling(20).std() * np.sqrt(252)

    daily["atr_pct_rank"] = _expanding_pct_rank(daily["atr14"], MIN_EXPANDING_PERIODS)
    daily["bb_width_pct_rank"] = _expanding_pct_rank(daily["bb_width"], MIN_EXPANDING_PERIODS)

    # Built with pandas .loc rather than nested np.where: np.where silently
    # coerces a mixed str/NaN result to a common string dtype, turning real
    # missing values into the literal string "nan" instead of leaving them
    # as missing -- which then survives downstream as a bogus category.
    daily["regime_trend"] = pd.Series(np.nan, index=daily.index, dtype=object)
    daily.loc[daily["adx14"] > ADX_TREND_THRESHOLD, "regime_trend"] = "Trend"
    daily.loc[daily["adx14"] <= ADX_TREND_THRESHOLD, "regime_trend"] = "Range"

    daily["regime_vol"] = pd.Series(np.nan, index=daily.index, dtype=object)
    daily.loc[daily["atr_pct_rank"] > 0.5, "regime_vol"] = "High Vol"
    daily.loc[daily["atr_pct_rank"] <= 0.5, "regime_vol"] = "Low Vol"

    daily["regime_expansion"] = pd.Series(np.nan, index=daily.index, dtype=object)
    daily.loc[daily["bb_width_pct_rank"] > 0.5, "regime_expansion"] = "Expansion"
    daily.loc[daily["bb_width_pct_rank"] <= 0.5, "regime_expansion"] = "Compression"

    # Shift by 1 day: a trade entered on day T may only use regime info
    # known as of the close of day T-1.
    for col in ["regime_trend", "regime_vol", "regime_expansion", "atr14", "adx14",
                "bb_width", "realized_vol_20d", "atr_pct_rank", "bb_width_pct_rank"]:
        daily[col + "_asof_prior_day"] = daily[col].shift(1)

    daily["date"] = daily["datetime_utc"].dt.normalize()
    return daily


def cusum_change_points(series: pd.Series, dates: pd.Series, threshold_std: float = 4.0,
                         drift_std: float = 0.5, min_gap_days: int = 21) -> list:
    """Classic two-sided CUSUM change-point detector.

    threshold_std / drift_std are expressed in units of the series' own
    std dev so the method is scale-free across different input series.
    Returns a list of dates where the cumulative deviation from the running
    mean crossed the threshold (and the accumulator was reset).

    A single regime shift (e.g. a volatility spike that persists for weeks)
    naturally re-triggers the detector many times in a row since the
    expanding mean/std adapt slowly. min_gap_days merges detections that
    fall within the same window into one event, reported at its first date,
    so the output reflects distinct regime shifts rather than every re-trigger.
    """
    x = series.dropna()
    if len(x) < 30:
        return []

    mu = x.expanding(min_periods=20).mean()
    sigma = x.expanding(min_periods=20).std()
    drift = drift_std * sigma
    thresh = threshold_std * sigma

    pos, neg = 0.0, 0.0
    change_dates = []
    vals = x.values
    idx_dates = dates.loc[x.index].values
    mu_v, thresh_v, drift_v = mu.values, thresh.values, drift.values

    for i in range(1, len(vals)):
        if np.isnan(thresh_v[i]) or np.isnan(mu_v[i]):
            continue
        pos = max(0, pos + vals[i] - mu_v[i] - drift_v[i])
        neg = min(0, neg + vals[i] - mu_v[i] + drift_v[i])
        if pos > thresh_v[i] or -neg > thresh_v[i]:
            change_dates.append(idx_dates[i])
            pos, neg = 0.0, 0.0

    change_dates = pd.to_datetime(change_dates)
    if len(change_dates) == 0:
        return []

    merged = [change_dates[0]]
    for d in change_dates[1:]:
        if (d - merged[-1]).days > min_gap_days:
            merged.append(d)
    return list(merged)


def attach_regime_to_trades(trades: pd.DataFrame, regime_daily: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["entry_date"] = out["dateStart_utc"].dt.normalize()

    cols = [c for c in regime_daily.columns if c.endswith("_asof_prior_day")]
    merged = out.merge(
        regime_daily[["date"] + cols], left_on="entry_date", right_on="date", how="left"
    ).drop(columns=["date"])
    return merged


if __name__ == "__main__":
    candles = data_loading.load_candles()
    regime = compute_regime(candles)
    print(regime[["date", "regime_trend", "regime_vol", "regime_expansion"]].dropna().tail(10))
    print()
    print("Trend distribution:", regime["regime_trend"].value_counts().to_dict())
    print("Vol distribution:", regime["regime_vol"].value_counts().to_dict())
    print("Expansion distribution:", regime["regime_expansion"].value_counts().to_dict())

    cps = cusum_change_points(regime["atr14"], regime["date"])
    print(f"\nATR CUSUM change points ({len(cps)}):")
    for d in cps:
        print(" ", d.date())

    trades = data_loading.load_trades()
    merged = attach_regime_to_trades(trades, regime)
    print(f"\ntrades with missing regime (insufficient warmup history): "
          f"{merged['regime_trend_asof_prior_day'].isna().sum()} / {len(merged)}")
