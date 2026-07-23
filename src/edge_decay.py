"""Stage 13: is the strategy's edge shrinking over time, and is that trend
itself real rather than one bad year?

Two tests, deliberately different in nature:
  1. Spearman correlation between trade sequence order and r_multiple --
     uses all ~789 trades (much more power than 5 yearly points).
  2. Yearly Profit Factor / expectancy / win rate table -- descriptive,
     shown per spec, but NOT used alone to claim a trend (n=5 years has
     very low statistical power for a regression).
Change points are detected on the cumulative-R equity curve (CUSUM, same
method as regime_detection.py) and cross-referenced against the market
regime change points so a decay can be tied to "since when" rather than
asserted in the abstract.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src import regime_detection, stats_utils


def yearly_performance(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["year"] = d["dateStart_utc"].dt.year
    rows = []
    for year, group in d.groupby("year"):
        vals = group["r_multiple"].dropna().values
        rows.append({
            "year": year,
            "n": len(vals),
            "win_rate": float((vals > 0).mean()) if len(vals) else np.nan,
            "expectancy": stats_utils.expectancy(vals),
            "profit_factor": stats_utils.profit_factor(vals),
        })
    return pd.DataFrame(rows)


def sequence_trend_test(df: pd.DataFrame) -> dict:
    ordered = df.sort_values("dateStart_utc").reset_index(drop=True)
    vals = ordered["r_multiple"].dropna()
    seq = np.arange(len(vals))

    rho, p_spearman = stats.spearmanr(seq, vals.values)
    slope, intercept, r_lin, p_lin, se = stats.linregress(seq, vals.values)

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_spearman),
        "linreg_slope_r_per_trade": float(slope),
        "linreg_p": float(p_lin),
        "direction": "decaying" if rho < 0 else "improving",
        "significant_at_0.05": bool(p_spearman < 0.05),
    }


def equity_curve_change_points(df: pd.DataFrame, smoothing_window: int = 20, threshold_std: float = 4.5,
                                drift_std: float = 0.5, min_gap_days: int = 60) -> list:
    """CUSUM runs on a rolling mean of r_multiple, not the raw per-trade
    series -- raw R-multiples are noisy enough (essentially near-iid win/
    loss outcomes) that CUSUM on the unsmoothed series fires on ordinary
    variance rather than genuine level shifts in the strategy's edge.
    """
    ordered = df.sort_values("dateStart_utc").reset_index(drop=True)
    smoothed = ordered["r_multiple"].rolling(smoothing_window, min_periods=smoothing_window // 2).mean().dropna()
    dates = ordered.loc[smoothed.index, "dateStart_utc"]
    return regime_detection.cusum_change_points(smoothed, dates, threshold_std=threshold_std,
                                                 drift_std=drift_std, min_gap_days=min_gap_days)


def tie_decay_to_regime_changes(equity_cps: list, regime_cps: list, window_days: int = 30) -> list:
    """For each equity/performance change point, report whether a market
    volatility regime change point (from regime_detection) occurred within
    window_days -- a coincidence suggests the decay tracks market character
    rather than being strategy-specific."""
    ties = []
    for ecp in equity_cps:
        nearby = [rcp for rcp in regime_cps if abs((rcp - ecp).days) <= window_days]
        ties.append({"equity_change_point": ecp, "nearby_regime_change_points": nearby,
                     "coincides_with_regime_shift": len(nearby) > 0})
    return ties


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    print("=== Yearly performance ===")
    print(yearly_performance(feats).to_string(index=False))
    print()

    print("=== Sequence-level trend test (n=789, more power than yearly) ===")
    trend = sequence_trend_test(feats)
    for k, v in trend.items():
        print(f"  {k}: {v}")
    print()

    print("=== Equity curve change points ===")
    eq_cps = equity_curve_change_points(feats)
    for d in eq_cps:
        print(" ", d.date())

    print()
    print("=== Market regime change points (ATR-based, from regime_detection) ===")
    candles_daily = regime_detection.compute_regime(candles)
    regime_cps = regime_detection.cusum_change_points(candles_daily["atr14"], candles_daily["date"])

    print("=== Ties between equity change points and regime shifts (+/-30d) ===")
    for tie in tie_decay_to_regime_changes(eq_cps, regime_cps):
        print(f"  {tie['equity_change_point'].date()}: coincides={tie['coincides_with_regime_shift']}, "
              f"nearby={[d.date() for d in tie['nearby_regime_change_points']]}")
