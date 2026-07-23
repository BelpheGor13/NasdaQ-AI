"""Per-configuration performance metrics for the trailing-stop grid sweep
(deliverable 1): win rate, expectancy, profit factor, Sharpe, max drawdown,
each vs the original (no-trailing) baseline.

Sharpe here is computed on R-multiples, annualized using the trade
frequency actually observed in the data (n_trades / n_years spanned) rather
than a hardcoded 252, since these are event-driven (per-trade) returns, not
daily bars. Max drawdown is reported in R (peak-to-trough decline of the
cumulative-R equity curve, in chronological trade order) -- the standard
convention for R-multiple-based systems, where a percentage drawdown is not
well-defined (cumulative R can cross zero).
"""
import numpy as np
import pandas as pd

from src import stats_utils


def _years_spanned(dates: pd.Series) -> float:
    span_days = (dates.max() - dates.min()).total_seconds() / 86400
    return max(span_days / 365.25, 1e-9)


def _sharpe(r_values: np.ndarray, trades_per_year: float) -> float:
    if len(r_values) < 2 or r_values.std(ddof=1) == 0:
        return np.nan
    return float(r_values.mean() / r_values.std(ddof=1) * np.sqrt(trades_per_year))


def _max_drawdown_r(r_values_chronological: np.ndarray) -> float:
    if len(r_values_chronological) == 0:
        return np.nan
    equity = np.cumsum(r_values_chronological)
    running_peak = np.maximum.accumulate(equity)
    drawdown = running_peak - equity
    return float(drawdown.max())


def _config_metrics(df_sorted: pd.DataFrame, r_col: str, trades_per_year: float) -> dict:
    r = df_sorted[r_col].dropna().values
    n = len(r)
    return {
        "n": n,
        "win_rate": float((r > 0).mean()) if n else np.nan,
        "expectancy": stats_utils.expectancy(r),
        "profit_factor": stats_utils.profit_factor(r),
        "sharpe": _sharpe(r, trades_per_year),
        "max_drawdown_r": _max_drawdown_r(r),
    }


def build_summary_table(sim: pd.DataFrame, scenario: str = "conservative") -> pd.DataFrame:
    """One row per trailing_pct (+ a 'baseline' row for the original, unmodified
    exits), each metric computed on that configuration's r-values in
    chronological trade order.
    """
    s = sim[sim["scenario"] == scenario].copy()
    s = s.sort_values("dateStart_utc")
    trades_per_year = len(s["id"].unique()) / _years_spanned(s["dateStart_utc"])

    rows = []
    baseline = s.drop_duplicates("id").sort_values("dateStart_utc")
    base_metrics = _config_metrics(baseline, "orig_r", trades_per_year)
    base_metrics["config"] = "baseline (no trailing)"
    rows.append(base_metrics)

    for pct, group in s.groupby("pct"):
        m = _config_metrics(group, "trail_r", trades_per_year)
        m["config"] = f"{pct*100:.0f}%"
        m["pct"] = pct
        rows.append(m)

    out = pd.DataFrame(rows)
    base_pf = out.loc[out["config"] == "baseline (no trailing)", "profit_factor"].values[0]
    base_exp = out.loc[out["config"] == "baseline (no trailing)", "expectancy"].values[0]
    out["pf_vs_baseline_pct_change"] = (out["profit_factor"] - base_pf) / base_pf * 100
    out["expectancy_vs_baseline_pct_change"] = (out["expectancy"] - base_exp) / abs(base_exp) * 100
    cols = ["config", "pct", "n", "win_rate", "expectancy", "profit_factor", "sharpe",
            "max_drawdown_r", "pf_vs_baseline_pct_change", "expectancy_vs_baseline_pct_change"]
    return out[cols]


if __name__ == "__main__":
    from src import data_loading, trailing_stop_simulation

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = trailing_stop_simulation.simulate_trailing_stops(trades, candles)

    for scenario in ("conservative", "aggressive"):
        print(f"\n=== {scenario} ===")
        table = build_summary_table(sim, scenario=scenario)
        print(table.to_string(index=False))
