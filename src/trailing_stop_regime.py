"""Regime check (critical note in the prompt): does the best trailing pct
work uniformly across market regimes (trend/range, high/low vol), or only
in specific ones? Reuses regime_detection.py's existing daily regime labels
(as-of-prior-day, no look-ahead) rather than deriving a new classification.
"""
import pandas as pd

from src import data_loading, regime_detection, stats_utils


def attach_regime(sim: pd.DataFrame, candles: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    regime_daily = regime_detection.compute_regime(candles)
    regime_trades = regime_detection.attach_regime_to_trades(trades, regime_daily)
    regime_cols = regime_trades[["id", "regime_trend_asof_prior_day", "regime_vol_asof_prior_day"]]
    return sim.merge(regime_cols, on="id", how="left")


def performance_by_regime(sim_with_regime: pd.DataFrame, pct: float, scenario: str = "conservative") -> pd.DataFrame:
    g = sim_with_regime[(sim_with_regime["scenario"] == scenario) & (sim_with_regime["pct"] == pct)]

    rows = []
    for regime_col, regime_name in [("regime_trend_asof_prior_day", "trend"),
                                     ("regime_vol_asof_prior_day", "volatility")]:
        for label, group in g.groupby(regime_col):
            trail_r = group["trail_r"].dropna().values
            orig_r = group["orig_r"].dropna().values
            rows.append({
                "regime_dimension": regime_name,
                "regime_label": label,
                "n": len(trail_r),
                "trailing_expectancy": stats_utils.expectancy(trail_r),
                "trailing_profit_factor": stats_utils.profit_factor(trail_r),
                "baseline_expectancy": stats_utils.expectancy(orig_r),
                "baseline_profit_factor": stats_utils.profit_factor(orig_r),
            })
    out = pd.DataFrame(rows)
    out["trailing_beats_baseline"] = out["trailing_profit_factor"] > out["baseline_profit_factor"]
    return out


if __name__ == "__main__":
    from src import trailing_stop_simulation, trailing_stop_metrics

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = trailing_stop_simulation.simulate_trailing_stops(trades, candles)
    summary = trailing_stop_metrics.build_summary_table(sim, scenario="conservative")
    best_pct = summary[summary["config"] != "baseline (no trailing)"].sort_values(
        "profit_factor", ascending=False).iloc[0]["pct"]

    sim_regime = attach_regime(sim, candles, trades)
    result = performance_by_regime(sim_regime, best_pct)
    print(f"best pct: {best_pct}")
    print(result.to_string(index=False))
