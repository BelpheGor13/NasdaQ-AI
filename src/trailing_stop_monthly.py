"""Monthly PnL comparison (deliverable 4): original vs best-trailing-config
PnL aggregated by the trade's entry month (YYYY-MM), in USD (rPnL-style,
i.e. r_multiple * risk_price * amount -- same dollar-unit convention
exit_quality.py already uses for left_on_table_usd).
"""
import pandas as pd


def monthly_comparison(sim: pd.DataFrame, best_pct: float, scenario: str = "conservative") -> pd.DataFrame:
    g = sim[(sim["scenario"] == scenario) & (sim["pct"] == best_pct)]

    monthly = g.groupby("month").agg(
        original_pnl_usd=("orig_pnl_usd", "sum"),
        trailing_pnl_usd=("trail_pnl_usd", "sum"),
        n_trades=("id", "count"),
    ).reset_index().sort_values("month")

    monthly["difference_usd"] = monthly["trailing_pnl_usd"] - monthly["original_pnl_usd"]
    monthly["direction"] = monthly["difference_usd"].apply(
        lambda d: "improved" if d > 0 else ("degraded" if d < 0 else "unchanged"))
    return monthly


if __name__ == "__main__":
    from src import data_loading, trailing_stop_simulation, trailing_stop_metrics

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = trailing_stop_simulation.simulate_trailing_stops(trades, candles)
    summary = trailing_stop_metrics.build_summary_table(sim, scenario="conservative")
    best_pct = summary[summary["config"] != "baseline (no trailing)"].sort_values(
        "profit_factor", ascending=False).iloc[0]["pct"]

    monthly = monthly_comparison(sim, best_pct)
    print(monthly.to_string(index=False))
    print()
    print(monthly["direction"].value_counts())
