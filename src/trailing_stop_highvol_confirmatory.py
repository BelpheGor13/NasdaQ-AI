"""Confirmatory follow-up to trailing_stop_regime.py's exploratory finding:
the best trailing-stop config (by profit factor, conservative scenario)
beat baseline ONLY in the High Vol regime slice (PF 1.38 vs 1.30), while
losing in every other regime slice. That discovery came from scanning 4
regime slices, so it needs its own dedicated confirmatory test rather than
being taken at face value -- this runs ONE pre-registered paired
significance test + Monte Carlo bootstrap, restricted to High Vol trades
only, using the exact same config and methodology as
trailing_stop_significance.py / trailing_stop_monte_carlo.py.
"""
from src import (
    data_loading, trailing_stop_simulation, trailing_stop_metrics,
    trailing_stop_regime, trailing_stop_significance, trailing_stop_monte_carlo,
)


def run_highvol_confirmatory_test(scenario: str = "conservative"):
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()

    sim = trailing_stop_simulation.simulate_trailing_stops(trades, candles)
    summary = trailing_stop_metrics.build_summary_table(sim, scenario=scenario)
    best_pct = summary[summary["config"] != "baseline (no trailing)"].sort_values(
        "profit_factor", ascending=False).iloc[0]["pct"]

    sim_regime = trailing_stop_regime.attach_regime(sim, candles, trades)
    high_vol = sim_regime[
        (sim_regime["scenario"] == scenario) & (sim_regime["pct"] == best_pct) &
        (sim_regime["regime_vol_asof_prior_day"] == "High Vol")
    ].sort_values("id")

    sig = trailing_stop_significance.paired_ttest(high_vol["trail_r"].values, high_vol["orig_r"].values)
    mc = trailing_stop_monte_carlo.bootstrap_pf_ci(high_vol["trail_r"].values, high_vol["orig_r"].values)

    return {"best_pct": best_pct, "n": len(high_vol), "significance": sig, "monte_carlo": mc}


if __name__ == "__main__":
    result = run_highvol_confirmatory_test()
    print(f"config tested: {result['best_pct']*100:.0f}% trailing, High Vol trades only, n={result['n']}")
    print()
    print("=== paired significance (trailing vs baseline, High Vol subset only) ===")
    for k, v in result["significance"].items():
        print(f"  {k}: {v}")
    print()
    print("=== Monte Carlo bootstrap PF CI (High Vol subset only) ===")
    for k, v in result["monte_carlo"].items():
        print(f"  {k}: {v}")
