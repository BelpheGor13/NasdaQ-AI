"""Data-quality finding (discovered after user pushback on the hidden-
pattern report): `idealTP` is NOT a stable pre-entry target for every
trade. For every trade where avgRiskReward == -1.0 (a full stop-loss
exit, 465 / 789 trades = 59%), idealTP has been reset to a value close to
the actual exit rather than preserving whatever target was set at entry --
tells: maxTP is null for 100% of these rows and non-null for 100% of
winners, and the idealTP-implied R:R is < 2 for 458 of these 465 rows
(median ~ -0.3 to -0.9R, i.e. ON THE LOSING SIDE of entry), while for the
other 324 trades idealTP-implied R:R has median 4.57 and is >= 2 for 86%
of rows -- consistent with the user's stated house rule of a >=1:2 minimum
R:R set at entry.

initalSL was checked and is NOT affected (0 nulls, always on the correct
side of entry for its trade direction) -- only idealTP is corrupted, and
only for full-stop-loss trades.

Practical consequence: any analysis that used idealTP as if it were a
reliable pre-entry value (the "let it ride to idealTP" exit-strategy
result in hidden_pattern_report_arabic.md, and the TP-size-based findings
in pattern_deepdive_report_arabic.md) is invalid for the 465 corrupted
rows. This module quantifies the impact and provides the corrected,
clean-subset-only numbers.
"""
import numpy as np
import pandas as pd

from src import data_loading, stats_utils


def classify_idealtp_trust(trades: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    out["idealtp_trustworthy"] = out["avgRiskReward"] != -1.0
    return out


def summarize_corruption(trades: pd.DataFrame) -> dict:
    entry, sl, tp, side = trades["entryPrice"], trades["initalSL"], trades["idealTP"], trades["side"]
    risk = (entry - sl).abs()
    reward = np.where(side == "buy", tp - entry, entry - tp)
    computed_rr = reward / risk

    full_loss = trades["avgRiskReward"] == -1.0
    return {
        "n_total": len(trades),
        "n_corrupted_full_loss": int(full_loss.sum()),
        "n_trustworthy": int((~full_loss).sum()),
        "pct_corrupted_with_rr_below_2": float((computed_rr[full_loss] < 2).mean()),
        "median_rr_corrupted": float(computed_rr[full_loss].median()),
        "median_rr_trustworthy": float(computed_rr[~full_loss].median()),
        "pct_trustworthy_with_rr_at_least_2": float((computed_rr[~full_loss] >= 2).mean()),
        "maxtp_null_rate_full_loss": float(trades.loc[full_loss, "maxTP"].isnull().mean()),
        "maxtp_null_rate_winners": float(trades.loc[trades["avgRiskReward"] > 0, "maxTP"].isnull().mean()),
    }


def quantify_impact_on_hidden_pattern_finding(sim_csv_path: str, trades: pd.DataFrame) -> dict:
    """How much of the previously-reported +176,787$ / +346R improvement
    from the 'let it ride to idealTP' strategy came from the corrupted
    rows exiting at a fabricated near-entry level instead of their real
    -1.0R loss."""
    sim = pd.read_csv(sim_csv_path)
    idealtp = sim[(sim["strategy"] == "fixed_tp_idealTP") & (sim["scenario"] == "conservative")]

    full_loss_ids = trades.loc[trades["avgRiskReward"] == -1.0, "id"]
    corrupted = idealtp[idealtp["id"].isin(full_loss_ids)]

    return {
        "n_corrupted_in_sim": len(corrupted),
        "n_exited_better_than_real_-1R": int((corrupted["exit_r"] > -1.0).sum()),
        "fabricated_r_gain_total": float((corrupted["exit_r"] - (-1.0)).sum()),
    }


def corrected_tp_effect(trades: pd.DataFrame, sim_csv_path: str) -> dict:
    """The HONEST effect of the idealTP exit strategy, computed only on
    the 324 trustworthy trades."""
    sim = pd.read_csv(sim_csv_path)
    idealtp = sim[(sim["strategy"] == "fixed_tp_idealTP") & (sim["scenario"] == "conservative")]
    baseline = sim[(sim["strategy"] == "baseline") & (sim["scenario"] == "conservative")]

    clean_ids = trades.loc[trades["avgRiskReward"] != -1.0, "id"]
    r_ideal = idealtp[idealtp["id"].isin(clean_ids)].sort_values("id")["exit_r"].values
    r_base = baseline[baseline["id"].isin(clean_ids)].sort_values("id")["exit_r"].values

    return {
        "n": len(r_base),
        "baseline_win_rate": float((r_base > 0).mean()), "idealtp_win_rate": float((r_ideal > 0).mean()),
        "baseline_expectancy": stats_utils.expectancy(r_base), "idealtp_expectancy": stats_utils.expectancy(r_ideal),
        "baseline_pf": stats_utils.profit_factor(r_base), "idealtp_pf": stats_utils.profit_factor(r_ideal),
        "baseline_total_r": float(r_base.sum()), "idealtp_total_r": float(r_ideal.sum()),
    }


def precise_intervention_segments(trades: pd.DataFrame, candles: pd.DataFrame, sim_csv_path: str) -> dict:
    """Refined version (v2): the first correction used avgRiskReward !=
    -1.0 as the 'trustworthy' cutoff, which still included the 93 exact-
    breakeven trades -- maxTP turns out to be null for those too (same
    corruption signature as the full-stop-loss rows), so they were
    incorrectly counted as trustworthy. This splits the 789 trades into
    four groups by observable evidence of manual intervention:

      A. avgRiskReward == -1.0 (465): hit the ORIGINAL initalSL exactly --
         mechanical, untouched. idealTP untrustworthy (maxTP null).
      B. avgRiskReward == 0.0 exactly (93): breakeven-exact is a strong
         signature of a manually-moved stop -- untouched trades essentially
         never close at exactly zero by chance. idealTP untrustworthy here
         too (maxTP also null) so it can't be used, but excursion.py's own
         MFE (computed independently of idealTP, straight from candles) can:
         it shows what these trades were worth at their peak before being
         given back to exactly zero.
      C1. Remaining trades that closed at >=90% of their OWN idealTP-implied
          target (16): already ran close to the real target, little room.
      C2. Remaining trades that closed at <90% of their target (197):
          genuine early-close signal on trustworthy idealTP data -- this is
          the only group where "let it ride to idealTP" is a fair test.
    """
    entry, sl, tp, side = trades["entryPrice"], trades["initalSL"], trades["idealTP"], trades["side"]
    risk = (entry - sl).abs()
    reward = np.where(side == "buy", tp - entry, entry - tp)
    t = trades.copy()
    t["computed_rr"] = reward / risk
    t["maxtp_null"] = t["maxTP"].isnull()

    group_a = t[t["avgRiskReward"] == -1.0]
    group_b = t[t["avgRiskReward"] == 0.0]
    other = t[(t["avgRiskReward"] != -1.0) & (t["avgRiskReward"] != 0.0)].copy()
    other["pct_of_target"] = other["avgRiskReward"] / other["computed_rr"]
    other_clean = other[~other["maxtp_null"]]
    group_c1 = other_clean[other_clean["pct_of_target"] >= 0.90]
    group_c2 = other_clean[other_clean["pct_of_target"] < 0.90]

    from src import excursion
    exc = excursion.reconstruct_excursions(trades, candles)
    mfe_b = exc[exc["id"].isin(group_b["id"])]["mfe_r"]

    sim = pd.read_csv(sim_csv_path)
    idealtp_sim = sim[(sim["strategy"] == "fixed_tp_idealTP") & (sim["scenario"] == "conservative")]
    c2_sim = idealtp_sim[idealtp_sim["id"].isin(group_c2["id"])].sort_values("id")
    c2_actual = group_c2.sort_values("id")["avgRiskReward"]

    return {
        "n_group_a_mechanical_full_sl": len(group_a),
        "n_group_b_breakeven_manual": len(group_b),
        "n_group_c1_ran_near_target": len(group_c1),
        "n_group_c2_genuine_early_close": len(group_c2),
        "group_b_mean_mfe_r_given_back_to_zero": float(mfe_b.mean()),
        "group_b_total_mfe_r_given_back": float(mfe_b.sum()),
        "group_b_pct_with_mfe_above_1R": float((mfe_b >= 1.0).mean()),
        "group_c2_actual_total_r": float(c2_actual.sum()),
        "group_c2_if_ran_to_real_target_total_r": float(c2_sim["exit_r"].sum()),
        "group_c2_genuine_r_left_on_table": float(c2_sim["exit_r"].sum() - c2_actual.sum()),
    }


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()

    print("=== corruption summary ===")
    for k, v in summarize_corruption(trades).items():
        print(f"  {k}: {v}")

    print("\n=== impact on previously-reported hidden-pattern finding ===")
    for k, v in quantify_impact_on_hidden_pattern_finding("outputs/data/hidden_pattern_exit_simulation.csv", trades).items():
        print(f"  {k}: {v}")

    print("\n=== precise intervention segments (v2, most accurate) ===")
    for k, v in precise_intervention_segments(trades, candles, "outputs/data/hidden_pattern_exit_simulation.csv").items():
        print(f"  {k}: {v}")
