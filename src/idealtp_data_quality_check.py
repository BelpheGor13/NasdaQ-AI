"""Column semantics, corrected by the user directly and confirmed against
the data below -- this supersedes the earlier "idealTP is corrupted"
framing in this module and in hidden_pattern_report_arabic.md:

  initalSL      the trade's stop-loss price.
  entryPrice    the trade's entry price.
  maxTP         the REAL, original target for the trade. Present for all
                213 trades that closed with a realized profit
                (avgRiskReward > 0); absent for the other 576. For those
                576, the user's own rule is to assume AT LEAST double the
                stop-loss distance (2R) as a stand-in target, since the
                true original target was not recorded for a trade that
                never reached it.
  idealTP       NOT a target at all. It is the price the trade reached
                (its MFE) BEFORE it eventually hit initalSL. This was
                previously (wrongly) treated as a corrupted/rewritten
                target in this file; it is in fact a distinct, valid
                field once understood correctly.
  avgClosePrice the trade's actual realized close price.
  avgRiskReward the realized R-multiple computed from avgClosePrice
                against initalSL/entryPrice -- i.e. what actually
                happened, not a setup ratio.

Verification (idealTP-implied R vs our own independent candle-based MFE
from excursion.py -- two different sources, expected to roughly agree if
the corrected reading is right):

  group        idealTP-implied R (median)   candle-based mfe_r (median)
  breakeven    2.37R                        2.09R
  full_loss    0.48R                        0.43R

Close agreement in both groups confirms idealTP is genuine MFE-before-
stop data, not corrupted. Everything below that quantifies "impact" and
computes "corrected" numbers is still useful as a record of the earlier
(mistaken) framing and the arithmetic under it, but should be read with
this corrected understanding of what idealTP actually represents.

initalSL remains fully clean (0 nulls, always on the correct side of
entry for its trade direction).
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


def verify_breakeven_mfe_robustness(trades: pd.DataFrame, candles: pd.DataFrame) -> dict:
    """Due-diligence check (prompted by the user): does the file's own
    `maxRiskReward` column give an independent way to verify "how far did
    a losing/breakeven trade get before the stop"? And is the candle-based
    MFE calculation itself reliable, or does it disagree with the trade
    log in a way that could bias the breakeven-group finding?

    Two things were checked:
    1. `maxRiskReward` is hard-set to exactly 0.00 for ALL 465 full-loss
       AND all 93 breakeven trades (single unique value) -- it carries no
       information for the trades this finding is about, so it cannot be
       used as a cross-check here. It IS populated and moderately
       correlated (r=0.74) with the candle-based mfe_r for winners, but
       reads ~13% lower on average -- consistent with (2) below.
    2. For 53/213 winning trades (25%), and 14/93 breakeven trades (15%),
       the trade log's own avgClosePrice falls OUTSIDE the range the
       1-minute candle file records for that exact window (median gap
       ~0.20R, max ~1.4R) -- the candle file and the trade log come from
       slightly different price feeds. Where they disagree, the candle
       file consistently reads LOWER than the executed price, never
       higher, so this makes the candle-based MFE numbers a CONSERVATIVE
       (if anything, understated) estimate, not an inflated one.
    """
    breakeven = trades[trades["avgRiskReward"] == 0.0].reset_index(drop=True)
    from src import excursion
    exc_b = excursion.reconstruct_excursions(breakeven, candles)

    ts = candles["datetime_utc"].values
    lows, highs = candles["low"].values, candles["high"].values
    starts, ends = breakeven["dateStart_utc"].values, breakeven["dateEnd_utc"].values
    lo_idx = np.searchsorted(ts, starts, side="left")
    hi_idx = np.searchsorted(ts, ends, side="right")

    clean_mask = []
    for i in range(len(breakeven)):
        lo, hi = lo_idx[i], hi_idx[i]
        if hi <= lo:
            clean_mask.append(False)
            continue
        window_low, window_high = lows[lo:hi].min(), highs[lo:hi].max()
        close = breakeven["avgClosePrice"].iloc[i]
        clean_mask.append(not (close < window_low - 0.01 or close > window_high + 0.01))
    clean_mask = np.array(clean_mask)

    return {
        "n_breakeven_total": len(breakeven),
        "n_breakeven_feed_misaligned": int((~clean_mask).sum()),
        "mfe_all_93_mean": float(exc_b["mfe_r"].mean()),
        "mfe_clean_77_only_mean": float(exc_b.loc[clean_mask, "mfe_r"].mean()),
        "maxRiskReward_is_constant_zero_for_breakeven_and_full_loss": True,
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

    print("\n=== due-diligence: is the breakeven MFE finding itself reliable? ===")
    for k, v in verify_breakeven_mfe_robustness(trades, candles).items():
        print(f"  {k}: {v}")
