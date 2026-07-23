"""Follow-up to shap_hypothesis_test.py: that test found the SHAP-stable
features (sl_pct_of_atr, pre_entry_momentum_15m/30m, regime_vol) don't
work as standalone or paired-among-themselves rules, and speculated their
SHAP importance might only show up through interaction with context like
day-of-week. This tests that directly: same 4 SHAP features, now paired
with day_of_week AND entry_hour_utc (a real pre-registered family, still
Bonferroni-corrected within itself -- not folded into the main search).

session was in the original request too, but is dropped here and
disclosed rather than silently included: 777/789 trades (98.5%) fall in
one single session bucket (see final_report_arabic.md's "unprompted
findings"), so it has almost no variance to interact with anything --
entry_hour_utc is the feature that actually carries the time-of-day
signal in this dataset.
"""
from src import (
    pattern_search, walk_forward, monte_carlo, bayesian_evidence,
    stability_regime, scoring, shap_hypothesis_test as sht,
)

TARGET_COL = "r_multiple"


def run_interaction_test(feats, shap_features: list = None):
    shap_features = shap_features or sht.SHAP_STABLE_FEATURES
    feature_list = list(shap_features) + ["day_of_week", "entry_hour_utc"]

    candidates = pattern_search.search_patterns(
        feats, target_col=TARGET_COL, min_n=20, max_condition_depth=2, restrict_cols=feature_list
    )
    if len(candidates) == 0:
        return candidates, {}

    wf, fold_details = walk_forward.run_walk_forward_for_candidates(
        feats, candidates, target_col=TARGET_COL, top_n=len(candidates))
    mc = monte_carlo.run_mc_for_candidates(feats, candidates, target_col=TARGET_COL)
    bay = bayesian_evidence.add_bayesian_evidence(feats, candidates, target_col=TARGET_COL)
    stab = stability_regime.run_stability_regime_for_candidates(feats, candidates, target_col=TARGET_COL)

    merged = candidates.copy()
    for extra, cols in [
        (wf, ["n_folds_with_data", "n_folds_positive", "expectancy_sign_consistent", "expectancy_std_across_folds"]),
        (mc, ["prob_sign_flip", "perm_p_expectancy", "mc_verdict"]),
        (bay, ["posterior_win_rate_mean", "posterior_win_rate_ci_lo", "posterior_win_rate_ci_hi",
               "bayes_factor_vs_base_rate", "bayes_evidence"]),
        (stab, ["stability_applicable", "stability_pass_fraction", "stability_verdict",
                "regime_dependent_on", "regime_verdict"]),
    ]:
        for c in cols:
            merged[c] = extra[c].values

    merged["reject_fdr"] = merged["p_bonferroni"] < 0.05  # Bonferroni within this small family
    scored = scoring.compute_robustness_score(merged)
    return scored, fold_details


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    scored, fold_details = run_interaction_test(feats)
    print(f"{len(scored)} conditions tested (SHAP features x day_of_week x entry_hour_utc)")
    print(f"survivors (Bonferroni alpha=0.05): {int(scored['reject_fdr'].sum())}")
    cols = ["condition", "n", "expectancy", "p_value_expectancy", "p_bonferroni", "mc_verdict", "confidence_tier"]
    print(scored[cols].head(15).to_string(index=False))
