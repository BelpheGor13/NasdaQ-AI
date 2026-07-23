"""Runs the macro/cross-asset features through the exact same robustness
stack as shap_hypothesis_test.py: a small, pre-registered feature family
(11 macro features -> up to 66 single/pairwise conditions), Bonferroni-
corrected within this family (not the ~900-combination main search's FDR),
then walk-forward + Monte Carlo + Bayesian evidence + stability/regime
checks on every condition that clears the initial bar.
"""
from src import (
    pattern_search, walk_forward, monte_carlo, bayesian_evidence,
    stability_regime, scoring, macro_feature_engineering as mfe,
)

TARGET_COL = "r_multiple"
MACRO_FEATURES = mfe.MACRO_CATEGORICAL_FEATURES + mfe.MACRO_CONTINUOUS_FEATURES


def run_macro_pattern_search(feats):
    candidates = pattern_search.search_patterns(
        feats, target_col=TARGET_COL, min_n=20, max_condition_depth=2, restrict_cols=MACRO_FEATURES,
        extra_categorical_features=mfe.MACRO_CATEGORICAL_FEATURES,
        extra_continuous_features=mfe.MACRO_CONTINUOUS_FEATURES,
    )
    if len(candidates) == 0:
        return candidates, {}

    wf, fold_details = walk_forward.run_walk_forward_for_candidates(
        feats, candidates, target_col=TARGET_COL, top_n=len(candidates),
        extra_continuous_features=mfe.MACRO_CONTINUOUS_FEATURES)
    mc = monte_carlo.run_mc_for_candidates(feats, candidates, target_col=TARGET_COL,
                                            extra_continuous_features=mfe.MACRO_CONTINUOUS_FEATURES)
    bay = bayesian_evidence.add_bayesian_evidence(feats, candidates, target_col=TARGET_COL,
                                                    extra_continuous_features=mfe.MACRO_CONTINUOUS_FEATURES)
    stab = stability_regime.run_stability_regime_for_candidates(
        feats, candidates, target_col=TARGET_COL, extra_continuous_features=mfe.MACRO_CONTINUOUS_FEATURES)

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

    # Bonferroni over this small pre-registered family, same convention as shap_hypothesis_test.py
    merged["reject_fdr"] = merged["p_bonferroni"] < 0.05
    scored = scoring.compute_robustness_score(merged)
    return scored, fold_details


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)
    macro_daily = mfe.build_macro_features(candles)
    feats = mfe.attach_macro_to_trades(feats, macro_daily)

    scored, fold_details = run_macro_pattern_search(feats)
    print(f"{len(scored)} macro conditions tested (Bonferroni alpha=0.05)")
    print(f"survivors: {int(scored['reject_fdr'].sum())}")
    cols = ["condition", "n", "expectancy", "p_value_expectancy", "p_bonferroni", "mc_verdict", "confidence_tier"]
    print(scored[cols].head(15).to_string(index=False))
