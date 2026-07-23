"""Stage 17 (follow-up to ml_shap.py): promotes the SHAP walk-forward-
stable features from "interpretability signal" to a directly-tested,
PRE-REGISTERED set of composite rules -- run through the exact same
walk-forward + Monte Carlo + Bayesian + stability gates as the main
exploratory search, per the report's next-step recommendation.

Why a separate module instead of just reading off pattern_search's main
results: this is a genuinely different statistical exercise. The main
search tests ~900+ combinations and needs FDR correction sized to that
family. Here we pre-select a small number of features because SHAP
flagged them as consistently important across every walk-forward fold,
then test only that small family (Bonferroni over ~10 conditions, not
~900) -- a fair, honest comparison for a hypothesis that was NOT mined
from this same search.
"""
import pandas as pd

from src import (
    config, pattern_search, walk_forward, monte_carlo, bayesian_evidence,
    stability_regime, scoring, ml_shap,
)

TARGET_COL = "r_multiple"
SHAP_STABLE_FEATURES = ["sl_pct_of_atr", "pre_entry_momentum_15m", "pre_entry_momentum_30m",
                         "regime_vol_asof_prior_day"]


def select_shap_stable_features(feats: pd.DataFrame, top_k: int = 4) -> list:
    """Re-derives the feature list from ml_shap's own walk-forward ranking
    rather than hardcoding it, so this stays correct if the underlying
    data changes."""
    imp_df = ml_shap.run_walk_forward_shap(feats, target_col=TARGET_COL)
    return list(imp_df.sort_values("mean_rank").head(top_k).index)


def run_shap_guided_test(feats: pd.DataFrame, feature_list: list = None) -> pd.DataFrame:
    feature_list = feature_list or SHAP_STABLE_FEATURES

    candidates = pattern_search.search_patterns(
        feats, target_col=TARGET_COL, min_n=20, max_condition_depth=2, restrict_cols=feature_list
    )
    if len(candidates) == 0:
        return candidates

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

    # Bonferroni over THIS small family, not the full-search FDR column --
    # reject using p_bonferroni (already computed relative to len(candidates)).
    merged["reject_fdr"] = merged["p_bonferroni"] < 0.05
    scored = scoring.compute_robustness_score(merged)
    return scored, fold_details


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    features = select_shap_stable_features(feats)
    print(f"SHAP-stable feature family (top 4 by walk-forward mean rank): {features}")

    scored, fold_details = run_shap_guided_test(feats, features)
    print(f"\n{len(scored)} conditions tested within this pre-registered family "
          f"(Bonferroni alpha=0.05)")
    cols = ["condition", "n", "expectancy", "p_value_expectancy", "p_bonferroni", "reject_fdr",
            "mc_verdict", "confidence_tier", "robustness_score"]
    print(scored[cols].to_string(index=False))
