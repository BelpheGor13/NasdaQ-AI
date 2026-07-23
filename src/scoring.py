"""Stage 14: combine every robustness gate a candidate has already passed
through (pattern_search, walk_forward, monte_carlo, bayesian_evidence,
stability_regime) into a single 0-100 Robustness Score, so ranking is by
robustness rather than raw historical profit.

Weights (documented, not arbitrary -- but still a judgment call disclosed
here rather than hidden in a formula):
  25% statistical significance (FDR-corrected p-value)
  20% walk-forward sign-consistency across folds
  15% Monte Carlo survival (1 - probability the bootstrap flips direction)
  15% effect size (|expectancy| relative to other candidates)
  15% threshold-perturbation stability (n/a conditions score neutral 0.5)
  10% sample size adequacy (saturates at 4x the minimum sample threshold)

All components are direction-aware: a candidate is scored on how reliably
it repeats ITS OWN observed direction (positive or negative), not on
whether it is profitable -- a robust "avoid this" filter scores as highly
as a robust "trade this" filter. Direction is reported separately so the
two are never confused in the final ranking.
"""
import numpy as np
import pandas as pd

from src import config


WEIGHTS = {
    "significance": 0.25,
    "walk_forward": 0.20,
    "monte_carlo": 0.15,
    "effect_size": 0.15,
    "stability": 0.15,
    "sample_size": 0.10,
}


def _score_significance(p_fdr: float) -> float:
    if pd.isna(p_fdr):
        return 0.0
    return float(np.clip(100 * (1 - p_fdr), 0, 100))


def _score_walk_forward(row) -> float:
    n_folds = row.get("n_folds_with_data", 0)
    if not n_folds:
        return 0.0
    n_pos = row.get("n_folds_positive", 0)
    frac_matching_direction = n_pos / n_folds if row["expectancy"] > 0 else 1 - n_pos / n_folds
    return float(np.clip(frac_matching_direction * 100, 0, 100))


def _score_monte_carlo(prob_sign_flip: float) -> float:
    if pd.isna(prob_sign_flip):
        return 0.0
    return float(np.clip(100 * (1 - prob_sign_flip), 0, 100))


def _score_effect_size(abs_expectancy: pd.Series) -> pd.Series:
    if abs_expectancy.max() == abs_expectancy.min():
        return pd.Series(50.0, index=abs_expectancy.index)
    return 100 * (abs_expectancy - abs_expectancy.min()) / (abs_expectancy.max() - abs_expectancy.min())


def _score_stability(row) -> float:
    if not row.get("stability_applicable", False):
        return 50.0  # categorical-only condition: no threshold to test, neutral score
    frac = row.get("stability_pass_fraction", np.nan)
    return float(np.clip(frac * 100, 0, 100)) if pd.notna(frac) else 0.0


def _score_sample_size(n: int, saturate_at: int = config.MIN_SAMPLE_SIZE * 4) -> float:
    return float(np.clip(100 * n / saturate_at, 0, 100))


def compute_robustness_score(enriched: pd.DataFrame) -> pd.DataFrame:
    out = enriched.copy()
    out["direction"] = np.where(out["expectancy"] > 0, "positive (tradeable edge candidate)",
                                 "negative (avoid-filter candidate)")

    out["score_significance"] = out["p_fdr_bh"].apply(_score_significance)
    out["score_walk_forward"] = out.apply(_score_walk_forward, axis=1)
    out["score_monte_carlo"] = out["prob_sign_flip"].apply(_score_monte_carlo)
    out["score_effect_size"] = _score_effect_size(out["expectancy"].abs())
    out["score_stability"] = out.apply(_score_stability, axis=1)
    out["score_sample_size"] = out["n"].apply(_score_sample_size)

    out["robustness_score"] = (
        WEIGHTS["significance"] * out["score_significance"]
        + WEIGHTS["walk_forward"] * out["score_walk_forward"]
        + WEIGHTS["monte_carlo"] * out["score_monte_carlo"]
        + WEIGHTS["effect_size"] * out["score_effect_size"]
        + WEIGHTS["stability"] * out["score_stability"]
        + WEIGHTS["sample_size"] * out["score_sample_size"]
    )

    out["confidence_tier"] = np.select(
        [
            out["reject_fdr"] & (out["mc_verdict"] == "survives_monte_carlo") & (out["n"] >= config.MIN_SAMPLE_SIZE),
            (out["p_value_expectancy"] < 0.05) & (out["mc_verdict"] == "survives_monte_carlo"),
        ],
        ["confirmed", "exploratory"],
        default="rejected_or_low_confidence",
    )

    return out.sort_values("robustness_score", ascending=False)


if __name__ == "__main__":
    from src import data_loading, feature_engineering, pattern_search, walk_forward, monte_carlo, \
        bayesian_evidence, stability_regime

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    candidates = pattern_search.search_patterns(feats, target_col="r_multiple", min_n=20, max_condition_depth=2)
    top = candidates.head(30).reset_index(drop=True)

    wf, _ = walk_forward.run_walk_forward_for_candidates(feats, top, target_col="r_multiple", top_n=30)
    mc = monte_carlo.run_mc_for_candidates(feats, top, target_col="r_multiple")
    bay = bayesian_evidence.add_bayesian_evidence(feats, top, target_col="r_multiple")
    stab = stability_regime.run_stability_regime_for_candidates(feats, top, target_col="r_multiple")

    merged = top.copy()
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

    scored = compute_robustness_score(merged)
    cols = ["condition", "direction", "n", "expectancy", "p_fdr_bh", "mc_verdict", "stability_verdict",
            "regime_verdict", "confidence_tier", "robustness_score"]
    print(scored[cols].head(15).to_string(index=False))
