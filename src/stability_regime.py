"""Stage 10: two more robustness gates for surviving candidates.

1. Threshold-perturbation stability -- if a candidate's condition includes
   a binned continuous feature, redefine that one feature's cut points a
   few reasonable different ways (quartiles instead of tertiles, tertile
   edges nudged +/-5 percentile points) and check whether the pattern's
   direction and approximate significance survive. A categorical-only
   condition (e.g. day_of_week, side) has no threshold to perturb -- those
   are reported as not applicable rather than silently marked stable.

2. Regime-conditioned validation -- does the edge concentrate in one
   market regime (Trend/Range, High/Low Vol, Expansion/Compression) or
   hold across all of them? Changes how a surviving pattern should be used,
   not just whether it's "valid".
"""
import numpy as np
import pandas as pd

from src import pattern_search, stats_utils

CONTINUOUS_BASE_NAMES = [c[:-4] for c in pattern_search.CONTINUOUS_FEATURES]  # strip nothing; kept for clarity
REGIME_COLS = ["regime_trend_asof_prior_day", "regime_vol_asof_prior_day", "regime_expansion_asof_prior_day"]


def _perturbed_bin_masks(df: pd.DataFrame, col: str, bin_label: str, n_bins_variants=(3, 4),
                          edge_shifts=(-0.05, 0.0, 0.05)) -> list:
    """Rebuilds alternative masks approximating the same relative position
    (e.g. 'bottom third') under different bin definitions for the underlying
    continuous column `col`, given the originally selected bin's label.
    """
    masks = []
    values = df[col]
    sorted_vals = values.dropna().sort_values().values
    n = len(sorted_vals)
    # Determine which tertile (0=low,1=mid,2=high...) the original bin represents
    # by matching its interval string against qcut categories in order.
    cats_sorted = sorted(pd.qcut(values, pattern_search.N_QUANTILE_BINS, duplicates="drop").cat.categories,
                          key=lambda iv: iv.left)
    cat_strs = [str(c) for c in cats_sorted]
    if bin_label not in cat_strs:
        return masks
    position = cat_strs.index(bin_label)
    n_orig_bins = len(cat_strs)

    for n_bins in n_bins_variants:
        for shift in edge_shifts:
            q_lo = position / n_orig_bins + shift
            q_hi = (position + 1) / n_orig_bins + shift
            q_lo, q_hi = max(0.0, min(q_lo, 1.0)), max(0.0, min(q_hi, 1.0))
            if q_hi <= q_lo:
                continue
            lo_val, hi_val = np.quantile(sorted_vals, [q_lo, q_hi])
            mask = (values >= lo_val) & (values <= hi_val)
            masks.append(mask)
    return masks


def threshold_perturbation_stability(df: pd.DataFrame, condition_dict: dict, target_col: str) -> dict:
    continuous_conditions = {c: v for c, v in condition_dict.items() if c.endswith("_bin")}
    categorical_conditions = {c: v for c, v in condition_dict.items() if not c.endswith("_bin")}

    if not continuous_conditions:
        return {"stability_applicable": False, "stability_pass_fraction": np.nan, "stability_verdict": "n/a (categorical-only condition)"}

    base_mask = pd.Series(True, index=df.index)
    for c, v in categorical_conditions.items():
        base_mask &= (df[c] == v)

    original_expectancy_sign = None
    n_variants, n_consistent = 0, 0

    for col_bin, bin_label in continuous_conditions.items():
        base_col = col_bin[:-4]
        variant_masks = _perturbed_bin_masks(df, base_col, bin_label)
        other_conditions_mask = base_mask.copy()
        for other_col_bin, other_label in continuous_conditions.items():
            if other_col_bin == col_bin:
                continue
            # keep other continuous conditions fixed at their original bin
            other_conditions_mask &= (df[other_col_bin] == other_label)

        exact_mask = other_conditions_mask & (df[col_bin] == bin_label)
        exact_vals = df.loc[exact_mask, target_col].dropna().values
        if len(exact_vals) == 0:
            continue
        original_expectancy_sign = np.sign(exact_vals.mean())

        for vmask in variant_masks:
            full_mask = other_conditions_mask & vmask
            vals = df.loc[full_mask, target_col].dropna().values
            if len(vals) < 10:
                continue
            n_variants += 1
            if np.sign(vals.mean()) == original_expectancy_sign:
                n_consistent += 1

    if n_variants == 0:
        return {"stability_applicable": True, "stability_pass_fraction": np.nan, "stability_verdict": "insufficient_data"}

    frac = n_consistent / n_variants
    verdict = "stable" if frac >= 0.7 else ("mixed" if frac >= 0.4 else "unstable")
    return {"stability_applicable": True, "stability_pass_fraction": frac, "stability_verdict": verdict}


def regime_conditioned_breakdown(df: pd.DataFrame, condition_dict: dict, target_col: str) -> dict:
    mask = pattern_search.apply_condition(df, condition_dict)
    matched = df.loc[mask]

    breakdown = {}
    concentration_flags = []
    for regime_col in REGIME_COLS:
        if regime_col not in matched.columns:
            continue
        sub_stats = {}
        for regime_val, group in matched.groupby(regime_col, observed=True, dropna=True):
            vals = group[target_col].dropna().values
            if len(vals) == 0:
                continue
            sub_stats[regime_val] = {"n": len(vals), "expectancy": stats_utils.expectancy(vals)}
        breakdown[regime_col] = sub_stats

        signs = {k: np.sign(v["expectancy"]) for k, v in sub_stats.items() if v["n"] >= 10}
        if len(signs) >= 2 and len(set(signs.values())) > 1:
            concentration_flags.append(regime_col)

    verdict = "regime_dependent" if concentration_flags else "holds_across_regimes_tested"
    return {"regime_breakdown": breakdown, "regime_dependent_on": concentration_flags, "regime_verdict": verdict}


def run_stability_regime_for_candidates(df: pd.DataFrame, candidates: pd.DataFrame, target_col: str) -> pd.DataFrame:
    disc = pattern_search.discretize(df)
    rows = []
    for _, row in candidates.iterrows():
        stab = threshold_perturbation_stability(disc, row["condition_dict"], target_col)
        reg = regime_conditioned_breakdown(disc, row["condition_dict"], target_col)
        rows.append({**stab, "regime_dependent_on": reg["regime_dependent_on"], "regime_verdict": reg["regime_verdict"]})
    return pd.concat([candidates.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    candidates = pattern_search.search_patterns(feats, target_col="r_multiple", min_n=20, max_condition_depth=2)
    result = run_stability_regime_for_candidates(feats, candidates.head(10), target_col="r_multiple")

    cols = ["condition", "n", "expectancy", "stability_applicable", "stability_pass_fraction",
            "stability_verdict", "regime_dependent_on", "regime_verdict"]
    print(result[cols].to_string(index=False))
