"""Stage 6: search the composite feature space for conditions associated
with win/loss and with high/low maxRiskReward potential.

Design note on bin edges: continuous features are discretized using
quantile bins fit ONCE on the full sample. This is a hypothesis-generation
step, not the validation step -- walk_forward.py re-tests every surviving
candidate out-of-sample using these same fixed bin definitions. That is a
deliberate simplification (re-fitting bin edges per fold would add its own
instability on a ~789-trade sample) and is disclosed here rather than
silently assumed; it means a pattern's bin boundaries are treated as
structural, not as something fit to any particular test period.
"""
from itertools import combinations

import numpy as np
import pandas as pd

from src import stats_utils

CATEGORICAL_FEATURES = [
    "side", "day_of_week", "session", "entry_hour_utc",
    "regime_trend_asof_prior_day", "regime_vol_asof_prior_day", "regime_expansion_asof_prior_day",
]

CONTINUOUS_FEATURES = [
    "sl_pct_of_atr", "pct_dist_from_high_20d", "pct_dist_from_low_20d",
    "pct_dist_from_high_50d", "pct_dist_from_low_50d",
    "pre_entry_momentum_5m", "pre_entry_momentum_15m", "pre_entry_momentum_30m",
    "pre_entry_pct_up_candles_15m",
]

N_QUANTILE_BINS = 3  # tertiles: keeps per-cell sample sizes viable at n=789


def discretize(df: pd.DataFrame, continuous_features: list = None) -> pd.DataFrame:
    """continuous_features overrides the module default so external
    callers (e.g. macro_pattern_search.py) can bin their own feature set
    without it needing to already be in CONTINUOUS_FEATURES -- that list
    stays scoped to the core NAS100-internal features it was written for.
    """
    continuous_features = continuous_features if continuous_features is not None else CONTINUOUS_FEATURES
    out = df.copy()
    for col in continuous_features:
        if col not in out.columns:
            continue
        try:
            binned = pd.qcut(out[col], N_QUANTILE_BINS, duplicates="drop")
            # Stringify only non-null entries -- astype(str) on the whole
            # column would turn real NaNs (missing warmup data etc.) into
            # the literal string "nan", which then survives as a bogus
            # "pattern" category instead of being dropped as missing data.
            out[col + "_bin"] = binned.astype(object).where(binned.notna(), np.nan)
            out.loc[binned.notna(), col + "_bin"] = binned[binned.notna()].astype(str)
        except ValueError:
            out[col + "_bin"] = np.nan
    return out


def _condition_columns(df: pd.DataFrame, categorical_features: list = None,
                        continuous_features: list = None) -> list:
    categorical_features = categorical_features if categorical_features is not None else CATEGORICAL_FEATURES
    continuous_features = continuous_features if continuous_features is not None else CONTINUOUS_FEATURES
    cols = [c for c in categorical_features if c in df.columns]
    cols += [c + "_bin" for c in continuous_features if c + "_bin" in df.columns]
    return cols


def _evaluate_mask(df: pd.DataFrame, mask: pd.Series, base_win_rate: float,
                    base_r_values: np.ndarray, target_col: str, min_n: int) -> dict:
    n = int(mask.sum())
    if n < min_n:
        return None

    subset = df.loc[mask, target_col].dropna().values
    wins = int((subset > 0).sum())
    win_rate = wins / len(subset) if len(subset) else np.nan

    p_win = stats_utils.two_proportion_ztest(wins, len(subset), int((base_r_values > 0).sum()), len(base_r_values))
    p_r = stats_utils.welch_ttest_pvalue(subset, base_r_values)

    exp_lo, exp_hi = stats_utils.bootstrap_ci(subset, np.mean)
    wr_lo, wr_hi = stats_utils.bootstrap_ci((subset > 0).astype(float), np.mean)

    return {
        "n": n,
        "win_rate": win_rate,
        "win_rate_delta": win_rate - base_win_rate,
        "win_rate_ci_lo": wr_lo,
        "win_rate_ci_hi": wr_hi,
        "expectancy": stats_utils.expectancy(subset),
        "expectancy_delta": stats_utils.expectancy(subset) - stats_utils.expectancy(base_r_values),
        "expectancy_ci_lo": exp_lo,
        "expectancy_ci_hi": exp_hi,
        "profit_factor": stats_utils.profit_factor(subset),
        "p_value_winrate": p_win,
        "p_value_expectancy": p_r,
    }


def apply_condition(disc_df: pd.DataFrame, condition_dict: dict) -> pd.Series:
    """Rebuilds the boolean mask for a candidate pattern's condition_dict
    against an already-discretized dataframe (see discretize()). Used by
    walk_forward.py and monte_carlo.py to re-evaluate the exact same rule
    on different slices of data.
    """
    mask = pd.Series(True, index=disc_df.index)
    for col, val in condition_dict.items():
        mask &= disc_df[col] == val
    return mask


def search_patterns(df: pd.DataFrame, target_col: str = "r_multiple", min_n: int = 20,
                     max_condition_depth: int = 2, restrict_cols: list = None,
                     extra_categorical_features: list = None, extra_continuous_features: list = None) -> pd.DataFrame:
    """restrict_cols limits the condition space to a specific subset of
    features (raw names, without the "_bin" suffix for continuous ones).
    Used for pre-registered, hypothesis-driven confirmatory tests (e.g. a
    small SHAP-flagged feature set) where correcting across a small family
    of tests is the honest comparison -- not the full ~900-combination
    exploratory search, which is a different, much larger family.

    extra_{categorical,continuous}_features let a caller search a feature
    set OUTSIDE this module's own CATEGORICAL_FEATURES/CONTINUOUS_FEATURES
    (e.g. macro_pattern_search.py's external market features) without
    polluting those core-pipeline lists -- they're unioned in just for
    this call's discretize()/_condition_columns() pass.
    """
    all_categorical = CATEGORICAL_FEATURES + (extra_categorical_features or [])
    all_continuous = CONTINUOUS_FEATURES + (extra_continuous_features or [])

    disc = discretize(df, continuous_features=all_continuous)
    cond_cols = _condition_columns(disc, categorical_features=all_categorical, continuous_features=all_continuous)
    if restrict_cols is not None:
        wanted = set(restrict_cols)
        cond_cols = [c for c in cond_cols if c in wanted or c.replace("_bin", "") in wanted]

    base_r_values = disc[target_col].dropna().values
    base_win_rate = float((base_r_values > 0).mean())

    results = []
    for depth in range(1, max_condition_depth + 1):
        for combo in combinations(cond_cols, depth):
            combo_cols = list(combo)
            clean = disc.dropna(subset=combo_cols)
            grouped = clean.groupby(combo_cols, observed=True, dropna=True)
            for keys, group_idx in grouped.groups.items():
                mask = disc.index.isin(group_idx)
                stats_row = _evaluate_mask(disc, pd.Series(mask, index=disc.index), base_win_rate,
                                            base_r_values, target_col, min_n)
                if stats_row is None:
                    continue
                keys_tuple = keys if isinstance(keys, tuple) else (keys,)
                condition_dict = dict(zip(combo, keys_tuple))
                stats_row["condition"] = " & ".join(f"{c}={k}" for c, k in condition_dict.items())
                stats_row["condition_dict"] = condition_dict
                stats_row["depth"] = depth
                results.append(stats_row)

    result_df = pd.DataFrame(results)
    if len(result_df) == 0:
        return result_df

    result_df = result_df.sort_values("p_value_expectancy").reset_index(drop=True)

    fdr = stats_utils.benjamini_hochberg(result_df["p_value_expectancy"])
    result_df["p_fdr_bh"] = fdr["p_fdr_bh"].values
    result_df["reject_fdr"] = fdr["reject_fdr"].values
    result_df["p_bonferroni"] = fdr["p_bonferroni"].values

    result_df["low_confidence_sample"] = result_df["n"] < 25

    return result_df


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    results = search_patterns(feats, target_col="r_multiple", min_n=20, max_condition_depth=2)
    print(f"total candidate patterns tested: {len(results)}")
    print(f"surviving FDR correction (BH, alpha=0.05): {results['reject_fdr'].sum()}")
    print()
    cols = ["condition", "n", "win_rate", "expectancy", "profit_factor", "p_value_expectancy", "p_fdr_bh"]
    print(results[cols].head(20).to_string(index=False))
