"""Stage 7: walk-forward robustness check for candidate patterns.

Interpretation used here: candidate patterns are rule-based (a fixed
composite condition on discretized features), not fitted ML models, and
their discretization bins are already fixed globally (see pattern_search.py
docstring). So "walk-forward" for a rule means checking whether the SAME
fixed rule keeps its edge when evaluated separately within each subsequent
out-of-sample calendar-year window, rather than re-fitting anything per
fold. A rule that only "works" in one year and inverts in the next is
flagged as unstable regardless of its full-sample p-value.
"""
import numpy as np
import pandas as pd

from src import config, pattern_search, stats_utils


def _fold_label(row_year: int, folds) -> str:
    for f in folds:
        test_year = pd.Timestamp(f["test_start"]).year
        if row_year == test_year:
            return f"{test_year}"
    return None


def walk_forward_breakdown(disc_df: pd.DataFrame, condition_dict: dict, target_col: str,
                            folds=None, min_n: int = config.MIN_SAMPLE_SIZE) -> pd.DataFrame:
    folds = folds or config.WALK_FORWARD_FOLDS
    mask = pattern_search.apply_condition(disc_df, condition_dict)
    matched = disc_df.loc[mask].copy()
    matched["fold_year"] = matched["dateStart_utc"].dt.year

    rows = []
    for f in folds:
        test_year = pd.Timestamp(f["test_start"]).year
        fold_data = matched.loc[matched["fold_year"] == test_year, target_col].dropna().values
        n = len(fold_data)
        wins = int((fold_data > 0).sum())
        rows.append({
            "fold": test_year,
            "n": n,
            "low_sample": n < min_n,
            "win_rate": wins / n if n else np.nan,
            "expectancy": stats_utils.expectancy(fold_data) if n else np.nan,
            "profit_factor": stats_utils.profit_factor(fold_data) if n else np.nan,
        })
    return pd.DataFrame(rows)


def summarize_stability(fold_df: pd.DataFrame) -> dict:
    valid = fold_df.dropna(subset=["expectancy"])
    if len(valid) == 0:
        return {"n_folds_with_data": 0, "n_folds_positive": 0, "expectancy_sign_consistent": False,
                "expectancy_std_across_folds": np.nan}
    n_pos = int((valid["expectancy"] > 0).sum())
    return {
        "n_folds_with_data": len(valid),
        "n_folds_positive": n_pos,
        "expectancy_sign_consistent": n_pos == len(valid) or n_pos == 0,
        "expectancy_std_across_folds": float(valid["expectancy"].std()),
    }


def run_walk_forward_for_candidates(df: pd.DataFrame, candidates: pd.DataFrame, target_col: str,
                                     top_n: int = 30, extra_continuous_features: list = None) -> pd.DataFrame:
    """Runs the per-fold breakdown for the top_n candidates (by raw
    p-value) and attaches a stability summary to each. extra_continuous_features
    must match whatever was passed to pattern_search.search_patterns() when
    these candidates were generated, so the same _bin columns exist here.
    """
    disc = pattern_search.discretize(df, continuous_features=(
        pattern_search.CONTINUOUS_FEATURES + (extra_continuous_features or [])))
    top = candidates.head(top_n).copy()

    stability_rows = []
    all_folds = {}
    for i, row in top.iterrows():
        fold_df = walk_forward_breakdown(disc, row["condition_dict"], target_col)
        all_folds[row["condition"]] = fold_df
        stability_rows.append(summarize_stability(fold_df))

    stability_df = pd.DataFrame(stability_rows, index=top.index)
    out = pd.concat([top, stability_df], axis=1)
    return out, all_folds


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    candidates = pattern_search.search_patterns(feats, target_col="r_multiple", min_n=20, max_condition_depth=2)
    result, fold_details = run_walk_forward_for_candidates(feats, candidates, target_col="r_multiple", top_n=10)

    for i, row in result.iterrows():
        print(f"\n{row['condition']}  (n={row['n']}, full-sample expectancy={row['expectancy']:.3f}, "
              f"p_raw={row['p_value_expectancy']:.4f})")
        print(f"  stable sign across folds: {row['expectancy_sign_consistent']}  "
              f"folds w/ data: {row['n_folds_with_data']}  positive folds: {row['n_folds_positive']}")
        print(fold_details[row["condition"]].to_string(index=False))
