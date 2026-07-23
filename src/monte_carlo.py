"""Stage 8: Monte Carlo robustness testing for individual candidate
patterns (not just the overall strategy). Two complementary tests:

1. Bootstrap survival -- resample the pattern's OWN matched trades with
   replacement, and ask: in what fraction of resamples does the edge
   disappear (expectancy <= 0) or the profit factor drop below 1? This
   answers "how fragile is this specific set of outcomes."

2. Permutation test -- repeatedly draw a random same-size subset from the
   REST of the population (trades not matching the pattern) and ask how
   often a random subset would look this good by chance. This answers
   "is finding a group this good unremarkable given how much of the
   feature space was searched."
"""
import numpy as np
import pandas as pd

from src import config, pattern_search, stats_utils


def bootstrap_survival(r_values: np.ndarray, n_iter: int = config.MC_ITERATIONS,
                        seed: int = config.RANDOM_SEED) -> dict:
    if len(r_values) < 5:
        return {"prob_expectancy_negative": np.nan, "prob_pf_below_1": np.nan,
                "expectancy_5th_pct": np.nan, "expectancy_95th_pct": np.nan}

    rng = np.random.default_rng(seed)
    n = len(r_values)
    idx = rng.integers(0, n, (n_iter, n))
    samples = r_values[idx]

    exp_dist = samples.mean(axis=1)
    wins = np.where(samples > 0, samples, 0).sum(axis=1)
    losses = np.where(samples < 0, -samples, 0).sum(axis=1)
    pf_dist = np.divide(wins, losses, out=np.full(n_iter, np.inf), where=losses > 0)

    return {
        "prob_expectancy_negative": float((exp_dist <= 0).mean()),
        "prob_pf_below_1": float((pf_dist < 1).mean()),
        "expectancy_5th_pct": float(np.percentile(exp_dist, 5)),
        "expectancy_95th_pct": float(np.percentile(exp_dist, 95)),
    }


def permutation_test(base_r_values: np.ndarray, pattern_r_values: np.ndarray,
                      n_iter: int = config.MC_ITERATIONS, seed: int = config.RANDOM_SEED) -> dict:
    n = len(pattern_r_values)
    if n < 5 or len(base_r_values) <= n:
        return {"perm_p_expectancy": np.nan, "perm_p_winrate": np.nan}

    observed_exp = pattern_r_values.mean()
    observed_wr = (pattern_r_values > 0).mean()

    rng = np.random.default_rng(seed)
    n_pop = len(base_r_values)
    idx = np.array([rng.choice(n_pop, size=n, replace=False) for _ in range(n_iter)])
    samples = base_r_values[idx]

    perm_exp = samples.mean(axis=1)
    perm_wr = (samples > 0).mean(axis=1)

    # two-sided: how often does a random group deviate from the population
    # mean at least as much as the observed group does
    pop_mean = base_r_values.mean()
    p_exp = float((np.abs(perm_exp - pop_mean) >= np.abs(observed_exp - pop_mean)).mean())
    pop_wr = (base_r_values > 0).mean()
    p_wr = float((np.abs(perm_wr - pop_wr) >= np.abs(observed_wr - pop_wr)).mean())

    return {"perm_p_expectancy": p_exp, "perm_p_winrate": p_wr}


def evaluate_pattern_mc(disc_df: pd.DataFrame, condition_dict: dict, target_col: str) -> dict:
    mask = pattern_search.apply_condition(disc_df, condition_dict)
    pattern_vals = disc_df.loc[mask, target_col].dropna().values
    rest_vals = disc_df.loc[~mask, target_col].dropna().values

    result = bootstrap_survival(pattern_vals)
    result.update(permutation_test(rest_vals, pattern_vals))
    observed_expectancy = float(pattern_vals.mean()) if len(pattern_vals) else np.nan
    # Direction-aware: for a NEGATIVE pattern (a candidate "avoid" filter),
    # a high prob_expectancy_negative means the negative edge is robust,
    # not fragile -- fragility is about the bootstrap distribution crossing
    # zero AGAINST the observed direction, so the tail that matters flips
    # with the sign of the observed effect.
    if pd.notna(observed_expectancy) and observed_expectancy < 0:
        result["prob_sign_flip"] = 1 - result["prob_expectancy_negative"]
    else:
        result["prob_sign_flip"] = result["prob_expectancy_negative"]
    result["mc_verdict"] = _verdict(result)
    return result


def _verdict(mc: dict) -> str:
    if pd.isna(mc.get("prob_sign_flip")):
        return "insufficient_data"
    if mc["prob_sign_flip"] > 0.10:
        return "rejected_fragile"  # >10% chance the bootstrap flips the observed direction
    if mc.get("perm_p_expectancy", 1.0) > 0.05:
        return "rejected_not_distinguishable_from_random_subset"
    return "survives_monte_carlo"


def run_mc_for_candidates(df: pd.DataFrame, candidates: pd.DataFrame, target_col: str,
                           extra_continuous_features: list = None) -> pd.DataFrame:
    disc = pattern_search.discretize(df, continuous_features=(
        pattern_search.CONTINUOUS_FEATURES + (extra_continuous_features or [])))
    mc_rows = [evaluate_pattern_mc(disc, row["condition_dict"], target_col) for _, row in candidates.iterrows()]
    mc_df = pd.DataFrame(mc_rows, index=candidates.index)
    return pd.concat([candidates, mc_df], axis=1)


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    candidates = pattern_search.search_patterns(feats, target_col="r_multiple", min_n=20, max_condition_depth=2)
    top = candidates.head(15)
    result = run_mc_for_candidates(feats, top, target_col="r_multiple")

    cols = ["condition", "n", "expectancy", "p_value_expectancy", "prob_sign_flip",
            "perm_p_expectancy", "mc_verdict"]
    print(result[cols].to_string(index=False))
    print()
    print(result["mc_verdict"].value_counts())
