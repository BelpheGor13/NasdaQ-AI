"""Stage 9: Bayesian evidence layer, computed alongside (not instead of)
the frequentist p-values already produced in pattern_search.py.

For each candidate's win rate: a Beta-Binomial posterior with credible
interval against a flat Beta(1,1) prior, plus an approximate Bayes factor
against the base win rate (Savage-Dickey ratio, see stats_utils). FDR/
Bonferroni correction (already computed in pattern_search.search_patterns)
is the multiple-testing control; this module adds strength-of-evidence,
not another significance gate.
"""
import numpy as np
import pandas as pd

from src import pattern_search, stats_utils


def add_bayesian_evidence(df: pd.DataFrame, candidates: pd.DataFrame, target_col: str) -> pd.DataFrame:
    disc = pattern_search.discretize(df)
    base_vals = disc[target_col].dropna().values
    base_win_rate = float((base_vals > 0).mean())

    rows = []
    for _, row in candidates.iterrows():
        mask = pattern_search.apply_condition(disc, row["condition_dict"])
        vals = disc.loc[mask, target_col].dropna().values
        n = len(vals)
        wins = int((vals > 0).sum())

        post = stats_utils.beta_binomial_posterior(wins, n)
        bf = stats_utils.bayes_factor_binomial(wins, n, base_win_rate)
        rows.append({
            "posterior_win_rate_mean": post["posterior_mean"],
            "posterior_win_rate_ci_lo": post["ci_lo"],
            "posterior_win_rate_ci_hi": post["ci_hi"],
            "bayes_factor_vs_base_rate": bf,
            "bayes_evidence": _interpret_bf(bf),
        })

    return pd.concat([candidates.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _interpret_bf(bf: float) -> str:
    # Standard Jeffreys/Kass-Raftery scale
    if pd.isna(bf):
        return "n/a"
    if bf < 1:
        return "favors base rate"
    if bf < 3:
        return "barely worth mentioning"
    if bf < 10:
        return "substantial"
    if bf < 30:
        return "strong"
    if bf < 100:
        return "very strong"
    return "decisive"


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    candidates = pattern_search.search_patterns(feats, target_col="r_multiple", min_n=20, max_condition_depth=2)
    result = add_bayesian_evidence(feats, candidates.head(15), target_col="r_multiple")

    cols = ["condition", "n", "win_rate", "posterior_win_rate_mean", "posterior_win_rate_ci_lo",
            "posterior_win_rate_ci_hi", "bayes_factor_vs_base_rate", "bayes_evidence"]
    print(result[cols].to_string(index=False))
