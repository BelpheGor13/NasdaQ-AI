"""Shared statistical primitives used by pattern_search, walk_forward, and
monte_carlo so every stage applies the same definitions consistently.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln


def profit_factor(r_values: np.ndarray) -> float:
    wins = r_values[r_values > 0].sum()
    losses = -r_values[r_values < 0].sum()
    if losses == 0:
        return np.inf if wins > 0 else np.nan
    return wins / losses


def expectancy(r_values: np.ndarray) -> float:
    return float(np.mean(r_values)) if len(r_values) else np.nan


def two_proportion_ztest(wins_a: int, n_a: int, wins_b: int, n_b: int) -> float:
    """Two-sided p-value for a difference in win rates (pattern vs base rate)."""
    if n_a == 0 or n_b == 0:
        return np.nan
    p_a, p_b = wins_a / n_a, wins_b / n_b
    p_pool = (wins_a + wins_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (p_a - p_b) / se
    return float(2 * (1 - stats.norm.cdf(abs(z))))


def welch_ttest_pvalue(sample: np.ndarray, population: np.ndarray) -> float:
    if len(sample) < 2 or len(population) < 2:
        return np.nan
    _, p = stats.ttest_ind(sample, population, equal_var=False)
    return float(p)


def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, n_boot: int = 2000,
                  alpha: float = 0.05, seed: int = 42) -> tuple:
    """Vectorized when stat_fn is np.mean (the hot path for pattern search /
    walk-forward, called thousands of times on small samples); falls back to
    a plain Python loop for arbitrary stat_fn.
    """
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(values)
    idx = rng.integers(0, n, (n_boot, n))
    if stat_fn is np.mean:
        boots = values[idx].mean(axis=1)
    else:
        boots = np.array([stat_fn(values[row]) for row in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def benjamini_hochberg(pvalues: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Returns a DataFrame aligned to pvalues.index with rank, corrected
    threshold, BH-adjusted p-value, and a reject flag."""
    valid = pvalues.dropna()
    m = len(valid)
    order = valid.sort_values().index
    ranks = pd.Series(range(1, m + 1), index=order)

    sorted_p = valid.loc[order].values
    adj = sorted_p * m / ranks.values
    adj_monotone = np.minimum.accumulate(adj[::-1])[::-1]
    adj_monotone = np.clip(adj_monotone, 0, 1)

    result = pd.DataFrame(index=pvalues.index)
    result["p_raw"] = pvalues
    result["p_fdr_bh"] = np.nan
    result.loc[order, "p_fdr_bh"] = adj_monotone
    result["reject_fdr"] = result["p_fdr_bh"] < alpha
    result["p_bonferroni"] = (pvalues * m).clip(upper=1.0)
    result["reject_bonferroni"] = result["p_bonferroni"] < alpha
    return result


def beta_binomial_posterior(wins: int, n: int, prior_alpha: float = 1.0,
                             prior_beta: float = 1.0, alpha: float = 0.05) -> dict:
    """Beta-Binomial posterior for a win rate, with credible interval."""
    a = prior_alpha + wins
    b = prior_beta + (n - wins)
    mean = a / (a + b)
    lo, hi = stats.beta.ppf([alpha / 2, 1 - alpha / 2], a, b)
    return {"posterior_mean": float(mean), "ci_lo": float(lo), "ci_hi": float(hi), "alpha_post": a, "beta_post": b}


def bayes_factor_binomial(wins: int, n: int, base_rate: float) -> float:
    """Approximate Bayes factor comparing H1 (rate != base_rate, uniform
    prior) vs H0 (rate == base_rate), via the Savage-Dickey density ratio
    using a Beta(1,1) prior under H1.
    """
    if n == 0:
        return np.nan
    log_marginal_h1 = betaln(1 + wins, 1 + n - wins) - betaln(1, 1)
    log_lik_h0 = wins * np.log(max(base_rate, 1e-9)) + (n - wins) * np.log(max(1 - base_rate, 1e-9))
    log_bf = log_marginal_h1 - log_lik_h0
    return float(np.exp(log_bf))
