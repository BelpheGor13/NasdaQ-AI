"""Stage 12: unsupervised clustering to check whether winners (and the
worst losers) form genuinely distinct archetypes rather than one
homogeneous profile. K-means on standardized pre-entry feature vectors;
k is chosen from {2,3} by silhouette score per the spec's "2-3 clusters"
expectation rather than left fully open-ended.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src import config, ml_shap

NUMERIC_CLUSTER_FEATURES = [
    "sl_pct_of_atr", "pct_dist_from_high_20d", "pct_dist_from_low_20d",
    "pct_dist_from_high_50d", "pct_dist_from_low_50d",
    "pre_entry_momentum_5m", "pre_entry_momentum_15m", "pre_entry_momentum_30m",
    "pre_entry_pct_up_candles_15m", "entry_hour_utc",
]


def longest_losing_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the trades (chronological) belonging to the single longest
    consecutive run of losing trades."""
    ordered = df.sort_values("dateStart_utc").reset_index(drop=True)
    is_loss = ordered["r_multiple"] < 0

    streak_id = (~is_loss).cumsum()
    streaks = ordered.loc[is_loss].groupby(streak_id[is_loss])
    lengths = streaks.size()
    if len(lengths) == 0:
        return ordered.iloc[0:0]
    longest_id = lengths.idxmax()
    return streaks.get_group(longest_id)


def _cluster(X: pd.DataFrame, k_candidates=(2, 3)) -> dict:
    X_clean = X.dropna()
    if len(X_clean) < 15:
        return {"error": "insufficient data for clustering", "n": len(X_clean)}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    best_k, best_score, best_labels = None, -1, None
    for k in k_candidates:
        if len(X_clean) <= k:
            continue
        km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        try:
            score = silhouette_score(X_scaled, labels)
        except ValueError:
            continue
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None:
        return {"error": "clustering failed for all k candidates", "n": len(X_clean)}

    profile = X_clean.copy()
    profile["cluster"] = best_labels
    cluster_means = profile.groupby("cluster").mean()
    cluster_sizes = profile.groupby("cluster").size()

    return {
        "k": best_k,
        "silhouette_score": best_score,
        "cluster_sizes": cluster_sizes.to_dict(),
        "cluster_means": cluster_means,
        "n": len(X_clean),
    }


def cluster_winners_and_worst_losers(df: pd.DataFrame, worst_loser_r_threshold_pct: float = 0.1) -> dict:
    winners = df[df["r_multiple"] > 0]
    winner_result = _cluster(winners[NUMERIC_CLUSTER_FEATURES])

    n_worst = max(int(len(df) * worst_loser_r_threshold_pct), 15)
    worst_losers = df.nsmallest(n_worst, "r_multiple")
    worst_loser_result = _cluster(worst_losers[NUMERIC_CLUSTER_FEATURES])

    streak = longest_losing_streak(df)
    streak_result = _cluster(streak[NUMERIC_CLUSTER_FEATURES]) if len(streak) >= 15 else {
        "error": "losing streak too short to cluster", "n": len(streak), "streak_length": len(streak)}

    return {
        "winners": winner_result,
        "worst_losers": worst_loser_result,
        "losing_streak": streak_result,
        "losing_streak_length": len(streak),
        "losing_streak_dates": (streak["dateStart_utc"].min(), streak["dateStart_utc"].max()) if len(streak) else None,
    }


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    result = cluster_winners_and_worst_losers(feats)

    print(f"Longest losing streak: {result['losing_streak_length']} trades, "
          f"{result['losing_streak_dates']}")
    print()
    for name in ["winners", "worst_losers", "losing_streak"]:
        r = result[name]
        print(f"=== {name} ===")
        if "error" in r:
            print(f"  {r['error']} (n={r['n']})")
            continue
        print(f"  k={r['k']}  silhouette={r['silhouette_score']:.3f}  sizes={r['cluster_sizes']}")
        print(r["cluster_means"].round(3))
        print()
