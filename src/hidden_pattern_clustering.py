"""Phase 2: unsupervised entry-context clustering ("hidden pattern
discovery"). K-means on standardized pre-entry-only features (see
hidden_pattern_features.CLUSTERING_FEATURES_NUMERIC/CATEGORICAL); k chosen
from config.CLUSTER_K_CANDIDATES by silhouette score.

fit_clusters/assign_clusters are split so Phase 7 (train/test validation)
can fit on the train fold only and assign test trades to the NEAREST
EXISTING centroid, rather than re-discovering clusters on the test fold
(which would leak future information into what's supposed to be a fixed,
pre-registered rule).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src import config, hidden_pattern_features as hpf

CATEGORICAL_LEVELS = {
    "regime_trend_asof_prior_day": ["Trend", "Range"],
    "regime_vol_asof_prior_day": ["High Vol", "Low Vol"],
    "session": ["Asia", "London", "NY_Open_Overlap", "NY_Afternoon", "Late_NY_PreAsia"],
    "prev_trade_was_win": ["prev_win", "prev_loss", "no_prior_trade"],
}


def build_cluster_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a fully numeric matrix (one-hot categoricals + numeric
    features), row-aligned to df's index, with rows containing any NaN
    dropped (caller must reconcile ids against the returned index)."""
    numeric = df[hpf.CLUSTERING_FEATURES_NUMERIC]

    dummies = []
    for col, levels in CATEGORICAL_LEVELS.items():
        d = pd.get_dummies(df[col].astype("category").cat.set_categories(levels), prefix=col)
        dummies.append(d)

    X = pd.concat([numeric] + dummies, axis=1)
    return X.dropna()


def fit_clusters(X: pd.DataFrame, k_candidates=config.CLUSTER_K_CANDIDATES) -> dict:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, best_score, best_labels, best_model = None, -1, None, None
    for k in k_candidates:
        if len(X) <= k:
            continue
        km = KMeans(n_clusters=k, random_state=config.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(X_scaled)
        try:
            score = silhouette_score(X_scaled, labels)
        except ValueError:
            continue
        if score > best_score:
            best_k, best_score, best_labels, best_model = k, score, labels, km

    return {
        "scaler": scaler, "model": best_model, "k": best_k,
        "silhouette_score": best_score, "labels": best_labels, "columns": list(X.columns),
    }


def assign_clusters(fit_result: dict, X_new: pd.DataFrame) -> np.ndarray:
    X_aligned = X_new.reindex(columns=fit_result["columns"], fill_value=0)
    X_scaled = fit_result["scaler"].transform(X_aligned)
    return fit_result["model"].predict(X_scaled)


def _majority(series: pd.Series) -> str:
    return series.mode().iloc[0] if len(series.dropna()) else "n/a"


def name_and_profile_clusters(df: pd.DataFrame, labels: np.ndarray, index) -> pd.DataFrame:
    """Human-readable name + key characteristics per cluster, computed on
    whichever rows the caller passes in (train-only or full sample)."""
    profile_df = df.loc[index].copy()
    profile_df["cluster"] = labels

    rows = []
    for cluster_id, group in profile_df.groupby("cluster"):
        trend = _majority(group["regime_trend_asof_prior_day"])
        vol = _majority(group["regime_vol_asof_prior_day"])
        mom_sign = "Momentum إيجابي" if group["pre_entry_momentum_15m"].mean() > 0 else "Momentum سلبي"
        name = f"{trend} + {vol} + {mom_sign}"
        rows.append({
            "cluster": cluster_id,
            "cluster_name": name,
            "size": len(group),
            "pct_of_total": len(group) / len(df),
            "low_confidence": len(group) < config.MIN_CLUSTER_SIZE,
            "dominant_regime_trend": trend,
            "dominant_regime_vol": vol,
            "mean_pre_entry_momentum_15m": group["pre_entry_momentum_15m"].mean(),
            "mean_sl_pct_of_atr": group["sl_pct_of_atr"].mean(),
            "mean_risk_reward_setup": group["risk_reward_setup"].mean(),
            "mean_speed_of_profit_r_30m": group["speed_of_profit_r_30m"].mean(),  # descriptive cross-tab only
        })
    return pd.DataFrame(rows).sort_values("cluster")


if __name__ == "__main__":
    from src import data_loading

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = hpf.build_hidden_pattern_features(trades, candles)

    X = build_cluster_matrix(feats)
    print(f"clustering matrix: {X.shape} (dropped {len(feats) - len(X)} rows with NaN features)")

    result = fit_clusters(X)
    print(f"best k={result['k']}, silhouette={result['silhouette_score']:.3f}")

    profile = name_and_profile_clusters(feats, result["labels"], X.index)
    print(profile.to_string(index=False))
