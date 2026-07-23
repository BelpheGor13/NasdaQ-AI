"""Stage 11 (optional/exploratory): LightGBM + SHAP purely as an
interpretability tool to see which composite features matter -- not a
signal generator, never traded directly. Fit fresh inside each
walk-forward fold (never once on all data) and check whether the
"important feature" ranking is stable across folds or reshuffles every
period. With ~789 trades this is treated as lower-confidence than any
directly-tested statistical pattern unless it also survives walk-forward +
Monte Carlo + stability.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from src import config

FEATURE_COLS = [
    "side", "day_of_week", "entry_hour_utc",
    "regime_trend_asof_prior_day", "regime_vol_asof_prior_day", "regime_expansion_asof_prior_day",
    "sl_pct_of_atr", "pct_dist_from_high_20d", "pct_dist_from_low_20d",
    "pct_dist_from_high_50d", "pct_dist_from_low_50d",
    "pre_entry_momentum_5m", "pre_entry_momentum_15m", "pre_entry_momentum_30m",
    "pre_entry_pct_up_candles_15m",
]
CATEGORICAL_COLS = ["side", "day_of_week", "entry_hour_utc",
                     "regime_trend_asof_prior_day", "regime_vol_asof_prior_day", "regime_expansion_asof_prior_day"]


def _prep_xy(df: pd.DataFrame, target_col: str):
    X = df[FEATURE_COLS].copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    y = df[target_col]
    valid = y.notna()
    return X.loc[valid], y.loc[valid]


def fit_fold_and_explain(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str = "r_multiple"):
    X_train, y_train = _prep_xy(train_df, target_col)
    X_test, y_test = _prep_xy(test_df, target_col)

    if len(X_train) < 40 or len(X_test) < 10:
        return None

    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=3, num_leaves=7, min_child_samples=15,
        learning_rate=0.05, random_state=config.RANDOM_SEED, verbosity=-1,
    )
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)

    return {"model": model, "mean_abs_shap": mean_abs_shap, "n_train": len(X_train), "n_test": len(X_test)}


def run_walk_forward_shap(df: pd.DataFrame, target_col: str = "r_multiple", folds=None) -> pd.DataFrame:
    folds = folds or config.WALK_FORWARD_FOLDS
    importances = {}

    for f in folds:
        test_year = pd.Timestamp(f["test_start"]).year
        train_mask = df["dateStart_utc"] < pd.Timestamp(f["train_end"])
        test_mask = (df["dateStart_utc"] >= pd.Timestamp(f["test_start"])) & \
                    (df["dateStart_utc"] <= pd.Timestamp(f["test_end"]))

        result = fit_fold_and_explain(df.loc[train_mask], df.loc[test_mask], target_col)
        if result is None:
            continue
        importances[test_year] = result["mean_abs_shap"]

    imp_df = pd.DataFrame(importances)
    imp_df["mean_rank"] = imp_df.rank(ascending=False).mean(axis=1)
    imp_df = imp_df.sort_values("mean_rank")
    return imp_df


def stability_of_top_features(imp_df: pd.DataFrame, top_k: int = 5) -> dict:
    fold_cols = [c for c in imp_df.columns if c != "mean_rank"]
    top_sets = []
    for c in fold_cols:
        top_sets.append(set(imp_df[c].dropna().sort_values(ascending=False).head(top_k).index))

    if len(top_sets) < 2:
        return {"stable": None, "note": "fewer than 2 folds with valid fits"}

    pairwise_overlap = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            inter = len(top_sets[i] & top_sets[j])
            pairwise_overlap.append(inter / top_k)

    mean_overlap = float(np.mean(pairwise_overlap))
    return {"mean_top_k_overlap_fraction": mean_overlap,
            "stable": mean_overlap >= 0.6,
            "top_sets_by_fold": top_sets}


if __name__ == "__main__":
    from src import data_loading, feature_engineering

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)

    imp_df = run_walk_forward_shap(feats, target_col="r_multiple")
    print(imp_df)
    print()
    print(stability_of_top_features(imp_df))
