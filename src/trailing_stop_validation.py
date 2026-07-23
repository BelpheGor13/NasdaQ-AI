"""Overfitting check (deliverable 5): chronological 60/40 train/test split.
Pick the "best" trailing pct by profit factor on the first
config.TRAILING_TRAIN_FRACTION of trades (by dateStart_utc), then check
whether that SAME pct is still the best -- or even still beats baseline --
on the held-out last 40%. Reported for the full grid so both train and test
performance are visible for the top-3 train-selected candidates.
"""
import pandas as pd

from src import config, stats_utils


def train_test_split_chronological(sim: pd.DataFrame, scenario: str = "conservative",
                                    train_fraction: float = config.TRAILING_TRAIN_FRACTION):
    s = sim[sim["scenario"] == scenario]
    ids_sorted = s.drop_duplicates("id").sort_values("dateStart_utc")["id"]
    split_at = int(len(ids_sorted) * train_fraction)
    train_ids = set(ids_sorted.iloc[:split_at])
    test_ids = set(ids_sorted.iloc[split_at:])
    return s[s["id"].isin(train_ids)], s[s["id"].isin(test_ids)]


def _pf_by_pct(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pct, group in df.groupby("pct"):
        r = group["trail_r"].dropna().values
        rows.append({
            "pct": pct,
            "n": len(r),
            "expectancy": stats_utils.expectancy(r),
            "profit_factor": stats_utils.profit_factor(r),
        })
    return pd.DataFrame(rows)


def run_overfitting_check(sim: pd.DataFrame, scenario: str = "conservative", top_n: int = 3) -> dict:
    train, test = train_test_split_chronological(sim, scenario=scenario)

    baseline_train_pf = stats_utils.profit_factor(train.drop_duplicates("id")["orig_r"].values)
    baseline_test_pf = stats_utils.profit_factor(test.drop_duplicates("id")["orig_r"].values)

    train_pf = _pf_by_pct(train).sort_values("profit_factor", ascending=False)
    test_pf = _pf_by_pct(test).set_index("pct")

    top_train = train_pf.head(top_n).copy()
    top_train["test_profit_factor"] = top_train["pct"].map(test_pf["profit_factor"])
    top_train["test_expectancy"] = top_train["pct"].map(test_pf["expectancy"])
    top_train = top_train.rename(columns={"expectancy": "train_expectancy", "profit_factor": "train_profit_factor",
                                           "n": "train_n"})

    best_on_train_pct = train_pf.iloc[0]["pct"]
    best_on_test_pct = test_pf["profit_factor"].idxmax() if len(test_pf) else None
    consistent = best_on_train_pct == best_on_test_pct

    return {
        "baseline_train_pf": baseline_train_pf,
        "baseline_test_pf": baseline_test_pf,
        "top_train_candidates": top_train,
        "best_on_train_pct": best_on_train_pct,
        "best_on_test_pct": best_on_test_pct,
        "best_pct_consistent_train_test": bool(consistent),
        "train_n_trades": len(train["id"].unique()),
        "test_n_trades": len(test["id"].unique()),
    }


if __name__ == "__main__":
    from src import data_loading, trailing_stop_simulation

    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = trailing_stop_simulation.simulate_trailing_stops(trades, candles)

    result = run_overfitting_check(sim)
    print(f"train n={result['train_n_trades']}, test n={result['test_n_trades']}")
    print(f"baseline PF: train={result['baseline_train_pf']:.3f}, test={result['baseline_test_pf']:.3f}")
    print(f"best pct on train: {result['best_on_train_pct']}, best pct on test: {result['best_on_test_pct']}")
    print(f"consistent: {result['best_pct_consistent_train_test']}")
    print()
    print(result["top_train_candidates"].to_string(index=False))
