"""Macro/cross-asset correlation test (test-only; direct answer to a user
question about gold/dollar/EUR/VIX/S&P500 composite patterns). Never
touches the core NAS100 pipeline or analytics_1.csv.

Run with: python -m src.macro_correlation_main
"""
import time

from src import (
    config, data_loading, feature_engineering, regime_detection,
    external_data, macro_feature_engineering as mfe, macro_pattern_search,
    macro_correlation_reporting,
)

REPORT_PATH = config.REPORTS_OUT / "macro_correlation_report_arabic.md"


def sanity_correlations(candles):
    nas_daily = regime_detection.resample_daily(candles)
    nas_daily["nas_ret"] = nas_daily["close"].pct_change()
    nas_daily["date"] = nas_daily["datetime_utc"].dt.normalize()

    macro = external_data.build_macro_daily_table()
    for col in ["vix", "dxy_proxy", "eurusd", "sp500", "gold"]:
        macro[col + "_ret"] = macro[col].pct_change()

    ret_cols = [c + "_ret" for c in ["vix", "dxy_proxy", "eurusd", "sp500", "gold"]]
    merged = nas_daily[["date", "nas_ret"]].merge(macro[["date"] + ret_cols], on="date", how="inner").dropna()
    return {c: float(merged["nas_ret"].corr(merged[c])) for c in ret_cols}


def main():
    t0 = time.time()
    config.REPORTS_OUT.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading data & fetching/caching external macro series...")
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()

    print("[2/4] Building macro features (no look-ahead) & sanity correlations...")
    feats = feature_engineering.build_features(trades, candles)
    macro_daily = mfe.build_macro_features(candles)
    feats = mfe.attach_macro_to_trades(feats, macro_daily)
    corr = sanity_correlations(candles)
    for k, v in corr.items():
        print(f"  {k}: {v:.3f}")

    print("\n[3/4] Running macro pattern search through full robustness stack...")
    scored, fold_details = macro_pattern_search.run_macro_pattern_search(feats)
    n_tested = len(scored)
    n_survivors = int(scored["reject_fdr"].sum()) if len(scored) else 0
    print(f"  {n_tested} conditions tested, {n_survivors} survive Bonferroni")
    if len(scored):
        scored.drop(columns=["condition_dict", "regime_dependent_on"]).to_csv(
            config.DATA_OUT / "macro_correlation_scored.csv", index=False)

    print("\n[4/4] Arabic report...")
    results = {"scored": scored, "sanity_correlations": corr, "n_tested": n_tested}
    report_text = macro_correlation_reporting.build_report(results)
    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
