"""End-to-end, re-runnable pipeline: data loading -> excursion/exit-quality
-> regime -> features -> pattern search -> walk-forward -> Monte Carlo ->
Bayesian evidence -> stability/regime validation -> ML/SHAP -> clustering
-> edge decay -> robustness scoring -> visualizations -> Arabic report.

Run with: python main.py
"""
import time

from src import (
    config, data_loading, excursion, exit_quality, regime_detection,
    feature_engineering, pattern_search, walk_forward, monte_carlo,
    bayesian_evidence, stability_regime, ml_shap, shap_hypothesis_test,
    clustering, edge_decay, scoring, visualizations, reporting,
)

TARGET_COL = "r_multiple"
POTENTIAL_TARGET_COL = "maxRiskReward"
N_CANDIDATES_TO_ENRICH = 30


def run_pattern_pipeline(feats, target_col: str, n_enrich: int = N_CANDIDATES_TO_ENRICH):
    """Runs search -> walk-forward -> Monte Carlo -> Bayesian evidence ->
    stability/regime -> scoring for one target column. Used for both the
    realized-outcome target (r_multiple) and the potential target
    (maxRiskReward), per spec item 5: search for patterns associated with
    (a) win/loss and (b) high/low maxRiskReward potential.
    """
    candidates = pattern_search.search_patterns(feats, target_col=target_col, min_n=20, max_condition_depth=2)
    n_candidates_tested = len(candidates)
    n_fdr_survivors = int(candidates["reject_fdr"].sum())
    print(f"  tested {n_candidates_tested} candidates, {n_fdr_survivors} survive FDR correction")

    top = candidates.head(n_enrich).reset_index(drop=True)

    wf, fold_details = walk_forward.run_walk_forward_for_candidates(feats, top, target_col=target_col, top_n=len(top))
    mc = monte_carlo.run_mc_for_candidates(feats, top, target_col=target_col)
    bay = bayesian_evidence.add_bayesian_evidence(feats, top, target_col=target_col)
    stab = stability_regime.run_stability_regime_for_candidates(feats, top, target_col=target_col)

    merged = top.copy()
    for extra, cols in [
        (wf, ["n_folds_with_data", "n_folds_positive", "expectancy_sign_consistent", "expectancy_std_across_folds"]),
        (mc, ["prob_sign_flip", "perm_p_expectancy", "mc_verdict"]),
        (bay, ["posterior_win_rate_mean", "posterior_win_rate_ci_lo", "posterior_win_rate_ci_hi",
               "bayes_factor_vs_base_rate", "bayes_evidence"]),
        (stab, ["stability_applicable", "stability_pass_fraction", "stability_verdict",
                "regime_dependent_on", "regime_verdict"]),
    ]:
        for c in cols:
            merged[c] = extra[c].values

    scored = scoring.compute_robustness_score(merged)
    return candidates, scored, fold_details, n_candidates_tested, n_fdr_survivors


def main():
    t0 = time.time()
    config.DATA_OUT.mkdir(parents=True, exist_ok=True)
    config.REPORTS_OUT.mkdir(parents=True, exist_ok=True)

    print("[1/9] Loading & profiling data...")
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    profile = data_loading.profile(trades, candles)
    data_loading.print_profile(profile)

    print("\n[2/9] Reconstructing excursions & exit quality...")
    exc = excursion.reconstruct_excursions(trades, candles)
    exit_q = exit_quality.classify_exit_quality(exc)
    exit_summary = {
        "counts": exit_q["exit_quality"].value_counts().to_dict(),
        "total_left_on_table_usd": float(exit_q["left_on_table_usd"].sum()),
    }

    print("\n[3/9] Building pre-entry features...")
    feats = feature_engineering.build_features(trades, candles)
    # carry exit-quality columns forward for reference in outputs (descriptive only)
    feats = feats.merge(
        exit_q[["id", "exit_quality", "exit_efficiency", "left_on_table_r", "left_on_table_usd", "mfe_r", "mae_r"]],
        on="id", how="left",
    )

    print("\n[4/9] Searching composite pattern space (realized outcome: r_multiple)...")
    candidates, scored, fold_details, n_candidates_tested, n_fdr_survivors = run_pattern_pipeline(feats, TARGET_COL)
    candidates.drop(columns=["condition_dict"]).to_csv(config.DATA_OUT / "all_candidates.csv", index=False)
    scored.drop(columns=["condition_dict", "regime_dependent_on"]).to_csv(
        config.DATA_OUT / "top_scored_patterns.csv", index=False)

    print("\n[4b/9] Searching composite pattern space (potential: maxRiskReward)...")
    candidates_pot, scored_pot, fold_details_pot, n_candidates_tested_pot, n_fdr_survivors_pot = \
        run_pattern_pipeline(feats, POTENTIAL_TARGET_COL)
    candidates_pot.drop(columns=["condition_dict"]).to_csv(config.DATA_OUT / "all_candidates_potential.csv", index=False)
    scored_pot.drop(columns=["condition_dict", "regime_dependent_on"]).to_csv(
        config.DATA_OUT / "top_scored_patterns_potential.csv", index=False)

    # Cross-reference: does the top realized-outcome pattern also show up
    # as significant for the potential target? Agreement across two
    # independently-derived targets (actual R vs MFE-based potential) is
    # stronger evidence than either alone.
    cross_ref = None
    if len(scored) and len(candidates_pot):
        top_condition = scored.iloc[0]["condition"]
        match = candidates_pot[candidates_pot["condition"] == top_condition]
        if len(match):
            cross_ref = {"condition": top_condition, "p_fdr_bh_on_potential": float(match.iloc[0]["p_fdr_bh"]),
                         "expectancy_on_potential": float(match.iloc[0]["expectancy"])}

    print("\n[6/9] ML/SHAP interpretability layer (exploratory only)...")
    imp_df = ml_shap.run_walk_forward_shap(feats, target_col=TARGET_COL)
    shap_stability = ml_shap.stability_of_top_features(imp_df)

    print("\n[6b/9] SHAP-guided targeted hypothesis test (pre-registered family)...")
    shap_features = shap_hypothesis_test.select_shap_stable_features(feats)
    shap_scored, _ = shap_hypothesis_test.run_shap_guided_test(feats, shap_features)
    shap_scored.drop(columns=["condition_dict", "regime_dependent_on"]).to_csv(
        config.DATA_OUT / "shap_guided_test.csv", index=False)
    print(f"  {len(shap_scored)} conditions tested, "
          f"{int(shap_scored['reject_fdr'].sum())} survive Bonferroni within this small family")

    print("\n[7/9] Clustering winners / worst losers / losing streak...")
    clusters = clustering.cluster_winners_and_worst_losers(feats)

    print("\n[8/9] Edge decay analysis...")
    yearly = edge_decay.yearly_performance(feats)
    trend = edge_decay.sequence_trend_test(feats)
    eq_cps = edge_decay.equity_curve_change_points(feats)
    regime_daily = regime_detection.compute_regime(candles)
    regime_cps = regime_detection.cusum_change_points(regime_daily["atr14"], regime_daily["date"])
    ties = edge_decay.tie_decay_to_regime_changes(eq_cps, regime_cps)

    print("\n[9/9] Visualizations & report...")
    base_win_rate = float((feats[TARGET_COL] > 0).mean())
    visualizations.plot_exit_quality_distribution(exit_q)
    visualizations.plot_mfe_mae_scatter(exit_q)
    visualizations.plot_win_rate_vs_base_rate(scored, base_win_rate)
    visualizations.plot_yearly_edge_decay(yearly)

    # Per-pattern visuals (entry conditions, pattern-specific MFE/MAE,
    # Monte Carlo distribution, walk-forward folds) for the top 3 ranked
    # patterns, not just #1 -- spec asks for these "for top confirmed
    # patterns" (plural).
    disc = pattern_search.discretize(feats)
    for rank, (_, row) in enumerate(scored.head(3).iterrows(), start=1):
        mask = pattern_search.apply_condition(disc, row["condition_dict"])
        vals = disc.loc[mask, TARGET_COL].dropna().values
        visualizations.plot_entry_conditions(feats, row["condition_dict"], rank=rank, condition_label=row["condition"])
        visualizations.plot_mfe_mae_for_pattern(feats, mask, rank=rank, condition_label=row["condition"])
        visualizations.plot_monte_carlo_distribution(vals, rank=rank)
        visualizations.plot_walk_forward_folds(row["condition"], fold_details[row["condition"]], rank=rank)

    results = {
        "profile": profile,
        "exit_summary": exit_summary,
        "yearly_performance": yearly,
        "sequence_trend": trend,
        "top_scored": scored,
        "clustering": clusters,
        "shap_stability": shap_stability,
        "base_win_rate": base_win_rate,
        "n_candidates_tested": n_candidates_tested,
        "n_fdr_survivors": n_fdr_survivors,
        "top_scored_potential": scored_pot,
        "n_candidates_tested_potential": n_candidates_tested_pot,
        "n_fdr_survivors_potential": n_fdr_survivors_pot,
        "cross_ref": cross_ref,
        "shap_features": shap_features,
        "shap_guided_scored": shap_scored,
        "equity_change_points": eq_cps,
        "regime_change_points": regime_cps,
        "decay_regime_ties": ties,
        "fold_details": fold_details,
    }

    report_text = reporting.build_report(results)
    report_path = config.REPORTS_OUT / "final_report_arabic.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Report: {report_path}")
    print(f"Candidate data: {config.DATA_OUT}")
    print(f"Figures: {config.FIGURES_OUT}")


if __name__ == "__main__":
    main()
