# Prompt for Claude Code — Composite Pattern Discovery & Robustness Engine (NAS100 Trading Data)
## Prior context from earlier analysis (verified, save re-discovery time)

- `idealTP` in the CSV is confirmed (cross-referenced against the broker
  platform's own "Max RR" display field) to represent Maximum Favorable
  Excursion achieved — NOT a pre-trade planned target. Treat any
  R-multiple derived from it as a *descriptive/outcome* metric, never
  as a pre-entry feature.
- Real MFE/MAE was already computed independently from 1-minute candles
  and correlates with idealTP-derived R at only r=0.51 — they are
  related but NOT the same measurement (idealTP appears to sometimes
  reflect price action beyond the trade's actual closed lifetime).
  Prefer the candle-derived MFE/MAE as ground truth over idealTP.
- Timezone: trade timestamps are in America/New_York (handles DST
  automatically); candle data is in UTC. Convert trade times to UTC
  before merging with candles.
- Entry/stop columns (`entryPrice`, `initalSL`) were verified reliable:
  zero trades found with stop on the wrong side of entry (no evidence
  of post-entry stop reversal contaminating this field).
## Context

I have a personal trading strategy on OANDA:NAS100USD with historical trade records. I want to build a rigorous, statistically-grounded **pattern discovery and robustness-testing system** — not a black box, not a simple backtest, but a research tool that finds, stress-tests, and ranks *composite* (multi-condition) patterns correlated with winning or losing trades, using both my trade log and raw 1-minute candle data.

**Guiding principle for the whole project — this shapes every decision below:**
Find statistically significant AND economically meaningful patterns that remain robust across different market regimes, time splits, and Monte Carlo perturbations. Prioritize robustness over raw performance. A pattern that looks great on one arbitrary split or one regime and falls apart elsewhere is not a finding — it's noise. A pattern with a modest edge that survives every stress test below is far more valuable than one with a huge edge that survives none.

**Data-scale warning to keep in mind throughout:** there are only ~796 trades across 5 years. Every additional layer of segmentation (regime × time-split × pattern condition) shrinks the sample in that cell further. Treat sample size as a hard constraint at every stage, not an afterthought — prefer fewer, well-justified regime categories over many.

## Data available (both files are local; do not fabricate schemas — inspect them first)

1. **`trades.csv`** (~796 closed trades, 2020–2024, single instrument):
   Columns include: `id, dateStart, dateEnd, pair, uPnL, rPnL, side, entryPrice, initalSL, maxTP, idealTP, amount, amountClosed, status, day, tags, avgClosePrice, avgRiskReward, maxRiskReward, exchangeRate, initialBalance, currentRealizedBalance`.
   - `maxRiskReward` and `maxTP` represent the *best the trade could have achieved* (maximum favorable excursion in R-multiples / price), different from where it was actually closed (`avgClosePrice`, `rPnL`).
   - `tags` is sparsely populated — treat as partial metadata, not a reliable label.

2. **1-minute OHLC(V) candle data** for the same instrument/period (file path and exact schema to be confirmed at runtime — inspect before assuming column names).

**First step of any session: actually load and profile both files (shapes, dtypes, date ranges, nulls, timezone alignment between the trade log and the candle data) before writing any analysis code. Do not assume alignment — verify it.**

### Optional external context data (secondary, do not block core analysis on this)
Regime detection (Task 4) should primarily be derived **internally** from the NAS100 1-minute candles themselves (ATR percentile, ADX, Bollinger Band width, realized volatility) so the core pipeline has no external dependency and can't silently break. As an optional enrichment layer only, you may pull:
- **VIX** (CBOE Volatility Index) — official series `VIXCLS` on **FRED** (Federal Reserve Economic Data, free, authoritative, no scraping): `https://fred.stlouisfed.org/series/VIXCLS`.
- A dollar-strength proxy — FRED's **Trade Weighted U.S. Dollar Index** (`DTWEXBGS`) is the closest free, official equivalent to the (non-free-licensed) ICE DXY; note it is *not identical* to DXY, so label it clearly as a proxy, not the real DXY.
- If you use `yfinance` for anything, flag it explicitly as an **unofficial, best-effort source** (it scrapes undocumented Yahoo endpoints, breaks occasionally, and its terms restrict it to personal/research use) — fine for exploratory context, not something to treat as ground truth. Cross-check any surprising value against a second source before relying on it.
- Skip economic-calendar/news-event data (FOMC, NFP, CPI dates) unless it turns out to be needed — it adds scope and most free sources for it have unclear scraping terms. If truly needed later, ask me first.

## Objective

Build a system that:

1. Merges each trade with its surrounding 1-minute candle context (before, during, and after the trade window).
2. Reconstructs, from 1-minute candles, the **full excursion path** of every trade — MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion), in both price and R-multiples — between `dateStart` and `dateEnd`.
3. Classifies **Exit Quality** per trade, not just raw MFE/MAE: e.g. `Perfect Exit` (closed at/near the best available price), `Good Exit`, `Early Exit`, `Terrible Exit` (gave back most of an open profit or held through most of an adverse move). Quantify, in dollars and in R, how much was left on the table by early/poor exits, separately from entry-side pattern quality — a trade can have a great entry pattern and a terrible exit, or vice versa, and the system should be able to say which.
4. Classifies each period into a **market regime**: Trend vs Range, High vs Low Volatility, Expansion vs Compression (derived from the instrument's own candles — see above). Detects **regime change points** (e.g., CUSUM or Bayesian change-point detection) so we know roughly when the market character shifted, not just that it did on average.
5. Constructs **composite features** — combinations, not single-variable checks (time-of-day × day-of-week × entry-side × initial-SL-size-as-%-of-ATR × pre-entry volatility regime × distance from recent high/low × candle pattern at entry, etc.) — and searches this space for combinations statistically associated with (a) win/loss, and (b) high vs low `maxRiskReward` potential.
6. Is **not limited to the pattern families listed here**. Treat this as a seed set. Actively search for other statistically robust composite patterns I haven't thought to ask about, and report any "unprompted findings" with the same rigor as the requested ones.
7. Tracks **Edge Decay** over time: compute rolling/yearly Profit Factor, expectancy, and win rate, and test whether there's a statistically meaningful downward trend (not just eyeballing the numbers) — i.e., is the strategy's edge shrinking year over year, and if so, does that trend itself survive scrutiny (vs. just being one bad year)?

## Methodology — robustness stack (this is the core of the project; do not skip or soften any layer)

1. **No look-ahead bias.** Any feature computed "before entry" must only use data strictly before `dateStart`. Anything using candles *during* the trade (MFE/MAE, exit-quality) is descriptive research about the trade's own path, not a tradeable entry signal — label it as such.

2. **Walk-forward validation, not a single static split.** Use an expanding-window walk-forward scheme, e.g.:
   - Train 2020 → Test 2021
   - Train 2020–2021 → Test 2022
   - Train 2020–2022 → Test 2023
   - Train 2020–2023 → Test 2024
   Report performance per fold, not just averaged — a pattern that works in 3 folds and collapses in 1 is a different finding than one that's mediocre-but-consistent everywhere. Flag folds with small training or test samples (early folds especially) as lower-confidence.

3. **Monte Carlo robustness testing for every candidate pattern (not just the overall strategy).** For each filter/pattern that clears the initial statistical bar, run Monte Carlo simulation (e.g., bootstrap resampling of the matched trades, and/or random reshuffling of trade order/outcomes under the pattern) and report the probability that the pattern's realized performance (PF, expectancy, drawdown) is a lucky artifact rather than a real edge. A pattern with PF = 1.9 but a 90% Monte Carlo probability of ruin/degradation should be flagged as **rejected**, regardless of its raw historical PF.

4. **Multiple hypothesis testing correction.** Scanning many composite combinations will surface "significant" patterns by chance. Apply Benjamini-Hochberg FDR correction (preferred) or Bonferroni, and report both raw and corrected p-values.

5. **Bayesian evidence alongside p-values.** Don't rely on p-values alone. For key comparisons (e.g., win rate of pattern vs. base rate), also compute a Bayesian estimate — a Beta-Binomial posterior for win rate with credible intervals, and/or a Bayes factor — so we can see the strength of evidence, not just a binary significant/not-significant call. Keep this lightweight and interpretable; this is a research aid, not a full Bayesian modeling exercise.

6. **Minimum sample size thresholds.** Any pattern backed by fewer than ~20–30 trades (or fewer than ~20–30 trades within a fold/regime) must be explicitly flagged as low-confidence/exploratory, regardless of effect size. Report sample size next to every claim, everywhere.

7. **Effect size + confidence intervals, not just significance.** Report win-rate delta, expectancy delta, and bootstrap confidence intervals for every candidate pattern.

8. **Stability checks.** For any pattern that survives the above, test whether it holds up under small perturbations of its condition thresholds (e.g., a session window defined a few reasonable different ways). A pattern that only works for one exact arbitrary cutoff is likely overfit.

9. **Regime-conditioned validation.** For every surviving pattern, report whether its edge is concentrated in a specific regime (e.g., only works in Trend/High-Vol) or holds across regimes — this changes how it should be used, not just whether it's "valid."

10. **If ML is used (optional, exploratory layer only):**
    - You may fit a tree ensemble (XGBoost or LightGBM) purely as an **interpretability tool**, not a signal generator — the goal is to see which composite features matter, not to trade the model's predictions.
    - Compute **SHAP values** for feature importance, nested inside the same walk-forward folds used everywhere else (fit fresh per fold — don't fit once on all data and call it validated).
    - Check **feature importance stability**: does a feature that ranks highly in one fold/year keep ranking highly in others, or does the "important feature" list reshuffle every period? Unstable importance is itself a finding (suggests overfitting or a non-stationary relationship), and should be reported as such, not hidden.
    - With ~796 trades, tree ensembles can overfit easily — treat any ML-derived finding as lower-confidence than a directly-tested statistical pattern unless it also survives the walk-forward + Monte Carlo + stability checks above.

11. **Unsupervised pattern clustering on winners.** Don't rely purely on rule mining. Run clustering (e.g., k-means or hierarchical) on the feature vectors of winning trades to see whether there are genuinely distinct "archetypes" of winners (e.g., 2–3 different clusters with different entry conditions) rather than one homogeneous winning profile — and do the same check for the worst losers, especially around the long losing streak in the data.

12. **Be explicit about what is exploratory vs. confirmed.** Final output must clearly separate: (a) patterns that survive walk-forward + correction + Monte Carlo + stability — genuinely promising; (b) interesting but low-sample/low-confidence findings; (c) things tested and found NOT significant (report some of these too — don't just show winners, that's survivorship/publication bias); (d) unstable/contradictory findings (e.g., ML feature importance that didn't hold up).

13. **Do not force a conclusion.** If nothing survives rigorous testing, say so plainly rather than presenting a marginal result as a discovery.

## Deliverables

1. A modular, well-commented Python codebase (comments/docstrings in English) with clearly separated stages: data loading/validation → candle-trade merge & excursion reconstruction → exit-quality classification → regime detection & change-point detection → feature engineering → pattern search → walk-forward validation → Monte Carlo stress test → Bayesian evidence → ML/SHAP layer (optional) → clustering → scoring → reporting. Make it re-runnable as new trade data arrives.

2. **A Robustness Score (0–100) per surviving pattern/filter**, combining (with the weighting scheme documented and justified, not arbitrary): Profit Factor, expectancy, drawdown, walk-forward stability across folds, holdout performance, Monte Carlo survival probability, sample size, and FDR-corrected significance. Rank all candidate patterns by this score — not by raw historical profit.

3. **An automatic "Top Filters" summary**: after everything runs, output the top ~10 surviving filters/patterns ranked by Robustness Score (not by raw PF), each with its full stat sheet (sample size, PF, expectancy, corrected p-value, Bayesian evidence, walk-forward per-fold results, Monte Carlo survival %, regime dependence, stability check result).

4. Visualizations: for top confirmed patterns — entry conditions, MFE/MAE/exit-quality path, win-rate vs. base rate, walk-forward fold-by-fold performance, Monte Carlo outcome distribution, and the yearly edge-decay trend.

5. A final written report **in Arabic** (code, variable names, comments, and column references stay in English — only narrative explanation, findings, and recommendations in Arabic), structured as:
   - Executive summary
   - Top robust filters (ranked, with full stat sheet)
   - Exploratory/low-confidence findings
   - Tested-but-rejected patterns (including any that failed specifically due to Monte Carlo or stability, not just p-value)
   - Edge decay analysis — is the strategy's edge shrinking, and since when (tie to change-point detection)?
   - Exit quality analysis — how much performance is being left on the table by exit timing, separate from entry pattern quality
   - Winner/loser clustering — distinct archetypes found, if any
   - Any unprompted findings outside the seed list
   - Concrete, prioritized next steps

6. Ask me before assuming file paths, exact column names, timezone handling, or regime-detection thresholds if anything is ambiguous after inspecting the data — don't guess silently on anything that affects correctness.
