"""Central configuration: paths, timezones, and documented threshold assumptions.

Any constant here that isn't directly derived from the data is a modeling
choice, not a fact. Where the source prompt flagged something as needing
confirmation, the choice made and its rationale are noted inline so it is
never a silent guess.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRADES_CSV = PROJECT_ROOT / "analytics_1.csv"
CANDLES_PARQUET = PROJECT_ROOT / "nasdaq_m1_2020_2024.parquet"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_OUT = OUTPUTS_DIR / "data"
FIGURES_OUT = OUTPUTS_DIR / "figures"
REPORTS_OUT = OUTPUTS_DIR / "reports"

# Trade timestamps are naive local America/New_York (handles DST). Candle
# timestamps are naive UTC. Both confirmed by prior verified analysis.
TRADE_TZ = "America/New_York"
CANDLE_TZ = "UTC"

# Minimum trades required before a pattern is reported as anything other
# than explicitly low-confidence/exploratory (spec: ~20-30).
MIN_SAMPLE_SIZE = 25

# Walk-forward folds: expanding window, one calendar year of test at a time.
WALK_FORWARD_FOLDS = [
    {"train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
]

# Monte Carlo iterations for bootstrap / reshuffle tests.
MC_ITERATIONS = 5000

# FDR correction level for multiple-hypothesis testing (Benjamini-Hochberg).
FDR_ALPHA = 0.05

RANDOM_SEED = 42

# --- Trailing-stop optimization (test-only, hypothetical post-analysis) ---
# See trailing-stop-optimization-prompt.md. Never touches rPnL / initalSL /
# entry logic -- purely simulates an alternative EXIT rule on the same trades.
TRAILING_STOP_REPORT = REPORTS_OUT / "trailing_stop_report_arabic.md"

# Grid of trailing-stop percentages tested, defined as: once a trade's
# running peak favorable price implies a gain of PEAK_R over entry, the
# hypothetical stop trails at peak - pct * (peak - entry). Exiting at that
# level means "gave back pct% of the peak open profit."
TRAILING_STOP_GRID = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

# Trailing only overrides the ORIGINAL stop-loss once the peak favorable
# excursion is material (reuses exit_quality.MATERIALITY_MFE_R) -- before
# that, the original initalSL is still the only active stop, consistent
# with "do not change stop-loss levels" in scope.
TRAILING_ACTIVATION_MFE_R = 0.15

# How far past the trade's original dateEnd_utc the "aggressive" scenario is
# allowed to look for a trailing-stop trigger that never fired within the
# original window. Documented assumption, not derived from data.
TRAILING_AGGRESSIVE_MAX_EXTENSION_MINUTES = 3 * 24 * 60  # 3 days

# Chronological train/test split fraction for the overfitting check (spec:
# fit on first 60%, validate on last 40%).
TRAILING_TRAIN_FRACTION = 0.6

# --- Hidden-pattern discovery + regime-conditional exit optimization ---
# (test-only; see hidden-patterns-exit-optimization-prompt.md). Same
# invariants as the trailing-stop analysis: never touches rPnL/initalSL/
# entry logic, only simulates alternative EXITS on the same 789 trades.
HIDDEN_PATTERN_REPORT = REPORTS_OUT / "hidden_pattern_report_arabic.md"

# Phase 2: candidate cluster counts to try (spec: "start with 2-5 clusters").
CLUSTER_K_CANDIDATES = (2, 3, 4, 5)

# Phase 3 exit-strategy grids.
FIXED_TP_R_MULTIPLES = [2.0, 3.0, 4.0]
TIME_BASED_EXIT_HOURS = [1, 2, 4, 8]
PARTIAL_PROFIT_LEGS = [(0.5, 2.0), (0.3, 4.0), (0.2, None)]  # (fraction, R-multiple); None = idealTP
# MFE-aware adaptive trailing: below the first breakpoint, no active
# management (ride to the original SL only); between the breakpoints, a
# moderate trail; above the second, a tight trail. Breakpoints and trail
# percentages are a documented operational choice (prompt marks this
# strategy "optional, advanced" without specifying the middle zone).
MFE_AWARE_LOW_R = 1.0
MFE_AWARE_HIGH_R = 5.0
MFE_AWARE_MID_TRAIL_PCT = 0.10
MFE_AWARE_TIGHT_TRAIL_PCT = 0.02

# Same extension cap as the trailing-stop analysis, reused for every
# exit strategy that might trigger after the trade's original dateEnd.
EXIT_STRATEGY_MAX_EXTENSION_MINUTES = TRAILING_AGGRESSIVE_MAX_EXTENSION_MINUTES

# Minimum trades per cluster before it's treated as anything other than
# low-confidence (spec: "<30 trades flagged as low-confidence").
MIN_CLUSTER_SIZE = 30

# --- No-stop-loss scenario (test-only, hypothetical): what if the original
# initalSL were removed entirely and the trade just ran? See
# no-stop-loss-prompt.md. Never touches the real trade log -- simulates one
# alternative, unprotected EXIT per trade on the same 789 trades.
NO_STOP_REPORT = REPORTS_OUT / "no_stop_report_arabic.md"

# Without a stop, a position has no natural end -- it needs a documented
# deadline. 30 days chosen because only 6/789 trades have less than 30 days
# of candle data available after them (checked against the dataset's actual
# end date), so this cap resolves the vast majority of trades on real
# candles rather than running out of data; the handful that do run out are
# flagged as data-censored, not silently treated as resolved.
NO_STOP_MAX_EXTENSION_MINUTES = 30 * 24 * 60

# A loss beyond this many R is labeled "catastrophic" in the tail-risk
# reporting. Chosen as 3x the normal fixed risk unit -- roughly the smallest
# adverse excursion actually observed among the worst historical trades
# (7.6R, 5.8R, 4.3R were seen with the real stop in place), so "catastrophic"
# reflects this dataset's own tail, not an arbitrary round number.
NO_STOP_CATASTROPHIC_R_THRESHOLD = -3.0

# --- "Target-or-stop" scenario (test-only): what if every trade were left
# alone -- original entry, original initalSL, a fixed target -- with NO
# early manual exit and no trailing, resolving only when price actually
# touches the target or the stop? Reuses exit_strategy_simulation.py's
# already-built "fixed_tp_idealTP"/"baseline" strategies (no new simulator
# needed); this section only adds the direct baseline-vs-fixed-target
# significance/Monte Carlo layer and its own standalone report.
TARGET_OR_STOP_REPORT = REPORTS_OUT / "target_or_stop_report_arabic.md"
