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
