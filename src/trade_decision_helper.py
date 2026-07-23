"""Decision-support tool for a SINGLE new trade -- not a new discovery, just
a mechanical application of two things this project already established
with real statistical rigor:

  1. entry check: does this trade match the one composite pattern that
     survived FDR correction on TWO independent targets (realized R and
     maxRiskReward potential) -- day_of_week=Wednesday with pre-entry
     15-minute momentum near 50% up-candles? (see
     outputs/reports/final_report_arabic.md). Historically n=37,
     expectancy=-0.595R.
  2. exit check: the target-or-stop rule (outputs/reports/
     target_or_stop_report_arabic.md) -- once in the trade, don't close
     manually; only the original stop-loss or the target should close it
     (PF 1.18 -> 2.42, p~1e-26, Monte Carlo 100% robust).

This tool does NOT place, modify, or close any real trade -- it only
reads candle data and reports what the two rules above say. Actual
execution stays entirely with the user, at their broker, by hand.

Works two ways:
  - Backtest mode: check a trade that's already in this project's
    2020-2024 dataset (pass no candles argument).
  - Live/paper mode: pass your own OHLCV DataFrame (columns: datetime_utc,
    open, high, low, close) covering the period around your new trade --
    exported from your own broker/data source in the same shape as
    nasdaq_m1_2020_2024.parquet. This module never fetches live data
    itself.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from src import data_loading

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    # Windows consoles default to the system codepage (cp1252 etc.), which
    # can't encode the Arabic recommendation text -- force UTF-8 so the CLI
    # works out of the box instead of crashing on print().
    sys.stdout.reconfigure(encoding="utf-8")

# The one pattern that survived FDR on both r_multiple and maxRiskReward
# (see final_report_arabic.md). The momentum bin boundary is NOT hardcoded
# as a rounded threshold -- pd.qcut's tertile edges don't land on "clean"
# numbers (e.g. 0.467 displayed is not exactly 7/15), so a hand-copied
# threshold silently mismatches some trades right at the edge. Instead the
# exact historical edges are recomputed once from the training data (see
# _avoid_pattern_bin_edges) and reused for every check.
AVOID_DAY_OF_WEEK = 3  # Wednesday (1=Mon..5=Fri, NY calendar)
AVOID_PATTERN_STATS = {"n": 37, "expectancy_r": -0.595, "win_rate": 0.081}

_bin_edges_cache = None


def _avoid_pattern_bin_edges() -> tuple:
    """Recomputes the exact tertile edges for pre_entry_pct_up_candles_15m
    on the same 2020-2024 dataset the pattern was discovered on.

    Must use qcut's retbins=True output, NOT bins.cat.categories[i].right --
    pandas rounds Interval category boundaries to `precision` (default 3)
    significant digits for display, e.g. 8/15=0.5333... is stored in the
    category as 0.533. Reclassifying a new value against that ROUNDED
    edge silently misclassifies anything between 0.533 and 0.5333... (a
    real bug caught here: 0.5333333 tested as > 0.533 and fell outside a
    bin it truly belongs in). retbins gives the true, unrounded edges
    actually used for assignment.
    """
    global _bin_edges_cache
    if _bin_edges_cache is not None:
        return _bin_edges_cache

    from src import feature_engineering
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    feats = feature_engineering.build_features(trades, candles)
    _, edges = pd.qcut(feats["pre_entry_pct_up_candles_15m"], 3, duplicates="drop", retbins=True)
    _bin_edges_cache = (float(edges[1]), float(edges[2]))  # middle tertile's [lo, hi]
    return _bin_edges_cache


def _pre_entry_pct_up_candles_15m(candles: pd.DataFrame, entry_dt_utc: pd.Timestamp) -> float:
    ts = candles["datetime_utc"].values
    closes = candles["close"].values
    opens = candles["open"].values

    end_idx = np.searchsorted(ts, np.datetime64(entry_dt_utc), side="left")
    lo = end_idx - 15
    if lo < 0 or end_idx <= lo:
        return np.nan
    return float((closes[lo:end_idx] > opens[lo:end_idx]).mean())


def check_entry(entry_datetime_ny: pd.Timestamp, candles: pd.DataFrame = None) -> dict:
    """entry_datetime_ny: naive local America/New_York timestamp, same
    convention as dateStart in analytics_1.csv.
    """
    entry_dt_ny = pd.Timestamp(entry_datetime_ny)
    entry_dt_utc = entry_dt_ny.tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)
    day_of_week = entry_dt_ny.weekday() + 1  # Monday=1

    candles = candles if candles is not None else data_loading.load_candles()
    momentum = _pre_entry_pct_up_candles_15m(candles, entry_dt_utc)
    lo_edge, hi_edge = _avoid_pattern_bin_edges()

    matches_avoid_pattern = bool(
        day_of_week == AVOID_DAY_OF_WEEK
        and pd.notna(momentum)
        and lo_edge < momentum <= hi_edge
    )

    return {
        "entry_datetime_ny": str(entry_dt_ny),
        "day_of_week": day_of_week,
        "pre_entry_pct_up_candles_15m": momentum,
        "matches_avoid_pattern": matches_avoid_pattern,
        "avoid_pattern_historical_stats": AVOID_PATTERN_STATS if matches_avoid_pattern else None,
        "recommendation": (
            "تحذير: هذا الدخول يطابق النمط الوحيد المؤكَّد إحصائيًا (يوم أربعاء + زخم ~50%) — "
            f"تاريخيًا n={AVOID_PATTERN_STATS['n']}, عائد متوقع={AVOID_PATTERN_STATS['expectancy_r']}R. "
            "فكّر مرتين قبل الدخول."
            if matches_avoid_pattern else
            "لا يطابق النمط المعروف الوحيد الذي يستحق التجنّب — لا يعني هذا أن الصفقة جيدة، فقط أنها لا "
            "تطابق التحذير المحدَّد."
        ),
    }


def track_position(entry_price: float, initial_sl: float, target_price: float, side: str,
                    entry_datetime_ny: pd.Timestamp, candles: pd.DataFrame = None,
                    as_of: pd.Timestamp = None) -> dict:
    """Applies the target-or-stop rule mechanically: walks real candles from
    entry to as_of (default: latest candle available) and reports whether
    the ORIGINAL stop or the target was touched first -- exactly the rule
    validated in target_or_stop_report_arabic.md. Never tells you to close
    early; that discretionary habit is the thing the report found costly.
    """
    entry_dt_ny = pd.Timestamp(entry_datetime_ny)
    entry_dt_utc = entry_dt_ny.tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)

    candles = candles if candles is not None else data_loading.load_candles()
    is_buy = side == "buy"

    ts = candles["datetime_utc"].values
    start_idx = np.searchsorted(ts, np.datetime64(entry_dt_utc), side="left")
    if as_of is not None:
        as_of_utc = pd.Timestamp(as_of).tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)
        end_idx = np.searchsorted(ts, np.datetime64(as_of_utc), side="right")
    else:
        end_idx = len(ts)

    if end_idx <= start_idx:
        return {"status": "no_candle_data_yet", "recommendation": "لا توجد بيانات شموع كافية بعد الدخول."}

    highs = candles["high"].values[start_idx:end_idx]
    lows = candles["low"].values[start_idx:end_idx]
    times = candles["datetime_utc"].values[start_idx:end_idx]

    for i in range(len(highs)):
        sl_hit = (lows[i] <= initial_sl) if is_buy else (highs[i] >= initial_sl)
        tp_hit = (highs[i] >= target_price) if is_buy else (lows[i] <= target_price)
        if sl_hit or tp_hit:
            outcome = "target" if tp_hit else "stop"
            return {
                "status": f"closed_at_{outcome}",
                "closed_at_utc": str(pd.Timestamp(times[i])),
                "recommendation": (
                    f"القاعدة تقول: أُغلقت الصفقة عند {'الهدف' if outcome == 'target' else 'الوقف'} "
                    "فعليًا — هذا ما كان يجب أن يحدث ميكانيكيًا، دون تدخّل يدوي."
                ),
            }

    return {
        "status": "still_open",
        "recommendation": (
            "لم يُلمَس الهدف أو الوقف بعد — القاعدة المؤكَّدة إحصائيًا تقول: **لا تُغلق يدويًا**، "
            "اترك الصفقة تصل لأحدهما فعليًا (راجع target_or_stop_report_arabic.md قبل أي قرار مخالف)."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision helper for a single new NAS100 trade")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_entry = sub.add_parser("check-entry")
    p_entry.add_argument("--datetime", required=True, help="Entry datetime, America/New_York, e.g. '2025-01-15 09:45:00'")

    p_track = sub.add_parser("track")
    p_track.add_argument("--entry", type=float, required=True)
    p_track.add_argument("--sl", type=float, required=True)
    p_track.add_argument("--target", type=float, required=True)
    p_track.add_argument("--side", choices=["buy", "sell"], required=True)
    p_track.add_argument("--datetime", required=True)
    p_track.add_argument("--as-of", default=None)

    args = parser.parse_args()

    if args.mode == "check-entry":
        result = check_entry(args.datetime)
    else:
        result = track_position(args.entry, args.sl, args.target, args.side, args.datetime, as_of=args.as_of)

    for k, v in result.items():
        print(f"{k}: {v}")
