"""Stage 1: load and validate trades + candle data. No downstream stage should
read the raw files directly -- everything flows through here so the
timezone/schema assumptions are enforced in exactly one place.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass
class ProfileReport:
    trades_shape: tuple
    candles_shape: tuple
    trades_date_range: tuple
    candles_date_range: tuple
    trades_covered_by_candles: bool
    null_counts: dict
    warnings: list


def load_trades() -> pd.DataFrame:
    df = pd.read_csv(config.TRADES_CSV)

    for col in ("dateStart", "dateEnd"):
        naive = pd.to_datetime(df[col], format="%Y/%m/%d %H:%M:%S")
        localized = naive.dt.tz_localize(
            config.TRADE_TZ, ambiguous="infer", nonexistent="shift_forward"
        )
        df[col + "_utc"] = localized.dt.tz_convert("UTC").dt.tz_localize(None)

    df = df.sort_values("dateStart_utc").reset_index(drop=True)

    df["is_win"] = df["rPnL"] > 0
    # avgRiskReward is the realized R-multiple (corr with rPnL = 0.999,
    # implying a ~$500 fixed risk unit); use it directly as ground truth
    # rather than re-deriving R from rPnL/amount.
    df["r_multiple"] = df["avgRiskReward"]

    return df


def load_candles() -> pd.DataFrame:
    df = pd.read_parquet(config.CANDLES_PARQUET)
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.rename(columns={"datetime": "datetime_utc"})
    return df


def profile(trades: pd.DataFrame, candles: pd.DataFrame) -> ProfileReport:
    warnings = []

    t_min, t_max = trades["dateStart_utc"].min(), trades["dateEnd_utc"].max()
    c_min, c_max = candles["datetime_utc"].min(), candles["datetime_utc"].max()

    covered = bool(t_min >= c_min and t_max <= c_max)
    if not covered:
        warnings.append(
            f"Trade date range [{t_min}, {t_max}] not fully covered by "
            f"candle range [{c_min}, {c_max}]"
        )

    gaps = candles["datetime_utc"].diff().dropna()
    expected = pd.Timedelta(minutes=1)
    n_gaps = int((gaps > expected).sum())
    if n_gaps:
        warnings.append(f"{n_gaps} gaps > 1 minute found in candle series (market closures expected)")

    dup_candles = int(candles["datetime_utc"].duplicated().sum())
    if dup_candles:
        warnings.append(f"{dup_candles} duplicate candle timestamps found")

    null_counts = {
        "trades": trades.isnull().sum().to_dict(),
        "candles": candles.isnull().sum().to_dict(),
    }

    return ProfileReport(
        trades_shape=trades.shape,
        candles_shape=candles.shape,
        trades_date_range=(t_min, t_max),
        candles_date_range=(c_min, c_max),
        trades_covered_by_candles=covered,
        null_counts=null_counts,
        warnings=warnings,
    )


def print_profile(report: ProfileReport) -> None:
    print("=== Data profile ===")
    print(f"trades: {report.trades_shape}, range {report.trades_date_range}")
    print(f"candles: {report.candles_shape}, range {report.candles_date_range}")
    print(f"trades fully covered by candles: {report.trades_covered_by_candles}")
    print("null counts (trades, nonzero only):")
    for k, v in report.null_counts["trades"].items():
        if v:
            print(f"  {k}: {v}")
    if report.warnings:
        print("warnings:")
        for w in report.warnings:
            print(f"  - {w}")
    else:
        print("no warnings")


if __name__ == "__main__":
    trades = load_trades()
    candles = load_candles()
    report = profile(trades, candles)
    print_profile(report)
