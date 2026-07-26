"""Single source of truth for "what was this trade's real target price,"
used by every other module in this project. Do not reimplement this
logic elsewhere -- import from here.

Column semantics, confirmed directly by the user and cross-validated
against this project's own independent candle-based MFE calculation (see
idealtp_data_quality_check.py for the verification numbers):

  maxTP    the REAL original target. Present only for the 213 trades that
           closed with a realized profit (avgRiskReward > 0); null for
           the other 576.
  idealTP  NOT a target. It is the price the trade reached (its MFE)
           BEFORE it eventually hit initalSL. Never used as a target
           anywhere in this project as of this module's introduction.

For the 576 trades without a real maxTP, the target is unknown -- the
user's own rule is to assume AT LEAST double the stop-loss distance (2R)
as a floor/stand-in, since a trade that never reached its real target
gives no way to recover what that target actually was.
"""
import numpy as np
import pandas as pd

MIN_TARGET_R = 2.0


def resolve_target_price(entry: float, sl: float, side: str, maxtp) -> tuple:
    """Single-trade version. Returns (target_price, is_real_target)."""
    if maxtp is not None and not (isinstance(maxtp, float) and np.isnan(maxtp)):
        return float(maxtp), True
    risk = abs(entry - sl)
    target = entry + MIN_TARGET_R * risk if side == "buy" else entry - MIN_TARGET_R * risk
    return target, False


def resolve_target_series(trades: pd.DataFrame, min_target_r: float = MIN_TARGET_R) -> pd.DataFrame:
    """Vectorized version. Returns a copy of trades with two added columns:
    target_price (float) and target_is_real (bool)."""
    out = trades.copy()
    entry = out["entryPrice"].values
    sl = out["initalSL"].values
    side = out["side"].values
    maxtp = out["maxTP"].values
    risk = np.abs(entry - sl)

    floor_target = np.where(side == "buy", entry + min_target_r * risk, entry - min_target_r * risk)
    is_real = ~np.isnan(maxtp)
    out["target_price"] = np.where(is_real, maxtp, floor_target)
    out["target_is_real"] = is_real
    return out


def target_r_multiple(trades_with_target: pd.DataFrame) -> np.ndarray:
    """The R:R implied by target_price -- min_target_r exactly for the
    576 floor-assumed rows, and whatever the real ratio is for the 213
    real-target rows."""
    entry = trades_with_target["entryPrice"].values
    sl = trades_with_target["initalSL"].values
    side = trades_with_target["side"].values
    target = trades_with_target["target_price"].values
    risk = np.abs(entry - sl)
    reward = np.where(side == "buy", target - entry, entry - target)
    return reward / risk


if __name__ == "__main__":
    from src import data_loading
    trades = data_loading.load_trades()
    out = resolve_target_series(trades)
    print("target_is_real value counts:")
    print(out["target_is_real"].value_counts())
    print()
    print("target_r_multiple describe:")
    print(pd.Series(target_r_multiple(out)).describe())
