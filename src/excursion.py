"""Stage 2: merge each trade with its candle window and reconstruct the full
MFE/MAE excursion path between dateStart_utc and dateEnd_utc.

All of this is descriptive research about a trade's own realized path
(it uses candles *during* the trade), never a pre-entry signal.
"""
import numpy as np
import pandas as pd

from src import data_loading


def _risk_per_unit(row) -> float:
    return abs(row["entryPrice"] - row["initalSL"])


def reconstruct_excursions(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Returns trades with MFE/MAE (price and R) and the excursion path
    (list of dicts) for each trade, computed strictly within [dateStart_utc, dateEnd_utc].
    """
    ts = candles["datetime_utc"].values
    highs = candles["high"].values
    lows = candles["low"].values
    closes = candles["close"].values

    starts = trades["dateStart_utc"].values
    ends = trades["dateEnd_utc"].values

    start_idx = np.searchsorted(ts, starts, side="left")
    end_idx = np.searchsorted(ts, ends, side="right")

    mfe_price = np.full(len(trades), np.nan)
    mae_price = np.full(len(trades), np.nan)
    mfe_r = np.full(len(trades), np.nan)
    mae_r = np.full(len(trades), np.nan)
    n_candles_in_window = np.zeros(len(trades), dtype=int)
    mfe_time_frac = np.full(len(trades), np.nan)  # fraction of trade duration to reach MFE
    mae_time_frac = np.full(len(trades), np.nan)

    entry = trades["entryPrice"].values
    sl = trades["initalSL"].values
    side = trades["side"].values
    risk = np.abs(entry - sl)
    risk = np.where(risk == 0, np.nan, risk)

    for i in range(len(trades)):
        lo, hi = start_idx[i], end_idx[i]
        n_candles_in_window[i] = hi - lo
        if hi <= lo:
            continue  # trade window shorter than 1 candle; no in-window data

        window_high = highs[lo:hi]
        window_low = lows[lo:hi]

        if side[i] == "buy":
            best_price = window_high.max()
            worst_price = window_low.min()
            best_pos = window_high.argmax()
            worst_pos = window_low.argmin()
            mfe_price[i] = best_price - entry[i]
            mae_price[i] = entry[i] - worst_price
        else:  # sell
            best_price = window_low.min()
            worst_price = window_high.max()
            best_pos = window_low.argmin()
            worst_pos = window_high.argmax()
            mfe_price[i] = entry[i] - best_price
            mae_price[i] = worst_price - entry[i]

        mfe_r[i] = mfe_price[i] / risk[i] if not np.isnan(risk[i]) else np.nan
        mae_r[i] = mae_price[i] / risk[i] if not np.isnan(risk[i]) else np.nan

        n = hi - lo
        mfe_time_frac[i] = best_pos / max(n - 1, 1)
        mae_time_frac[i] = worst_pos / max(n - 1, 1)

    out = trades.copy()
    out["mfe_price"] = mfe_price
    out["mae_price"] = mae_price
    out["mfe_r"] = mfe_r
    out["mae_r"] = mae_r
    out["n_candles_in_window"] = n_candles_in_window
    out["mfe_time_frac"] = mfe_time_frac
    out["mae_time_frac"] = mae_time_frac
    out["risk_price"] = risk

    return out


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    result = reconstruct_excursions(trades, candles)

    zero_window = (result["n_candles_in_window"] == 0).sum()
    print(f"trades with zero in-window candles: {zero_window} / {len(result)}")
    print(result[["mfe_r", "mae_r", "avgRiskReward", "mfe_time_frac", "mae_time_frac"]].describe())

    corr = result["mfe_r"].corr(result["maxRiskReward"])
    print(f"corr(mfe_r, maxRiskReward) = {corr:.3f}  (prior context: idealTP-derived R correlates at r=0.51)")
