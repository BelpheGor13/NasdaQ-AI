"""No-stop-loss scenario (test-only, hypothetical): reconstructs, bar-by-bar
and strictly without look-ahead, what would have happened to each of the
789 trades if the original initalSL had never been active at all -- the
trade simply rides on real candles until a documented deadline.

Never touches analytics_1.csv or any original-strategy logic. This module
only produces one hypothetical, UNPROTECTED alternative exit per trade.

Rule used (deliberately the simplest defensible one -- no target, no
partial exits, nothing that could be read as "the market was aiming
somewhere"): hold from dateStart_utc for up to
config.NO_STOP_MAX_EXTENSION_MINUTES (30 days) of real candles, with no
exit condition at all, and mark-to-market at the close of whichever candle
the window ends on. Two things are tracked throughout the ride:
  - final_r: the R-multiple at the window's end (the "did it end up a
    win or a loss" the user asked about).
  - worst_r_reached: the deepest adverse excursion touched at ANY point
    during the ride, even if the trade recovered afterward -- this is the
    number that answers "how bad could it have gotten," independent of
    where it ended up.

risk (the R-multiple unit) is still computed from the ORIGINAL
entryPrice/initalSL distance -- not because the stop is active, but
because it is the same fixed risk unit used everywhere else in this
project, so results stay comparable in R.

A trade whose 30-day window runs past the end of the available candle
data is flagged data_censored=True: we genuinely don't know what would
have happened after that point, and this is disclosed rather than treated
as a resolved outcome.
"""
import numpy as np
import pandas as pd

from src import config, data_loading


def _walk_window(entry: float, is_buy: bool, opens, highs, lows, closes):
    n = len(opens)
    peak = entry
    trough = entry
    for idx in range(n):
        h, l = highs[idx], lows[idx]
        if is_buy:
            peak = max(peak, h)
            trough = min(trough, l)
        else:
            peak = min(peak, l)
            trough = max(trough, h)
    return peak, trough, closes[n - 1]


def simulate_no_stop(trades: pd.DataFrame, candles: pd.DataFrame,
                      max_extension_minutes: int = config.NO_STOP_MAX_EXTENSION_MINUTES) -> pd.DataFrame:
    ts = candles["datetime_utc"].values
    opens_all = candles["open"].values
    highs_all = candles["high"].values
    lows_all = candles["low"].values
    closes_all = candles["close"].values
    data_end = ts[-1]

    starts = trades["dateStart_utc"].values
    ext = np.timedelta64(max_extension_minutes, "m")
    deadlines = starts + ext

    start_idx = np.searchsorted(ts, starts, side="left")
    deadline_idx = np.minimum(np.searchsorted(ts, deadlines, side="right"), len(ts))

    entry_arr = trades["entryPrice"].values
    sl_arr = trades["initalSL"].values
    side_arr = trades["side"].values
    orig_r_arr = trades["avgRiskReward"].values
    risk_arr = np.abs(entry_arr - sl_arr)
    risk_arr = np.where(risk_arr == 0, np.nan, risk_arr)

    rows = []
    for i in range(len(trades)):
        lo, hi = start_idx[i], deadline_idx[i]
        r_i = risk_arr[i]
        is_buy = side_arr[i] == "buy"
        trade_id = trades["id"].iloc[i]

        data_censored = bool(deadlines[i] > data_end)

        if hi <= lo or pd.isna(r_i):
            rows.append(dict(
                id=trade_id, no_sl_final_r=np.nan, no_sl_worst_r_reached=np.nan,
                no_sl_best_r_reached=np.nan, n_candles_walked=0,
                data_censored=data_censored, resolvable=False,
            ))
            continue

        peak, trough, final_price = _walk_window(
            entry_arr[i], is_buy, opens_all[lo:hi], highs_all[lo:hi], lows_all[lo:hi], closes_all[lo:hi]
        )

        if is_buy:
            final_r = (final_price - entry_arr[i]) / r_i
            worst_r = (trough - entry_arr[i]) / r_i
            best_r = (peak - entry_arr[i]) / r_i
        else:
            # _walk_window already tracks peak=favorable extreme,
            # trough=adverse extreme for sells too (see its is_buy branch),
            # so the R conversion mirrors the buy case with entry/price
            # swapped -- no separate min/max reconciliation needed.
            final_r = (entry_arr[i] - final_price) / r_i
            worst_r = (entry_arr[i] - trough) / r_i
            best_r = (entry_arr[i] - peak) / r_i

        rows.append(dict(
            id=trade_id, no_sl_final_r=final_r, no_sl_worst_r_reached=worst_r,
            no_sl_best_r_reached=best_r, n_candles_walked=int(hi - lo),
            data_censored=data_censored, resolvable=True,
        ))

    out = pd.DataFrame(rows)
    meta = trades[["id", "dateStart_utc", "dateEnd_utc", "side", "amount", "avgRiskReward"]].rename(
        columns={"avgRiskReward": "orig_r"})
    meta["risk_price"] = np.abs(trades["entryPrice"] - trades["initalSL"])
    out = out.merge(meta, on="id", how="left")
    out["no_sl_final_pnl_usd"] = out["no_sl_final_r"] * out["risk_price"] * out["amount"]
    out["orig_pnl_usd"] = out["orig_r"] * out["risk_price"] * out["amount"]
    out["catastrophic"] = out["no_sl_worst_r_reached"] <= config.NO_STOP_CATASTROPHIC_R_THRESHOLD
    return out


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = simulate_no_stop(trades, candles)

    print(f"trades simulated: {len(sim)}")
    print(f"resolvable (had candle data): {sim['resolvable'].sum()}")
    print(f"data-censored (30-day window ran past available data): {sim['data_censored'].sum()}")
    print(f"catastrophic (worst_r_reached <= {config.NO_STOP_CATASTROPHIC_R_THRESHOLD}R): {sim['catastrophic'].sum()}")
    print()
    print(sim[["orig_r", "no_sl_final_r", "no_sl_worst_r_reached", "no_sl_best_r_reached"]].describe())
