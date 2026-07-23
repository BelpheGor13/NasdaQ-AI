"""Direct follow-up to the target-or-stop finding, with a constraint the
user specified: never hold a position overnight. Each trade is left to
run toward its original stop or idealTP target, exactly like
exit_strategy_simulation.py's fixed_tp_idealTP -- EXCEPT it is also
force-closed at market at a fixed deadline (default 16:00 America/New_York,
the user's choice) on the SAME calendar day it was entered, if neither
level has been touched by then. No extension past that deadline, ever --
this is a same-day-only rule, unlike the other analyses' multi-day
extension windows.

Tie-break when both stop and target fall inside the same 1-minute candle:
target wins (documented, same convention as exit_strategy_simulation.py --
verified elsewhere to affect <1.3% of trades and move the headline number
by low single-digit percent at most).
"""
import numpy as np
import pandas as pd

from src import data_loading

DEFAULT_DEADLINE_HOUR_NY = 16


def _deadline_utc(entry_dt_ny: pd.Timestamp, deadline_hour: int) -> pd.Timestamp:
    deadline_ny = entry_dt_ny.normalize() + pd.Timedelta(hours=deadline_hour)
    if deadline_ny <= entry_dt_ny:
        # entry itself is at/after the deadline hour (doesn't occur in this
        # dataset, but handled rather than silently mis-simulated): force
        # an immediate close at entry.
        deadline_ny = entry_dt_ny
    return deadline_ny.tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)


def simulate_eod_target_or_stop(trades: pd.DataFrame, candles: pd.DataFrame,
                                 deadline_hour_ny: int = DEFAULT_DEADLINE_HOUR_NY) -> pd.DataFrame:
    ts = candles["datetime_utc"].values
    opens_all = candles["open"].values
    highs_all = candles["high"].values
    lows_all = candles["low"].values
    closes_all = candles["close"].values

    entry_arr = trades["entryPrice"].values
    sl_arr = trades["initalSL"].values
    tp_arr = trades["idealTP"].values
    side_arr = trades["side"].values
    orig_r_arr = trades["avgRiskReward"].values
    risk_arr = np.abs(entry_arr - sl_arr)
    risk_arr = np.where(risk_arr == 0, np.nan, risk_arr)

    rows = []
    for i in range(len(trades)):
        entry_dt_ny = pd.Timestamp(trades["dateStart"].iloc[i])
        entry_dt_utc = pd.Timestamp(trades["dateStart_utc"].iloc[i])
        deadline_utc = _deadline_utc(entry_dt_ny, deadline_hour_ny)

        r_i = risk_arr[i]
        trade_id = trades["id"].iloc[i]

        if pd.isna(r_i) or pd.isna(tp_arr[i]):
            rows.append(dict(id=trade_id, eod_exit_r=np.nan, eod_exit_reason="no_data",
                              eod_exit_time_utc=pd.NaT))
            continue

        lo = np.searchsorted(ts, np.datetime64(entry_dt_utc), side="left")
        hi = np.searchsorted(ts, np.datetime64(deadline_utc), side="right")

        is_buy = side_arr[i] == "buy"
        entry, sl, tp = entry_arr[i], sl_arr[i], tp_arr[i]

        if hi <= lo:
            rows.append(dict(id=trade_id, eod_exit_r=np.nan, eod_exit_reason="no_data",
                              eod_exit_time_utc=pd.NaT))
            continue

        highs, lows, closes, times = highs_all[lo:hi], lows_all[lo:hi], closes_all[lo:hi], ts[lo:hi]

        exit_r, exit_reason, exit_time = None, None, None
        for idx in range(len(highs)):
            sl_hit = (lows[idx] <= sl) if is_buy else (highs[idx] >= sl)
            tp_hit = (highs[idx] >= tp) if is_buy else (lows[idx] <= tp)
            if sl_hit or tp_hit:
                if tp_hit:
                    exit_r = (tp - entry) / r_i if is_buy else (entry - tp) / r_i
                    exit_reason = "target"
                else:
                    exit_r = (sl - entry) / r_i if is_buy else (entry - sl) / r_i
                    exit_reason = "stop"
                exit_time = times[idx]
                break

        if exit_r is None:
            # deadline reached without touching either level: force-close
            # at the last available candle's close at/before the deadline.
            exit_price = closes[-1]
            exit_r = (exit_price - entry) / r_i if is_buy else (entry - exit_price) / r_i
            exit_reason = "eod_forced"
            exit_time = times[-1]

        rows.append(dict(id=trade_id, eod_exit_r=float(exit_r), eod_exit_reason=exit_reason,
                          eod_exit_time_utc=pd.Timestamp(exit_time)))

    out = pd.DataFrame(rows)
    meta = trades[["id", "dateStart_utc", "dateEnd_utc", "side", "amount", "avgRiskReward"]].rename(
        columns={"avgRiskReward": "orig_r"})
    meta["risk_price"] = np.abs(trades["entryPrice"] - trades["initalSL"])
    out = out.merge(meta, on="id", how="left")
    out["eod_exit_pnl_usd"] = out["eod_exit_r"] * out["risk_price"] * out["amount"]
    out["orig_pnl_usd"] = out["orig_r"] * out["risk_price"] * out["amount"]
    return out


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = simulate_eod_target_or_stop(trades, candles)

    print(f"trades simulated: {len(sim)}")
    print(sim["eod_exit_reason"].value_counts())
    print()
    print(sim[["orig_r", "eod_exit_r"]].describe())
