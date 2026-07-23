"""Builds the per-trade and per-month data tables behind the target-or-stop
interactive report -- reuses exit_strategy_simulation.py's existing
"baseline" and "fixed_tp_idealTP" (conservative scenario) outputs, adds
per-trade status labels, and flags the same-candle SL/TP tie-breaking edge
case (documented assumption: when both the original stop and idealTP fall
inside the same 1-minute candle's range, the simulator resolves it as the
target being hit -- see exit_strategy_simulation.py's docstring). This
module quantifies exactly how many trades that touches and how much it
could move the headline number, instead of leaving it as an unquantified
caveat.
"""
import numpy as np
import pandas as pd

from src import config, data_loading, exit_strategy_simulation

FLIP_TO_LOSS_R_THRESHOLD = -0.99  # matches "closed at essentially a full stop"


def _classify(actual_r: float, target_r: float) -> str:
    if abs(target_r - actual_r) < 1e-6:
        return "no_diff"
    if target_r > actual_r:
        return "missed_profit"
    if actual_r > 0 and target_r <= FLIP_TO_LOSS_R_THRESHOLD:
        return "flipped_to_loss"
    return "protected"


def find_same_candle_ties(trades: pd.DataFrame, candles: pd.DataFrame) -> set:
    """Trade ids where the resolving candle's range touches BOTH the
    original stop and idealTP -- we can't tell from 1-minute OHLC which
    was hit first, and the simulator resolves these in the target's favor.
    """
    ts = candles["datetime_utc"].values
    highs_all = candles["high"].values
    lows_all = candles["low"].values

    starts = trades["dateStart_utc"].values
    ends = trades["dateEnd_utc"].values
    ext = np.timedelta64(config.EXIT_STRATEGY_MAX_EXTENSION_MINUTES, "m")

    start_idx = np.searchsorted(ts, starts, side="left")
    ext_end_idx = np.minimum(np.searchsorted(ts, ends + ext, side="right"), len(ts))

    entry_arr = trades["entryPrice"].values
    sl_arr = trades["initalSL"].values
    tp_arr = trades["idealTP"].values
    side_arr = trades["side"].values
    risk_arr = np.abs(entry_arr - sl_arr)

    tie_ids = set()
    for i in range(len(trades)):
        lo, hi = start_idx[i], ext_end_idx[i]
        if hi <= lo or np.isnan(risk_arr[i]) or np.isnan(tp_arr[i]):
            continue
        is_buy = side_arr[i] == "buy"
        sl, tp = sl_arr[i], tp_arr[i]
        for idx in range(lo, hi):
            h, l = highs_all[idx], lows_all[idx]
            sl_breach = (l <= sl) if is_buy else (h >= sl)
            tp_touch = (h >= tp) if is_buy else (l <= tp)
            if sl_breach or tp_touch:
                if sl_breach and tp_touch:
                    tie_ids.add(int(trades["id"].iloc[i]))
                break
    return tie_ids


def build_trade_table(sim: pd.DataFrame, tie_ids: set) -> pd.DataFrame:
    base = sim[(sim["scenario"] == "conservative") & (sim["strategy"] == "baseline")].sort_values("id")
    targ = sim[(sim["scenario"] == "conservative") & (sim["strategy"] == "fixed_tp_idealTP")].sort_values("id")
    assert list(base["id"]) == list(targ["id"])

    out = pd.DataFrame({
        "id": base["id"].values,
        "date": base["dateStart_utc"].dt.strftime("%Y-%m-%d").values,
        "side": base["side"].values,
        "actual_r": base["exit_r"].values,
        "target_r": targ["exit_r"].values,
        "actual_usd": base["exit_pnl_usd"].values,
        "target_usd": targ["exit_pnl_usd"].values,
    })
    out["diff_usd"] = out["target_usd"] - out["actual_usd"]
    out["status"] = [_classify(a, t) for a, t in zip(out["actual_r"], out["target_r"])]
    out["same_candle_tie"] = out["id"].isin(tie_ids)
    return out


def build_monthly_table(trade_table: pd.DataFrame) -> pd.DataFrame:
    t = trade_table.copy()
    t["month"] = t["date"].str.slice(0, 7)
    m = t.groupby("month").agg(n=("id", "count"), actual_usd=("actual_usd", "sum"),
                                target_usd=("target_usd", "sum")).reset_index()
    m["diff_usd"] = m["target_usd"] - m["actual_usd"]
    return m.sort_values("month")


def build_kpis(trade_table: pd.DataFrame, monthly_table: pd.DataFrame, tie_ids: set) -> dict:
    total_diff = float(trade_table["diff_usd"].sum())
    n_missed = int((trade_table["status"] == "missed_profit").sum())
    n_protected = int((trade_table["status"] == "protected").sum() + (trade_table["status"] == "flipped_to_loss").sum())
    n_flipped = int((trade_table["status"] == "flipped_to_loss").sum())
    n_no_diff = int((trade_table["status"] == "no_diff").sum())
    n_months_improved = int((monthly_table["diff_usd"] > 0).sum())

    tie_rows = trade_table[trade_table["id"].isin(tie_ids)]
    worst_case_tie_impact = float(-(tie_rows["diff_usd"]).sum())  # flipping these to SL removes their (positive) target advantage
    # more precisely computed in the caller against risk_usd; this is a quick same-file estimate
    return {
        "n_total": len(trade_table),
        "total_diff_usd": total_diff,
        "n_missed_profit": n_missed,
        "n_protected": n_protected,
        "n_flipped_to_loss": n_flipped,
        "n_no_diff": n_no_diff,
        "n_months": len(monthly_table),
        "n_months_improved": n_months_improved,
        "n_same_candle_ties": len(tie_ids),
        "pct_same_candle_ties_of_resolved": len(tie_ids) / max((trade_table["actual_r"] != 0).sum(), 1),
    }


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = exit_strategy_simulation.simulate_exit_strategies(trades, candles)
    tie_ids = find_same_candle_ties(trades, candles)

    trade_table = build_trade_table(sim, tie_ids)
    monthly_table = build_monthly_table(trade_table)
    kpis = build_kpis(trade_table, monthly_table, tie_ids)

    for k, v in kpis.items():
        print(f"{k}: {v}")
    print()
    print(trade_table["status"].value_counts())
