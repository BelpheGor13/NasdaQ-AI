"""Head-to-head comparison of three exit policies, rebuilt for every trade
directly from the 1-minute candles, using each trade's own entryPrice,
initalSL and take-profit level.

The three policies:

  ACTUAL      what the trade log says really happened (avgRiskReward).

  LEAVE_ALONE entry -> whichever of (original initalSL, target) price
              touches first. No intervention at all.

  BREAKEVEN   the user's real rule: for any trade whose target sits beyond
              2R, once price trades 2R in favour, the stop is moved to
              entryPrice and the trade is then left to run to its target
              or be stopped out at breakeven. Trades whose target is at or
              inside 2R never trigger the rule and behave like LEAVE_ALONE.

Conventions, identical across all policies so the comparison is fair:
  - A bar that spans both the active stop and the target is charged as a
    STOP. Intrabar ordering is unknowable from OHLC, so ambiguity is always
    resolved against the trade.
  - A gap through a level fills at the bar open (the worse price).
  - The breakeven stop becomes active on the bar AFTER 2R is confirmed, so
    a single bar is never allowed to both arm and trigger the rule.
  - Walking continues up to config.EXIT_STRATEGY_MAX_EXTENSION_MINUTES past
    the trade's own dateEnd_utc; anything still unresolved is reported as
    such rather than scored.

Target source is recorded per trade. maxTP is populated for all 213
winners and is a genuine level; idealTP is the only column present on the
other 576 rows but is known to be rewritten at close (see
idealtp_data_quality_check.py), so those rows carry target_trustworthy=False
and are reported separately instead of being folded into the headline.
"""
import numpy as np
import pandas as pd

from src import config, data_loading, stats_utils

BREAKEVEN_TRIGGER_R = 2.0


def _resolve_target(row_maxtp, row_idealtp):
    if not np.isnan(row_maxtp):
        return row_maxtp, "maxTP", True
    return row_idealtp, "idealTP", False


def median_real_target_r(trades: pd.DataFrame) -> float:
    """Median R:R of the 213 targets we can actually verify (maxTP). Used
    as the uniform target for the unbiased comparison, because the only
    trades carrying a genuine target are the winners -- so any policy
    comparison restricted to them is selected on the outcome and cannot
    be read as a forward-looking result."""
    w = trades[trades["maxTP"].notna()]
    risk = (w["entryPrice"] - w["initalSL"]).abs()
    reward = np.where(w["side"] == "buy", w["maxTP"] - w["entryPrice"], w["entryPrice"] - w["maxTP"])
    return float((reward / risk).median())


def simulate_policies(trades: pd.DataFrame, candles: pd.DataFrame,
                       be_trigger_r: float = BREAKEVEN_TRIGGER_R,
                       uniform_target_r: float = None) -> pd.DataFrame:
    """uniform_target_r: when set, every trade uses a target at this R
    multiple instead of its own (possibly rewritten) file target. This is
    the unbiased mode -- it applies one rule to all 789 trades without
    peeking at which ones won."""
    ts = candles["datetime_utc"].values
    opens_all = candles["open"].values
    highs_all = candles["high"].values
    lows_all = candles["low"].values

    starts = trades["dateStart_utc"].values
    ends = trades["dateEnd_utc"].values
    ext = np.timedelta64(config.EXIT_STRATEGY_MAX_EXTENSION_MINUTES, "m")

    start_idx = np.searchsorted(ts, starts, side="left")
    ext_end_idx = np.minimum(np.searchsorted(ts, ends + ext, side="right"), len(ts))

    rows = []
    for i in range(len(trades)):
        t = trades.iloc[i]
        entry, sl, side = t["entryPrice"], t["initalSL"], t["side"]
        risk = abs(entry - sl)
        lo, ext_hi = start_idx[i], ext_end_idx[i]

        is_buy = side == "buy"
        if uniform_target_r is not None:
            target = entry + uniform_target_r * risk if is_buy else entry - uniform_target_r * risk
            target_source, trustworthy = f"uniform_{uniform_target_r:g}R", True
        else:
            target, target_source, trustworthy = _resolve_target(t["maxTP"], t["idealTP"])
        signed_target = (target - entry) if is_buy else (entry - target)
        target_r = signed_target / risk if risk > 0 else np.nan

        base = dict(
            id=t["id"], dateStart_utc=t["dateStart_utc"], side=side,
            entryPrice=entry, initalSL=sl, target_price=target,
            target_source=target_source, target_trustworthy=trustworthy,
            target_r=target_r, risk_price=risk, amount=t["amount"],
            actual_r=t["avgRiskReward"],
        )

        if ext_hi <= lo or risk == 0 or np.isnan(target) or signed_target <= 0:
            rows.append(dict(base, leave_outcome="unresolved", leave_r=np.nan,
                             be_outcome="unresolved", be_r=np.nan, be_armed=False))
            continue

        be_level = entry + be_trigger_r * risk if is_buy else entry - be_trigger_r * risk
        rule_applies = target_r > be_trigger_r  # user's rule only for >1:2 setups

        leave_outcome, leave_r = None, np.nan
        be_outcome, be_r, be_armed, be_active = None, np.nan, False, False

        for k in range(lo, ext_hi):
            o, h, l = opens_all[k], highs_all[k], lows_all[k]

            hit_sl = (l <= sl) if is_buy else (h >= sl)
            hit_tp = (h >= target) if is_buy else (l <= target)
            hit_be_stop = (l <= entry) if is_buy else (h >= entry)

            # ---- LEAVE_ALONE ----
            if leave_outcome is None:
                if hit_sl:
                    fill = sl if (o >= sl if is_buy else o <= sl) else o
                    leave_outcome = "stop"
                elif hit_tp:
                    fill = target if (o <= target if is_buy else o >= target) else o
                    leave_outcome = "target"
                if leave_outcome is not None:
                    leave_r = (fill - entry) / risk if is_buy else (entry - fill) / risk

            # ---- BREAKEVEN (user's rule) ----
            if be_outcome is None:
                active_stop = entry if be_active else sl
                hit_active = hit_be_stop if be_active else hit_sl
                if hit_active:
                    fill = active_stop if (o >= active_stop if is_buy else o <= active_stop) else o
                    be_outcome = "breakeven_stop" if be_active else "stop"
                elif hit_tp:
                    fill = target if (o <= target if is_buy else o >= target) else o
                    be_outcome = "target"
                if be_outcome is not None:
                    be_r = (fill - entry) / risk if is_buy else (entry - fill) / risk
                elif rule_applies and not be_active:
                    # arm on this bar, active from the next one
                    reached_be_trigger = (h >= be_level) if is_buy else (l <= be_level)
                    if reached_be_trigger:
                        be_active, be_armed = True, True

            if leave_outcome is not None and be_outcome is not None:
                break

        rows.append(dict(base,
                         leave_outcome=leave_outcome or "unresolved",
                         leave_r=leave_r,
                         be_outcome=be_outcome or "unresolved",
                         be_r=be_r, be_armed=be_armed))

    out = pd.DataFrame(rows)
    unit = out["risk_price"] * out["amount"]
    for col in ("actual", "leave", "be"):
        out[f"{col}_usd"] = out[f"{col}_r"] * unit
    out["actual_group"] = np.select(
        [out["actual_r"] == -1.0, out["actual_r"] == 0.0, out["actual_r"] > 0],
        ["full_loss", "breakeven", "winner"], default="partial")
    return out.sort_values("dateStart_utc").reset_index(drop=True)


def max_drawdown_r(series: np.ndarray) -> float:
    if len(series) == 0:
        return np.nan
    equity = np.cumsum(series)
    return float((np.maximum.accumulate(equity) - equity).max())


def headline_table(sim: pd.DataFrame, trustworthy_only: bool = False) -> pd.DataFrame:
    d = sim[sim["target_trustworthy"]] if trustworthy_only else sim
    d = d[(d["leave_outcome"] != "unresolved") & (d["be_outcome"] != "unresolved")]

    rows = []
    for label, r_col, usd_col in [("ACTUAL (what you did)", "actual_r", "actual_usd"),
                                   ("LEAVE ALONE", "leave_r", "leave_usd"),
                                   ("BREAKEVEN at 2R (your rule)", "be_r", "be_usd")]:
        r = d[r_col].dropna().values
        rows.append({
            "policy": label, "n": len(r),
            "win_rate": float((r > 0).mean()),
            "expectancy_r": stats_utils.expectancy(r),
            "profit_factor": stats_utils.profit_factor(r),
            "total_r": float(r.sum()),
            "total_usd": float(d[usd_col].sum()),
            "max_drawdown_r": max_drawdown_r(r),
        })
    return pd.DataFrame(rows)


def outcome_counts(sim: pd.DataFrame) -> pd.DataFrame:
    d = sim[(sim["leave_outcome"] != "unresolved") & (sim["be_outcome"] != "unresolved")]
    rows = []
    for grp, g in d.groupby("actual_group"):
        rows.append({
            "actual_group": grp, "n": len(g),
            "leave_hit_target": int((g["leave_outcome"] == "target").sum()),
            "leave_hit_stop": int((g["leave_outcome"] == "stop").sum()),
            "be_rule_armed": int(g["be_armed"].sum()),
            "be_stopped_at_breakeven": int((g["be_outcome"] == "breakeven_stop").sum()),
            "be_hit_target": int((g["be_outcome"] == "target").sum()),
            "be_hit_full_stop": int((g["be_outcome"] == "stop").sum()),
            "actual_r": float(g["actual_r"].sum()),
            "leave_r": float(g["leave_r"].sum()),
            "be_r": float(g["be_r"].sum()),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    sim = simulate_policies(trades, candles)

    print("=== ALL 789 TRADES (includes untrustworthy idealTP targets) ===")
    print(headline_table(sim).to_string(index=False))

    med = median_real_target_r(trades)
    print(f"\n=== UNBIASED: uniform target at {med:.2f}R (median of the 213 verifiable targets) ===")
    sim_u = simulate_policies(trades, candles, uniform_target_r=med)
    print(headline_table(sim_u).to_string(index=False))

    print("\n=== outcome counts, uniform-target run, by what the trade actually did ===")
    print(outcome_counts(sim_u).to_string(index=False))
