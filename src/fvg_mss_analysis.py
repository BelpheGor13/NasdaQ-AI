"""ICT Fair Value Gap (FVG) + Market Structure Shift (MSS) entry
optimization (test-only; no original trade data modified).

Definitions used, taken directly from the two sources the user provided
(fluxcharts.com/articles/fair-value-gaps-fvg-explained and
innercircletrader.net/tutorials/ict-market-structure-shift), not
reinvented:

Swing high/low: strict 3-candle fractal. Candle i is a swing high if
high[i] > high[i-1] and high[i] > high[i+1] (mirror for swing low).

Market Structure Shift (MSS): "A wick poke past the swing is not an MSS --
wait for the body close past the swing extreme." A bullish MSS fires the
first time a candle's CLOSE trades above the most recent (not yet
consumed) swing high; a bearish MSS fires the first time a close trades
below the most recent swing low.

Fair Value Gap (FVG): a 3-candle imbalance. Bullish: candle1.high <
candle3.low, zone = [candle1.high, candle3.low]. Bearish: candle1.low >
candle3.high, zone = [candle3.high, candle1.low]. Invalidated if a later
candle CLOSES beyond the zone's far boundary before being retested.

MSS-to-FVG flow (per innercircletrader.net): "Wait for the retrace back
into that PD Array... Execute the trade on the retest," not on the
displacement itself. So for each trade we look for the most recent MSS in
the trade's own direction within a bounded lookback, find the first FVG
its displacement leg left behind, and find the first time price re-tests
the NEAR edge of that FVG (the edge closest to the continuation --
confirmed with the user: first-touch entry, not the 50% equilibrium)
BEFORE the trade's actual entry timestamp. If found, that touch's price
and time become the hypothetical entry.

Stop-loss (corrected per the user): the ORIGINAL initalSL is NOT reused
here -- it comes from a different price feed than this candle file (the
two are already known to disagree by ~0.2R on a meaningful fraction of
trades, see idealtp_data_quality_check.py), so pairing a candle-derived
entry with a broker-feed stop is an internal mismatch. Instead the stop is
placed at the low/high of the SAME structural swing that the MSS broke --
i.e. the origin of the impulse leg being re-entered on its FVG retest, not
any later minor pullback low formed after the breakout (that would be a
different, smaller swing, not "the" swing this setup is built on). This
level is already computed during MSS detection as each event's
`stop_level` and is exact at 1-minute resolution (the 5-min structure bars
are built with low=min/high=max over their five 1-min candles, so a 5-min
swing extreme already IS the true 1-minute extreme, not an approximation).

The target price is unaffected by this and still stays exactly as it was
(maxTP's absolute price level, or the 2R-floor computed from the trade's
original entry/stop, per target_resolution.py's rule) -- only entry and
stop move together as a pair. Trades with no qualifying MSS+FVG+retest are
left exactly as they actually happened.
"""
import numpy as np
import pandas as pd

from src import config, data_loading, stats_utils

STRUCTURE_TIMEFRAME_MINUTES = 5  # swings/MSS/FVG are read on 5-min bars, not raw 1-min --
                                   # checked visually: on raw 1-min bars the strict 3-candle
                                   # fractal flags tiny, insignificant pivots as "swings," which
                                   # produced at least one MSS whose "confirmed" direction was
                                   # immediately and fully reversed by the next few bars (see
                                   # outputs/figures/fvg_sanity_check_216315790.png) -- a false
                                   # positive from noise, not a real structural shift. 5-min bars
                                   # is the standard ICT compromise (read structure one notch
                                   # up from the entry-trigger timeframe).
LOOKBACK_MINUTES = 600            # ~120 5-min bars back from entry to search for a recent MSS
FVG_SEARCH_FORWARD_BARS = 3        # the FVG left by the displacement leg forms AT/right next
                              # to the MSS breakout candle itself, not somewhere later --
                              # a wide window here picks up unrelated later micro-patterns
                              # once price has already reversed


def find_swings(highs: np.ndarray, lows: np.ndarray):
    n = len(highs)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    if n < 3:
        return swing_high, swing_low
    swing_high[1:-1] = (highs[1:-1] > highs[:-2]) & (highs[1:-1] > highs[2:])
    swing_low[1:-1] = (lows[1:-1] < lows[:-2]) & (lows[1:-1] < lows[2:])
    return swing_high, swing_low


def find_mss_events(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                     swing_high: np.ndarray, swing_low: np.ndarray) -> list:
    """Each event's `stop_level` is the extreme of the impulse leg that
    produced it -- the lowest low since the previous MSS (of either
    direction) up to and including the breakout candle, for a bullish
    event (mirror: highest high, for bearish). This is "the main swing"
    the stop belongs to, not any later minor pullback low/high formed
    after the breakout has already fired -- the window closes at the
    breakout candle itself, so nothing after it can leak in."""
    events = []
    last_sh_price, last_sl_price = None, None
    consumed_sh, consumed_sl = True, True
    leg_start = 0
    for i in range(len(closes)):
        if swing_high[i]:
            last_sh_price, consumed_sh = highs[i], False
        if swing_low[i]:
            last_sl_price, consumed_sl = lows[i], False
        if last_sh_price is not None and not consumed_sh and closes[i] > last_sh_price:
            stop_level = float(lows[leg_start:i + 1].min())
            events.append({"idx": i, "direction": "bullish", "level": last_sh_price, "stop_level": stop_level})
            consumed_sh = True
            leg_start = i + 1
        if last_sl_price is not None and not consumed_sl and closes[i] < last_sl_price:
            stop_level = float(highs[leg_start:i + 1].max())
            events.append({"idx": i, "direction": "bearish", "level": last_sl_price, "stop_level": stop_level})
            consumed_sl = True
            leg_start = i + 1
    return events


def find_fvgs(highs: np.ndarray, lows: np.ndarray) -> list:
    fvgs = []
    for i in range(1, len(highs) - 1):
        if highs[i - 1] < lows[i + 1]:
            fvgs.append({"idx": i, "direction": "bullish", "bottom": highs[i - 1], "top": lows[i + 1]})
        elif lows[i - 1] > highs[i + 1]:
            fvgs.append({"idx": i, "direction": "bearish", "bottom": highs[i + 1], "top": lows[i - 1]})
    return fvgs


def _find_setup_for_trade(side: str, opens, highs, lows, closes, entry_idx_in_window: int):
    """opens..closes are the LOOKBACK window ending at (not including) the
    trade's actual entry candle; entry_idx_in_window == len(window).
    Returns None, or a dict describing the qualifying MSS+FVG+retest."""
    want_dir = "bullish" if side == "buy" else "bearish"
    swing_high, swing_low = find_swings(highs, lows)
    mss_events = find_mss_events(closes, highs, lows, swing_high, swing_low)
    fvgs = find_fvgs(highs, lows)

    matching_mss = [e for e in mss_events if e["direction"] == want_dir]
    if not matching_mss:
        return None
    mss = matching_mss[-1]  # most recent

    candidate_fvgs = [f for f in fvgs if f["direction"] == want_dir
                       and mss["idx"] - 1 <= f["idx"] <= mss["idx"] + FVG_SEARCH_FORWARD_BARS]
    if not candidate_fvgs:
        return None
    fvg = min(candidate_fvgs, key=lambda f: f["idx"])

    is_buy = side == "buy"
    near_edge = fvg["top"] if is_buy else fvg["bottom"]
    far_edge = fvg["bottom"] if is_buy else fvg["top"]

    search_start = fvg["idx"] + 2  # first candle after the FVG's own 3-candle formation
    retest_idx, invalidated = None, False
    for j in range(search_start, entry_idx_in_window):
        if is_buy:
            if closes[j] < far_edge:
                invalidated = True
                break
            if lows[j] <= near_edge:
                retest_idx = j
                break
        else:
            if closes[j] > far_edge:
                invalidated = True
                break
            if highs[j] >= near_edge:
                retest_idx = j
                break

    if retest_idx is None:
        return None

    return {
        "mss_idx": mss["idx"], "mss_level": mss["level"], "mss_stop_level": mss["stop_level"],
        "fvg_idx": fvg["idx"], "fvg_bottom": fvg["bottom"], "fvg_top": fvg["top"],
        "retest_idx": retest_idx, "retest_price": near_edge,
    }


def resample_structure_timeframe(candles: pd.DataFrame, minutes: int = STRUCTURE_TIMEFRAME_MINUTES) -> pd.DataFrame:
    c = candles.set_index("datetime_utc").resample(f"{minutes}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).dropna().reset_index()
    return c


def build_fvg_entries(trades: pd.DataFrame, candles: pd.DataFrame,
                       lookback_minutes: int = LOOKBACK_MINUTES,
                       structure_candles: pd.DataFrame = None) -> pd.DataFrame:
    """structure_candles: pre-resampled 5-min bars (pass in a cached copy to
    avoid re-resampling 1.7M 1-min rows on every call); computed on the fly
    if not given."""
    sc = structure_candles if structure_candles is not None else resample_structure_timeframe(candles)
    ts = sc["datetime_utc"].values
    opens_all = sc["open"].values
    highs_all = sc["high"].values
    lows_all = sc["low"].values
    closes_all = sc["close"].values

    starts = trades["dateStart_utc"].values
    lookback_start = starts - np.timedelta64(lookback_minutes, "m")
    win_lo = np.searchsorted(ts, lookback_start, side="left")
    win_hi = np.searchsorted(ts, starts, side="left")  # up to (not including) entry candle

    rows = []
    for i in range(len(trades)):
        t = trades.iloc[i]
        lo, hi = win_lo[i], win_hi[i]
        if hi - lo < 5:
            rows.append(dict(id=t["id"], has_setup=False))
            continue

        o = opens_all[lo:hi]; h = highs_all[lo:hi]; l = lows_all[lo:hi]; c = closes_all[lo:hi]
        setup = _find_setup_for_trade(t["side"], o, h, l, c, entry_idx_in_window=len(c))

        if setup is None:
            rows.append(dict(id=t["id"], has_setup=False))
            continue

        retest_time = ts[lo + setup["retest_idx"]]
        rows.append(dict(
            id=t["id"], has_setup=True,
            mss_level=setup["mss_level"], mss_stop_level=setup["mss_stop_level"],
            fvg_bottom=setup["fvg_bottom"], fvg_top=setup["fvg_top"],
            new_entry_price=setup["retest_price"], new_entry_time=retest_time,
            mss_time=ts[lo + setup["mss_idx"]], fvg_time=ts[lo + setup["fvg_idx"]],
        ))

    return pd.DataFrame(rows)


MIN_TARGET_R = 2.0  # user's rule: assume AT LEAST double the stop distance when maxTP is absent


def _resolve_target(row_maxtp, entry, sl, side):
    """Corrected per the user's direct clarification (see the header of
    idealtp_data_quality_check.py): maxTP is the real target and is used
    whenever present (213 trades that realized a profit). idealTP is NOT
    a target -- it is MFE-before-stop -- so it is never used here. When
    maxTP is absent (576 trades), the target is assumed to be AT LEAST
    2x the stop-loss distance, per the user's own stated rule."""
    if not np.isnan(row_maxtp):
        return row_maxtp, True
    risk = abs(entry - sl)
    target = entry + MIN_TARGET_R * risk if side == "buy" else entry - MIN_TARGET_R * risk
    return target, False


def simulate_fvg_outcomes(trades: pd.DataFrame, candles: pd.DataFrame, setups: pd.DataFrame) -> pd.DataFrame:
    """For trades with a qualifying setup: re-race mechanically from the
    new (retest) entry, on 1-minute candles, against a NEW stop at the
    swing that the MSS broke (mss_stop_level -- see module docstring for
    why the original initalSL is not reused).

    Target: when maxTP is real (213 trades), it's an absolute historical
    price level -- kept exactly as recorded, same as the entry/stop for
    THAT case would need no adjustment even if it mixed feeds slightly
    (the candle-vs-trade-log gap is already bounded and documented, ~0.2R
    median, in idealtp_data_quality_check.py). But when maxTP is absent
    and the 2R-floor applies (576 trades), that floor must NOT be computed
    from the original entryPrice/initalSL (a different feed) while entry
    and stop for this setup are candle-derived -- that would silently
    re-introduce the exact cross-feed mismatch the new stop was built to
    avoid. So the floor is recomputed from the NEW entry/stop, in the same
    candle frame, for setup trades specifically.

    Trades with no qualifying setup are reported completely unchanged
    (actual result carried through, target resolved in the original
    entryPrice/initalSL frame since that's the frame the real trade lived in).
    """
    ts = candles["datetime_utc"].values
    opens_all = candles["open"].values
    highs_all = candles["high"].values
    lows_all = candles["low"].values

    ext = np.timedelta64(config.EXIT_STRATEGY_MAX_EXTENSION_MINUTES, "m")
    merged = trades.merge(setups, on="id", how="left")

    rows = []
    for i in range(len(merged)):
        t = merged.iloc[i]
        entry_orig, sl, side = t["entryPrice"], t["initalSL"], t["side"]
        is_buy = side == "buy"
        target_orig_frame, trustworthy = _resolve_target(t["maxTP"], entry_orig, sl, side)
        risk_orig = abs(entry_orig - sl)

        base = dict(id=t["id"], side=side, has_setup=bool(t["has_setup"]),
                    actual_r=t["avgRiskReward"], target_trustworthy=trustworthy,
                    entry_orig=entry_orig, initalSL=sl, amount=t["amount"],
                    risk_price_orig=risk_orig)

        if not t["has_setup"] or np.isnan(target_orig_frame) or risk_orig == 0:
            rows.append(dict(base, new_entry=entry_orig, new_sl=sl, target_price=target_orig_frame,
                             new_r=t["avgRiskReward"], outcome="unchanged", risk_price_new=risk_orig))
            continue

        new_entry = t["new_entry_price"]
        new_sl = t["mss_stop_level"]
        # the new stop must actually sit on the protective side of the new
        # entry -- if the swing low/high isn't below/above it (shouldn't
        # happen structurally, but not assumed), the setup isn't usable
        sl_is_valid = (new_sl < new_entry) if is_buy else (new_sl > new_entry)
        risk_new = abs(new_entry - new_sl)
        if not sl_is_valid or risk_new == 0 or np.isnan(new_sl):
            rows.append(dict(base, new_entry=entry_orig, new_sl=sl, target_price=target_orig_frame,
                             new_r=t["avgRiskReward"], outcome="unchanged", risk_price_new=risk_orig))
            continue

        if trustworthy:
            target = target_orig_frame  # real maxTP: an absolute price level, feed-independent
        else:
            target = new_entry + MIN_TARGET_R * risk_new if is_buy else new_entry - MIN_TARGET_R * risk_new

        start_dt = pd.Timestamp(t["new_entry_time"])
        end_dt = start_dt + pd.Timedelta(ext.astype("timedelta64[m]").astype(int), unit="m")
        lo = np.searchsorted(ts, np.datetime64(start_dt), side="left")
        hi = np.searchsorted(ts, np.datetime64(end_dt), side="right")

        outcome, r = "unresolved", np.nan
        for k in range(lo, hi):
            o, h, l = opens_all[k], highs_all[k], lows_all[k]
            hit_sl = (l <= new_sl) if is_buy else (h >= new_sl)
            hit_tp = (h >= target) if is_buy else (l <= target)
            if hit_sl:
                fill = new_sl if (o >= new_sl if is_buy else o <= new_sl) else o
                outcome = "stop"
            elif hit_tp:
                fill = target if (o <= target if is_buy else o >= target) else o
                outcome = "target"
            else:
                continue
            r = (fill - new_entry) / risk_new if is_buy else (new_entry - fill) / risk_new
            break

        if outcome == "unresolved":
            rows.append(dict(base, new_entry=entry_orig, new_sl=sl, target_price=target_orig_frame,
                             new_r=t["avgRiskReward"], outcome="unchanged_unresolved", risk_price_new=risk_orig))
        else:
            rows.append(dict(base, new_entry=new_entry, new_sl=new_sl, target_price=target, new_r=r,
                             outcome=outcome, risk_price_new=risk_new))

    out = pd.DataFrame(rows)
    out["actual_pnl_usd"] = out["actual_r"] * out["risk_price_orig"] * out["amount"]
    out["new_pnl_usd"] = out["new_r"] * out["risk_price_new"] * out["amount"]
    return out


def max_drawdown_r(series: np.ndarray) -> float:
    if len(series) == 0:
        return np.nan
    equity = np.cumsum(series)
    return float((np.maximum.accumulate(equity) - equity).max())


def summary_report(sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, mask in [("all trades", pd.Series(True, index=sim.index)),
                        ("had a qualifying FVG+MSS setup", sim["has_setup"]),
                        ("no setup (left unchanged)", ~sim["has_setup"])]:
        d = sim[mask]
        rows.append({
            "group": label, "n": len(d),
            "actual_win_rate": float((d["actual_r"] > 0).mean()),
            "new_win_rate": float((d["new_r"] > 0).mean()),
            "actual_total_r": float(d["actual_r"].sum()),
            "new_total_r": float(d["new_r"].sum()),
            "actual_total_usd": float(d["actual_pnl_usd"].sum()),
            "new_total_usd": float(d["new_pnl_usd"].sum()),
            "actual_max_dd_r": max_drawdown_r(d["actual_r"].values),
            "new_max_dd_r": max_drawdown_r(d["new_r"].values),
        })
    return pd.DataFrame(rows)


def change_breakdown(sim: pd.DataFrame, tolerance: float = 1e-6) -> dict:
    d = sim[sim["has_setup"]]
    improved = (d["new_r"] - d["actual_r"] > tolerance).sum()
    worse = (d["actual_r"] - d["new_r"] > tolerance).sum()
    same = len(d) - improved - worse
    return {"n_with_setup": len(d), "n_improved": int(improved), "n_worse": int(worse), "n_same": int(same)}


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    c5 = resample_structure_timeframe(candles)

    setups = build_fvg_entries(trades, candles, structure_candles=c5)
    print(f"trades with a qualifying MSS+FVG+retest before actual entry: "
          f"{setups['has_setup'].sum()} / {len(setups)}")

    sim = simulate_fvg_outcomes(trades, candles, setups)
    print("\n=== headline: actual vs FVG-adjusted entry ===")
    print(summary_report(sim).to_string(index=False))

    print("\n=== change breakdown (trades that HAD a setup) ===")
    print(change_breakdown(sim))
