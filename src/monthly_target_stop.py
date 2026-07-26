"""Monthly profit-target-then-stop (test-only): once cumulative realized
PnL within a calendar month reaches +X% of the account's fixed reference
base ($100,000 -- see config.ACCOUNT_BASE_USD), stop taking any further
trades for the rest of that month. Trades are processed in chronological
order within each month; a trade already "in progress" when the target is
crossed is left as-is (the target check happens between trades, using each
trade's own realized rPnL, matching how a trader would actually observe
their own running total intra-month).

Two variants:
  - profit_only: halt once +X% is reached; losing months are left to run
    their full course (no downside stop).
  - target_or_stop: halt on EITHER +X% or -X% (symmetric), a natural
    extension worth reporting alongside the profit-only rule the user asked
    for.
"""
import numpy as np
import pandas as pd

from src import config, data_loading, stats_utils


def simulate_monthly_target_stop(trades: pd.DataFrame, target_pct: float,
                                  variant: str = "profit_only") -> pd.DataFrame:
    """Returns trades with an added 'taken' column (False for trades that
    would have been skipped because the month's target/stop was already hit)
    and 'capped_pnl' (0 for skipped trades, rPnL otherwise)."""
    target_usd = target_pct * config.ACCOUNT_BASE_USD
    out = trades.sort_values("dateStart_utc").copy()
    out["month"] = out["dateStart_utc"].dt.to_period("M").astype(str)

    taken = np.ones(len(out), dtype=bool)
    running = out.groupby("month")["rPnL"]

    for month, idx in out.groupby("month").groups.items():
        idx = list(idx)
        cum = 0.0
        halted = False
        for i in idx:
            pos = out.index.get_loc(i)
            if halted:
                taken[pos] = False
                continue
            cum += out.loc[i, "rPnL"]
            if cum >= target_usd:
                halted = True
            elif variant == "target_or_stop" and cum <= -target_usd:
                halted = True

    out["taken"] = taken
    out["capped_pnl"] = np.where(out["taken"], out["rPnL"], 0.0)
    return out


def monthly_before_after(capped: pd.DataFrame) -> pd.DataFrame:
    monthly = capped.groupby("month").agg(
        n_trades=("id", "count"),
        n_taken=("taken", "sum"),
        original_pnl=("rPnL", "sum"),
        capped_pnl=("capped_pnl", "sum"),
    ).reset_index()
    monthly["n_skipped"] = monthly["n_trades"] - monthly["n_taken"]
    monthly["diff"] = monthly["capped_pnl"] - monthly["original_pnl"]
    return monthly


def yearly_before_after(capped: pd.DataFrame) -> pd.DataFrame:
    c = capped.copy()
    c["year"] = c["dateStart_utc"].dt.year
    yearly = c.groupby("year").agg(
        n_trades=("id", "count"), n_taken=("taken", "sum"),
        original_pnl=("rPnL", "sum"), capped_pnl=("capped_pnl", "sum"),
    ).reset_index()
    yearly["diff"] = yearly["capped_pnl"] - yearly["original_pnl"]
    return yearly


def grid_summary(trades: pd.DataFrame, target_grid=config.MONTHLY_TARGET_GRID) -> pd.DataFrame:
    rows = []
    baseline_total = trades["rPnL"].sum()
    baseline_dd = _max_dd_usd(trades.sort_values("dateStart_utc")["rPnL"].values)

    rows.append({"variant": "baseline (no cap)", "target_pct": np.nan, "total_pnl": baseline_total,
                 "max_drawdown_usd": baseline_dd, "n_months_improved": np.nan, "n_months_degraded": np.nan})

    for variant in ("profit_only", "target_or_stop"):
        for pct in target_grid:
            capped = simulate_monthly_target_stop(trades, pct, variant=variant)
            monthly = monthly_before_after(capped)
            total = capped["capped_pnl"].sum()
            dd = _max_dd_usd(capped.sort_values("dateStart_utc")["capped_pnl"].values)
            rows.append({
                "variant": variant, "target_pct": pct, "total_pnl": total,
                "max_drawdown_usd": dd,
                "n_months_improved": int((monthly["diff"] > 0).sum()),
                "n_months_degraded": int((monthly["diff"] < 0).sum()),
            })
    out = pd.DataFrame(rows)
    out["pnl_vs_baseline_pct"] = (out["total_pnl"] - baseline_total) / abs(baseline_total) * 100
    return out


def _max_dd_usd(pnl_values: np.ndarray) -> float:
    if len(pnl_values) == 0:
        return np.nan
    equity = np.cumsum(pnl_values)
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max())


if __name__ == "__main__":
    trades = data_loading.load_trades()

    print("=== grid summary ===")
    summary = grid_summary(trades)
    print(summary.to_string(index=False))

    print("\n=== monthly detail for 2% profit_only (as requested) ===")
    capped_2pct = simulate_monthly_target_stop(trades, 0.02, variant="profit_only")
    monthly = monthly_before_after(capped_2pct)
    print(monthly.to_string(index=False))
    print(f"\nmonths improved: {(monthly['diff']>0).sum()}, degraded: {(monthly['diff']<0).sum()}, "
          f"unchanged: {(monthly['diff']==0).sum()}")

    print("\n=== yearly detail for 2% profit_only ===")
    yearly = yearly_before_after(capped_2pct)
    print(yearly.to_string(index=False))
