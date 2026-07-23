"""Stage 3: classify how well each trade's exit captured the excursion that
was actually available before it closed (entry-side pattern quality is a
separate question, handled in feature_engineering.py / pattern_search.py).

Definitions (all measured strictly within [dateStart, dateEnd], i.e. this is
never a claim about profit available *after* the trade closed):
  - exit_efficiency = realized_r / mfe_r, only meaningful when mfe_r > 0
  - left_on_table_r = mfe_r - realized_r, when positive
  - left_on_table_usd = left_on_table_r * (risk_price * amount)   [actual
    per-trade dollar risk unit, ~$500 by position-sizing design but not
    exactly, so computed per trade rather than assumed]
"""
import numpy as np
import pandas as pd

from src import data_loading, excursion

EFFICIENCY_PERFECT = 0.90
EFFICIENCY_GOOD = 0.60
EFFICIENCY_EARLY = 0.30
# Below this, a positive MFE is a trivial intra-candle blip, not a real
# favorable move that was subsequently "given back" -- avoids mislabeling
# ordinary stop-outs as poor exits.
MATERIALITY_MFE_R = 0.15


def classify_exit_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    realized_r = out["avgRiskReward"]
    mfe_r = out["mfe_r"]

    dollar_risk_unit = out["risk_price"] * out["amount"]

    efficiency = np.where(mfe_r > 0, realized_r / mfe_r, np.nan)
    out["exit_efficiency"] = efficiency

    left_on_table_r = (mfe_r - realized_r).clip(lower=0)
    out["left_on_table_r"] = left_on_table_r
    out["left_on_table_usd"] = left_on_table_r * dollar_risk_unit

    def label(row):
        mfe = row["mfe_r"]
        eff = row["exit_efficiency"]
        r = row["avgRiskReward"]

        if pd.isna(mfe) or mfe < MATERIALITY_MFE_R:
            return "No Favorable Excursion"  # stopped out with no meaningful upside ever seen
        if pd.isna(eff):
            return "Undefined"
        if eff >= EFFICIENCY_PERFECT:
            return "Perfect Exit"
        if eff >= EFFICIENCY_GOOD:
            return "Good Exit"
        if r < 0:
            # had real favorable excursion, still closed a loser: gave it all back
            return "Terrible Exit"
        if eff >= EFFICIENCY_EARLY:
            return "Early Exit"
        return "Terrible Exit"

    out["exit_quality"] = out.apply(label, axis=1)
    return out


if __name__ == "__main__":
    trades = data_loading.load_trades()
    candles = data_loading.load_candles()
    exc = excursion.reconstruct_excursions(trades, candles)
    result = classify_exit_quality(exc)

    print(result["exit_quality"].value_counts())
    print()
    summary = result.groupby("exit_quality").agg(
        n=("id", "count"),
        total_left_on_table_usd=("left_on_table_usd", "sum"),
        mean_left_on_table_r=("left_on_table_r", "mean"),
        mean_realized_r=("avgRiskReward", "mean"),
    )
    print(summary)
    print()
    print(f"Total $ left on the table across all trades: {result['left_on_table_usd'].sum():,.2f}")
