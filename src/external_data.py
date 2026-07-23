"""Fetches and caches the external macro series used by
macro_feature_engineering.py (test-only; answers a direct question about
gold/dollar/EUR/VIX/equity-market composite correlations -- never used by
the core NAS100 pipeline, which is deliberately self-contained per the
original spec).

Sources (each is official/free unless noted, and disclosed per-series):
  - VIXCLS   -- CBOE Volatility Index, official, via FRED (Federal Reserve).
  - DTWEXBGS -- Trade Weighted U.S. Dollar Index (Broad), official, FRED.
                NOT identical to ICE DXY -- a free official proxy, not DXY
                itself (documented explicitly, per the original prompt's
                own instruction on this point).
  - DEXUSEU  -- USD/EUR spot rate, official Fed H.10 data, via FRED.
  - SP500    -- S&P 500 daily close, official, via FRED. Used here as the
                free/reliable proxy for "aggregate corporate earnings /
                market profit sentiment" -- no free official source
                publishes a daily aggregate corporate-earnings series;
                the index level is the standard practitioner substitute.
                Flagged as a proxy, not literal earnings data.
  - Gold     -- COMEX gold futures (GC=F), UNOFFICIAL best-effort via
                Yahoo Finance's public chart endpoint. FRED's official
                LBMA gold fixing series (GOLDAMGBD228NLBM /
                GOLDPMGBD228NLBM) was discontinued in 2015 and returns no
                data for this project's 2020-2024 window -- there is no
                remaining free official daily gold series, so this is the
                best available free source, explicitly flagged as such.

All series are cached as raw CSVs under external_data/ (committed to the
repo, not regenerated outputs) so the analysis is reproducible without
network access after the first fetch -- external data can't be
re-derived from anything else already in this repo, unlike this
project's other regenerable CSV/PNG outputs.
"""
import datetime
import subprocess

import pandas as pd

from src import config

EXTERNAL_DIR = config.PROJECT_ROOT / "external_data"

FRED_SERIES = {
    "vix": "VIXCLS",
    "dxy_proxy": "DTWEXBGS",
    "eurusd": "DEXUSEU",
    "sp500": "SP500",
}

START_DATE = "2019-12-01"  # a little before analytics_1.csv's 2020-01-13 start, for prior-day/5d-change warmup
END_DATE = "2024-12-31"

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def _fred_csv_url(series_id: str) -> str:
    return (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            f"&cosd={START_DATE}&coed={END_DATE}")


def _curl_get(url: str, use_browser_ua: bool = False) -> str:
    """Python's requests/urllib3 gets its TLS handshake reset by these
    hosts in this environment, while curl (verified working) does not --
    shelling out to curl is a pragmatic, verified-reliable workaround.

    use_browser_ua is per-host, not a blanket default: Yahoo's chart API
    needs a browser User-Agent or it 429s, but sending that SAME header to
    FRED's fredgraph.csv endpoint gets multi-year requests silently reset
    by FRED's own bot protection (verified by testing both ways) -- curl's
    default UA is what actually works there.
    """
    cmd = ["curl", "-s", "--max-time", "30", "--retry", "2", "--retry-delay", "2"]
    if use_browser_ua:
        cmd += ["-H", f"User-Agent: {BROWSER_HEADERS['User-Agent']}"]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


def fetch_fred_series(name: str, series_id: str, force: bool = False) -> pd.DataFrame:
    cache_path = EXTERNAL_DIR / f"{name}_{series_id}.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, parse_dates=["date"])

    text = _curl_get(_fred_csv_url(series_id), use_browser_ua=False)
    from io import StringIO
    df = pd.read_csv(StringIO(text))
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"])
    df[name] = pd.to_numeric(df[name], errors="coerce")  # FRED uses "." for missing values

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def fetch_gold_yahoo(force: bool = False) -> pd.DataFrame:
    cache_path = EXTERNAL_DIR / "gold_GCF_yahoo.csv"
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, parse_dates=["date"])

    p1 = int(datetime.datetime.strptime(START_DATE, "%Y-%m-%d").timestamp())
    p2 = int((datetime.datetime.strptime(END_DATE, "%Y-%m-%d") + datetime.timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F?period1={p1}&period2={p2}&interval=1d"

    import json
    text = _curl_get(url, use_browser_ua=True)
    data = json.loads(text)["chart"]["result"][0]
    ts = data["timestamp"]
    closes = data["indicators"]["quote"][0]["close"]

    df = pd.DataFrame({
        "date": pd.to_datetime(ts, unit="s").normalize(),
        "gold": closes,
    })
    df = df.dropna(subset=["gold"]).drop_duplicates(subset="date")

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def build_macro_daily_table(force: bool = False) -> pd.DataFrame:
    """Outer-merges all 5 series by calendar date, then forward-fills gaps
    up to 5 calendar days (holiday/exchange-calendar mismatches between
    FX/futures markets which trade ~24/5 and NYSE-hours series like VIX
    and SP500) -- documented, bounded fill, not an unbounded carry-forward.
    """
    frames = [fetch_fred_series(name, sid, force=force) for name, sid in FRED_SERIES.items()]
    frames.append(fetch_gold_yahoo(force=force))

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    out = out.sort_values("date").reset_index(drop=True)

    value_cols = list(FRED_SERIES.keys()) + ["gold"]
    out[value_cols] = out[value_cols].ffill(limit=5)
    return out


if __name__ == "__main__":
    daily = build_macro_daily_table()
    print(f"shape: {daily.shape}")
    print(f"date range: {daily['date'].min()} .. {daily['date'].max()}")
    print()
    print("nulls per column:")
    print(daily.isnull().sum())
    print()
    print(daily.tail(10))
