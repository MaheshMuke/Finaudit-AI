"""
Nifty 500 financial data downloader (free, no API key).

Sources:
  - Company list: NSE official Nifty 500 constituents CSV
  - Financials:   Yahoo Finance via yfinance (annual + quarterly, latest available)

Install:  pip install yfinance pandas requests openpyxl
Run:      python nifty500_financials.py
Resume:   just run it again - finished companies are cached in ./cache and skipped.

Output (in ./output):
  income_statement_quarterly.csv, income_statement_annual.csv
  balance_sheet_quarterly.csv,    balance_sheet_annual.csv
  cash_flow_quarterly.csv,        cash_flow_annual.csv
  latest_quarter_summary.xlsx     (one row per company, key metrics)
"""

import io
import os
import time
import random

import pandas as pd
import requests
import yfinance as yf

CACHE_DIR = "cache"
OUT_DIR = "output"
NIFTY500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
LOCAL_LIST = "ind_nifty500list.csv"  # fallback: download it manually from NSE and put it here
DELAY = (0.6, 1.4)  # random pause between companies (seconds) - be polite
RETRIES = 3

STATEMENTS = {
    "income_statement_quarterly": "quarterly_income_stmt",
    "income_statement_annual": "income_stmt",
    "balance_sheet_quarterly": "quarterly_balance_sheet",
    "balance_sheet_annual": "balance_sheet",
    "cash_flow_quarterly": "quarterly_cashflow",
    "cash_flow_annual": "cashflow",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,*/*",
    "Referer": "https://www.nseindia.com/",
}


def load_company_list() -> pd.DataFrame:
    """Get Nifty 500 constituents from NSE, or from a local CSV if NSE blocks us."""
    try:
        r = requests.get(NIFTY500_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.to_csv(LOCAL_LIST, index=False)
        print(f"Downloaded Nifty 500 list from NSE ({len(df)} companies)")
    except Exception as e:
        if os.path.exists(LOCAL_LIST):
            print(f"NSE download failed ({e}); using local {LOCAL_LIST}")
            df = pd.read_csv(LOCAL_LIST)
        else:
            raise SystemExit(
                f"Could not fetch the company list ({e}).\n"
                f"Download it manually from {NIFTY500_URL} and save as {LOCAL_LIST}"
            )
    return df


def tidy(frame: pd.DataFrame, symbol: str, name: str, statement: str) -> pd.DataFrame:
    """Convert yfinance's wide table (items x dates) to a long table."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    long = frame.copy()
    long.index.name = "item"
    long = long.reset_index().melt(id_vars="item", var_name="period_end", value_name="value")
    long = long.dropna(subset=["value"])
    long["period_end"] = pd.to_datetime(long["period_end"]).dt.date
    long.insert(0, "statement", statement)
    long.insert(0, "company", name)
    long.insert(0, "symbol", symbol)
    return long


def fetch_company(symbol: str, name: str) -> dict:
    ticker = yf.Ticker(f"{symbol}.NS")
    result = {}
    for key, attr in STATEMENTS.items():
        result[key] = pd.DataFrame()
        for attempt in range(RETRIES):
            try:
                result[key] = tidy(getattr(ticker, attr), symbol, name, key)
                break
            except Exception as e:
                # Show the real reason (rate limit, outdated yfinance, network...)
                print(f"    ! {symbol} {key} attempt {attempt + 1}: {type(e).__name__}: {e}")
                time.sleep(3 * (attempt + 1))
    return result


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    companies = load_company_list()
    total = len(companies)

    print(f"yfinance version: {yf.__version__}")
    empty_streak = 0
    got_any = False

    for i, row in companies.iterrows():
        symbol, name = str(row["Symbol"]).strip(), str(row["Company Name"]).strip()
        cache_file = os.path.join(CACHE_DIR, f"{symbol}.csv")
        if os.path.exists(cache_file):
            got_any = True
            continue

        data = fetch_company(symbol, name)
        parts = [d for d in data.values() if not d.empty]
        print(f"[{i + 1}/{total}] {symbol}: {sum(len(p) for p in parts)} data points")

        if parts:
            pd.concat(parts, ignore_index=True).to_csv(cache_file, index=False)  # only cache real data
            empty_streak = 0
            got_any = True
        else:
            empty_streak += 1
            if not got_any and empty_streak >= 5:
                raise SystemExit(
                    "\nFirst 5 companies returned nothing - Yahoo is blocking/rate-limiting you or "
                    "yfinance is outdated.\nTry:  pip install -U yfinance   then wait a few minutes "
                    "(or switch network / hotspot) and run again."
                )
        time.sleep(random.uniform(*DELAY))

    # ---- merge cache into final files ----
    frames = [pd.read_csv(os.path.join(CACHE_DIR, f)) for f in os.listdir(CACHE_DIR) if f.endswith(".csv")]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("No data collected - nothing to write.")
    alldata = pd.concat(frames, ignore_index=True)

    for key in STATEMENTS:
        part = alldata[alldata["statement"] == key]
        part.drop(columns="statement").to_csv(os.path.join(OUT_DIR, f"{key}.csv"), index=False)

    # ---- latest-quarter summary (one row per company) ----
    q = alldata[alldata["statement"] == "income_statement_quarterly"].copy()
    q["period_end"] = pd.to_datetime(q["period_end"])
    wanted = ["Total Revenue", "Operating Income", "EBITDA", "Net Income", "Basic EPS", "Diluted EPS"]
    q = q[q["item"].isin(wanted)]
    latest = q.sort_values("period_end").groupby(["symbol", "company", "item"]).last().reset_index()
    summary = latest.pivot_table(index=["symbol", "company"], columns="item", values="value").reset_index()
    dates = latest.groupby("symbol")["period_end"].max().rename("latest_quarter_end")
    summary = summary.merge(dates, on="symbol", how="left")
    summary.to_excel(os.path.join(OUT_DIR, "latest_quarter_summary.xlsx"), index=False)

    print(f"\nDone. Files are in ./{OUT_DIR}")
    missing = total - summary["symbol"].nunique()
    if missing:
        print(f"{missing} companies had no quarterly data on Yahoo (common for newly listed or small caps).")


if __name__ == "__main__":
    main()