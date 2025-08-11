# src/data/fetch_data.py
import yfinance as yf
import pandas as pd
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

TICKERS = ["TSLA", "BND", "SPY"]
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def fetch_save(ticker, start="2015-07-01", end="2025-07-31"):
    logging.info(f"Fetching {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        logging.error(f"No data for {ticker} — check yfinance and ticker name.")
        return
    df.index = pd.to_datetime(df.index)
    df.to_csv(RAW_DIR / f"{ticker}.csv")
    logging.info(f"Saved {RAW_DIR / f'{ticker}.csv'} (rows={len(df)})")
    return df

def main(args):
    for t in TICKERS:
        fetch_save(t, start=args.start, end=args.end)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-07-01")
    parser.add_argument("--end", default="2025-07-31")
    args = parser.parse_args()
    main(args)
