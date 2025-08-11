# src/data/fetch_data.py
import yfinance as yf
import pandas as pd
import logging
from pathlib import Path
import argparse

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def fetch_and_save(ticker, start, end):
    logging.info(f"Fetching {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    df.reset_index(inplace=True)  # Date as first column
    # Rename 'Adj Close' to 'Price' if you prefer that naming
    df.rename(columns={'Adj Close': 'Price'}, inplace=True)
    # Save clean without metadata
    df.to_csv(RAW_DIR / f"{ticker}.csv", index=False)
    logging.info(f"Saved clean CSV to {RAW_DIR / f'{ticker}.csv'} (rows={len(df)})")

def main(args):
    tickers = ["TSLA", "BND", "SPY"]
    for t in tickers:
        fetch_and_save(t, args.start, args.end)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    main(args)
