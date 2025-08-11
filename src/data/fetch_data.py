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

import time

def fetch_save(ticker, start="2015-07-01", end="2025-07-31", retries=3, delay=5):
    logging.info(f"Fetching {ticker} from {start} to {end}")
    for attempt in range(1, retries + 1):
        try:
            # Increase timeout from default to 60s
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                timeout=60
            )
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            df.index = pd.to_datetime(df.index)
            df.index.name = "Date"

            expected_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
            for col in expected_cols:
                if col not in df.columns:
                    logging.warning(f"{ticker}: Missing expected column {col} — will fill with NaN")
                    df[col] = pd.NA
            df = df[expected_cols]

            out_path = RAW_DIR / f"{ticker}.csv"
            df.to_csv(out_path, index=True)
            logging.info(f"Saved clean CSV to {out_path} (rows={len(df)})")
            return df

        except Exception as e:
            logging.error(f"Attempt {attempt} failed for {ticker}: {e}")
            if attempt < retries:
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"All {retries} attempts failed for {ticker}.")
                return None


def main(args):
    for t in TICKERS:
        fetch_save(t, start=args.start, end=args.end)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-07-01")
    parser.add_argument("--end", default="2025-07-31")
    args = parser.parse_args()
    main(args)
