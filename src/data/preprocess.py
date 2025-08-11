# src/data/preprocess.py
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
import logging
import argparse

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def load_csv(ticker):
    df = pd.read_csv(RAW_DIR / f"{ticker}.csv", index_col=0, parse_dates=True)
    return df

def reindex_business(df, start=None, end=None):
    idx = pd.date_range(start=start or df.index.min(), end=end or df.index.max(), freq='B')  # business days
    df = df.reindex(idx)
    return df

def compute_basic_features(df):
    df = df.copy()
    # Ensure numeric types
    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Ffill prices, zero fill volume
    df[["Open","High","Low","Close","Adj Close"]] = df[["Open","High","Low","Close","Adj Close"]].ffill()
    df["Volume"] = df["Volume"].fillna(0)
    # Daily returns
    df["ret"] = df["Adj Close"].pct_change()
    # log-returns for modeling if needed
    df["log_ret"] = np.log1p(df["ret"])
    # rolling vol (30-day)
    df["vol_30"] = df["ret"].rolling(window=30).std() * np.sqrt(252)  # annualized
    # rolling mean for trend
    df["ma_50"] = df["Adj Close"].rolling(50).mean()
    df["ma_200"] = df["Adj Close"].rolling(200).mean()
    return df

def adf_test(series, name):
    series = series.dropna()
    if len(series) < 10:
        return {"adf_stat": np.nan, "pvalue": np.nan}
    res = adfuller(series, autolag='AIC')
    return {"adf_stat": res[0], "pvalue": res[1], "n_lags": res[2], "n_obs": res[3]}

def var_historic(returns, level=0.05):
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return np.percentile(returns, 100 * level)

def sharpe_ratio(returns, risk_free=0.0):
    r = returns.dropna()
    if r.empty:
        return np.nan
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    if ann_vol == 0:
        return np.nan
    return (ann_ret - risk_free) / ann_vol

def process_ticker(ticker, start=None, end=None):
    logging.info("Processing %s", ticker)
    df = load_csv(ticker)
    df = reindex_business(df, start=start, end=end)
    df = compute_basic_features(df)
    # missing check
    missing = df.isna().sum()
    logging.info("Missing values for %s:\n%s", ticker, missing.to_dict())
    # ADF tests
    adf_close = adf_test(df["Adj Close"], ticker + " Adj Close")
    adf_ret = adf_test(df["ret"], ticker + " returns")
    stats = {
        "adf_close": adf_close,
        "adf_ret": adf_ret,
        "var_5pct": var_historic(df["ret"], 0.05),
        "sharpe": sharpe_ratio(df["ret"])
    }
    logging.info("Stats for %s: %s", ticker, stats)
    df.to_csv(PROC_DIR / f"{ticker}_processed.csv")
    return df, stats

def main(args):
    all_stats = {}
    for t in ["TSLA","BND","SPY"]:
        df, stats = process_ticker(t, start=args.start, end=args.end)
        all_stats[t] = stats
    # save a combined dataset for merging price series (Adj Close)
    dfs = []
    for t in ["TSLA","BND","SPY"]:
        df = pd.read_csv(PROC_DIR / f"{t}_processed.csv", index_col=0, parse_dates=True)
        dfs.append(df[["Adj Close","ret"]].rename(columns={"Adj Close": f"{t}_adj", "ret": f"{t}_ret"}))
    combined = pd.concat(dfs, axis=1)
    combined.to_csv(PROC_DIR / "combined_adj_and_returns.csv")
    logging.info("Saved combined processed data.")
    # store stats
    pd.DataFrame.from_dict({k: {"adf_close_p": v["adf_close"]["pvalue"], "adf_ret_p": v["adf_ret"]["pvalue"], "var_5pct": v["var_5pct"], "sharpe": v["sharpe"]} for k,v in all_stats.items()}, orient='index').to_csv(PROC_DIR / "summary_stats.csv")
    logging.info("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-07-01")
    parser.add_argument("--end", default="2025-07-31")
    args = parser.parse_args()
    main(args)
