# src/backtest/backtest.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
PROC = Path("data/processed/combined_adj_and_returns.csv")
OUT = Path("reports")
OUT.mkdir(parents=True, exist_ok=True)

def load_returns():
    df = pd.read_csv(PROC, index_col=0, parse_dates=True)
    return df[["TSLA_ret","BND_ret","SPY_ret"]].dropna()

def run_backtest(weights, start="2024-08-01", end="2025-07-31", rebalance="none"):
    returns = load_returns()
    returns = returns[start:end]
    # portfolio returns series
    w = np.array([weights["TSLA"], weights["BND"], weights["SPY"]])
    # if monthly rebalance: resample
    if rebalance == "monthly":
        # iterate months and apply weights to each month returns
        grouped = returns.resample('M')
        portfolio_ret = []
        for name, group in grouped:
            # compute daily portfolio returns for the month
            daily = group.dot(w)
            portfolio_ret.append(daily)
        portfolio_ret = pd.concat(portfolio_ret)
    else:
        portfolio_ret = returns.dot(w)
    cum = (1 + portfolio_ret).cumprod()
    return portfolio_ret, cum

def calc_sharpe(returns):
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    return ann_ret / ann_vol

if __name__ == "__main__":
    # load optimized weights from earlier results file
    weights_df = pd.read_csv("models/portfolio/portfolio_weights.csv", index_col=0).T
    # pick first row as max_sharpe (depending on save order)
    # If missing, fallback to equal weights
    if not weights_df.empty:
        weights = weights_df.loc[0].to_dict()
    else:
        weights = {"TSLA":0.2,"BND":0.4,"SPY":0.4}
    # Make sure floats
    weights = {k: float(v) for k,v in weights.items()}
    logging.info("Strategy weights: %s", weights)
    # backtest strategy
    strat_ret, strat_cum = run_backtest(weights, rebalance="none")
    # benchmark 60% SPY / 40% BND
    bench_w = {"TSLA":0.0, "BND":0.4, "SPY":0.6}
    bench_ret, bench_cum = run_backtest(bench_w, rebalance="none")
    # plot
    plt.figure(figsize=(10,6))
    plt.plot(strat_cum, label='Strategy')
    plt.plot(bench_cum, label='60/40 Benchmark')
    plt.title("Backtest cumulative returns (2024-08-01 to 2025-07-31)")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT / "backtest_cumulative.png", dpi=150)
    # stats
    strat_total_return = strat_cum.iloc[-1] - 1
    bench_total_return = bench_cum.iloc[-1] - 1
    strat_sharpe = calc_sharpe(strat_ret)
    bench_sharpe = calc_sharpe(bench_ret)
    pd.DataFrame({
        "strategy_total_return": [strat_total_return],
        "benchmark_total_return": [bench_total_return],
        "strategy_sharpe": [strat_sharpe],
        "benchmark_sharpe": [bench_sharpe]
    }).to_csv(OUT / "backtest_summary.csv", index=False)
    logging.info("Saved backtest outputs to %s", OUT)
