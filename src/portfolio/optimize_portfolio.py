# src/portfolio/optimize_portfolio.py
import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting, CLA, objective_functions
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
PROC_DIR = Path("data/processed")
OUT_DIR = Path("models/portfolio")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_expected_returns_from_forecast(tsla_forecast_csv, combined_returns_csv):
    # tsla_forecast_csv should be a CSV output from best model containing forecasted price series
    fc = pd.read_csv(tsla_forecast_csv, index_col=0, parse_dates=True)
    # Calculate forecasted return = (mean(fc)/last_price - 1) annualized? Simpler: use mean daily return of forecast horizon then annualize
    y_pred = fc['y_pred'] if 'y_pred' in fc.columns else fc.iloc[:,0]
    # daily return average over forecast horizon
    daily_mean_ret = y_pred.pct_change().dropna().mean()
    annualized = (1 + daily_mean_ret) ** 252 - 1
    logging.info("TSLA forecast implied annual return: %.2f%%", annualized * 100)
    # For BND and SPY use historical mean daily returns
    comb = pd.read_csv(combined_returns_csv, index_col=0, parse_dates=True)
    bnd_annual = comb["BND_ret"].dropna().mean() * 252
    spy_annual = comb["SPY_ret"].dropna().mean() * 252
    logging.info("BND annual return: %.2f%%, SPY annual return: %.2f%%", bnd_annual*100, spy_annual*100)
    return {"TSLA": annualized, "BND": bnd_annual, "SPY": spy_annual}

def run_optimization(expected_returns_dict, combined_returns_csv):
    comb = pd.read_csv(combined_returns_csv, index_col=0, parse_dates=True)
    # convert to daily returns series with columns TSLA_ret, BND_ret, SPY_ret
    returns = comb[["TSLA_ret","BND_ret","SPY_ret"]].dropna()
    mu = pd.Series([expected_returns_dict["TSLA"], expected_returns_dict["BND"], expected_returns_dict["SPY"]], index=["TSLA","BND","SPY"])
    S = returns.cov() * 252  # annualize covariance
    # Use Efficient Frontier
    ef = EfficientFrontier(mu, S)
    # Max Sharpe
    w_max_sharpe = ef.max_sharpe()
    ef.clean_weights()
    # Minimum volatility
    ef_min_vol = EfficientFrontier(mu, S)
    w_min_vol = ef_min_vol.min_volatility()
    # compute performance metrics
    from pypfopt import expected_returns as er, risk_models as rm
    # Calculate portfolio performance
    def perf(weights):
        w = np.array([weights.get(k,0) for k in ["TSLA","BND","SPY"]])
        port_return = np.dot(w, mu.values)
        port_vol = np.sqrt(np.dot(w.T, np.dot(S.values, w)))
        sharpe = port_return / port_vol
        return {"ret": port_return, "vol": port_vol, "sharpe": sharpe}
    p1 = perf(w_max_sharpe)
    p2 = perf(w_min_vol)
    # Save results
    pd.DataFrame([w_max_sharpe, w_min_vol]).to_csv(OUT_DIR / "portfolio_weights.csv")
    pd.DataFrame([p1,p2], index=["max_sharpe","min_vol"]).to_csv(OUT_DIR / "portfolio_perf.csv")
    # Efficient frontier plot using random portfolios
    import matplotlib.pyplot as plt
    results = []
    n_portfolios = 20000
    np.random.seed(42)
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(3), size=1).flatten()
        r = np.dot(w, mu.values)
        v = np.sqrt(np.dot(w.T, np.dot(S.values, w)))
        results.append([v, r])
    results = np.array(results)
    plt.figure(figsize=(8,6))
    plt.scatter(results[:,0], results[:,1], s=5, alpha=0.3)
    plt.scatter(p1["vol"], p1["ret"], c='r', marker='*', s=200, label='Max Sharpe')
    plt.scatter(p2["vol"], p2["ret"], c='g', marker='*', s=200, label='Min Vol')
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Efficient Frontier (simulated)")
    plt.legend()
    plt.grid(True)
    plt.savefig(OUT_DIR / "efficient_frontier.png", dpi=150)
    return {"max_sharpe_weights": w_max_sharpe, "min_vol_weights": w_min_vol, "perf_max_sharpe": p1, "perf_min_vol": p2}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsla_forecast", default="models/arima/arima_forecast_test.csv", help="Use forecast from the chosen model")
    parser.add_argument("--combined", default="data/processed/combined_adj_and_returns.csv")
    args = parser.parse_args()
    mu = compute_expected_returns_from_forecast(args.tsla_forecast, args.combined)
    out = run_optimization(mu, args.combined)
    print("Done. Results saved to", OUT_DIR)
