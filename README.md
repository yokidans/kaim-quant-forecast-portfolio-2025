# Kaim Quant Forecast Portfolio (2025)

## Project Overview

This project aims to build a quantitative forecasting and portfolio management system using financial market data from various assets such as TSLA (Tesla), BND (Bond ETF), and SPY (S&P 500 ETF). The goal is to fetch, preprocess, analyze, and prepare clean time series data for downstream modeling and strategy development.

---

## Repository Structure

### kaim-quant-forecast-portfolio-2025/
### ├─ README.md
### ├─ requirements.txt
### ├─ environment.yml            # optional conda env
### ├─ LICENSE
### ├─ data/
### │  ├─ raw/                    # raw downloaded CSVs
### │  └─ processed/              # cleaned and feature-engineered csvs
### ├─ notebooks/
### │  ├─ 01_EDA.ipynb
### │  ├─ 02_ARIMA_modeling.ipynb
### │  └─ 03_LSTM_modeling.ipynb
### ├─ src/
### │  ├─ __init__.py
### │  ├─ data/
### │  │  ├─ fetch_data.py
### │  │  └─ preprocess.py
### │  ├─ features/
### │  │  └─ feature_engineering.py
### │  ├─ models/
### │  │  ├─ arima_model.py
### │  │  └─ lstm_model.py
### │  ├─ portfolio/
### │  │  └─ optimize_portfolio.py
### │  ├─ backtest/
### │  │  └─ backtest.py
### │  └─ utils/
### │     ├─ metrics.py
### │     └─ viz.py
### ├─ reports/
### │  └─ Investment_Memo.pdf     # final output
### └─ scripts/
###    ├─ run_all.sh              # reproducible end-to-end run script
###    └─ run_interim.sh          # runs Task 1 only (for interim)


---

## Data Fetching

- Command example:
  ```bash
  python src\data\fetch_data.py --start "2015-07-01" --end "2025-07-31"
Fetches historical daily price data for assets TSLA, BND, SPY.

Saves cleaned raw CSV files to data/raw/ directory.

    python src\data\preprocess.py --start "2015-07-01" --end "2025-07-31"
    Loads raw CSV files and processes each ticker.

## Reindex data to business days (using reindex_business function)

- Calculate returns (ret, log_ret)
- Compute rolling statistics (vol_30, ma_50, ma_200)
- Perform stationarity tests (ADF test)
- Calculate risk metrics (VaR, Sharpe ratio)

## Saves combined processed dataset for modeling.


NameError: name 'reindex_business' is not defined
Fix: Added/Imported reindex_business function to reindex DataFrame to business day frequency.

Upon successful execution, warnings appeared:

UserWarning: Could not infer format, falling back to dateutil
FutureWarning: Default fill_method='pad' in Series.pct_change deprecated
Consider specifying date format explicitly when reading CSV.

Adjust pct_change calls with fill_method=None.

Most critically, all columns except Volume showed 2632 missing values:

Missing values for TSLA:
{'Open': 2632, 'High': 2632, ...}
Stats are all NaN.
Likely causes:
- CSV files are empty or corrupted.
- Date parsing mismatch causing index misalignment.
- Reindexing to dates outside the data range, creating all-NaN rows.

Recommended to:
- Inspect raw CSV files for validity.
- Print DataFrame head immediately after loading.
- Confirm date column and index handling.
- Validate reindex_business generates correct date ranges.


