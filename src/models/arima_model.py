# src/models/arima_model.py
import sys
import logging
from pathlib import Path
import warnings
import argparse
import joblib
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Create required directories if they don't exist
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("models/arima").mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(Path("logs/arima_model.log")),
        logging.StreamHandler()
    ]
)

# Suppress warnings
warnings.filterwarnings("ignore")

def train_and_evaluate(ts_series, train_end_date="2023-12-31", test_end_date="2025-07-31"):
    """Train ARIMA model and evaluate on test set"""
    try:
        # Convert dates to datetime objects
        train_end = pd.to_datetime(train_end_date)
        test_end = pd.to_datetime(test_end_date)
        
        ts = ts_series.dropna()
        train = ts[ts.index <= train_end]
        test = ts[(ts.index > train_end) & (ts.index <= test_end)]
        
        logging.info("Train length: %d, Test length: %d", len(train), len(test))
        
        model = auto_arima(
            train,
            seasonal=True,  # Changed from False to True
            m=12,          # Monthly seasonality if data is daily
            stepwise=True,
            suppress_warnings=True
        )
        
        logging.info("Auto-ARIMA selected: %s", model.summary())
        
        n_periods = len(test)
        fc, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
        fc_series = pd.Series(fc, index=test.index)
        
        mae = mean_absolute_error(test, fc_series)
        rmse = np.sqrt(mean_squared_error(test, fc_series))
        mape = np.mean(np.abs((test - fc_series) / test)) * 100
        
        logging.info("ARIMA MAE=%.4f RMSE=%.4f MAPE=%.4f%%", mae, rmse, mape)
        
        joblib.dump(model, Path("models/arima/arima_model.joblib"))
        pd.DataFrame({
            "y_true": test,
            "y_pred": fc_series,
            "lower": conf_int[:, 0],
            "upper": conf_int[:, 1]
        }).to_csv(Path("models/arima/arima_forecast_test.csv"))
        
        return {
            "model": model,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "forecast": fc_series,
            "conf_int": conf_int
        }
        
    except Exception as e:
        logging.error("Error in train_and_evaluate: %s", str(e))
        raise

def load_series():
    """Load time series data with validation"""
    try:
        file_path = Path("data/processed/combined_adj_and_returns.csv")
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at {file_path}")
            
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        if "TSLA_adj" not in df.columns:
            raise ValueError("TSLA_adj column not found in the data")
        return df["TSLA_adj"].dropna()
    except Exception as e:
        logging.error("Failed to load data: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        # Log environment info
        logging.info("Python version: %s", sys.version)
        logging.info("NumPy version: %s", np.__version__)
        logging.info("Pandas version: %s", pd.__version__)
        
        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--train_end", default="2023-12-31")
        parser.add_argument("--test_end", default="2025-07-31")
        args = parser.parse_args()
        
        # Load data
        ts = load_series()
        
        # Train and evaluate
        res = train_and_evaluate(ts, train_end_date=args.train_end, test_end_date=args.test_end)
        
        print("Done. Metrics:", {k: res[k] for k in ["mae", "rmse", "mape"]})
        
    except Exception as e:
        logging.error("Script failed: %s", str(e))
        sys.exit(1)