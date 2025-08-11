# src/models/arima_model.py
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import argparse
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROC_DIR = Path("data/processed")
OUT_DIR = Path("models/arima")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate(ts_series, train_end_date="2023-12-31", test_end_date="2025-07-31"):
    ts = ts_series.dropna()
    train = ts[:train_end_date]
    test = ts[train_end_date + pd.Timedelta(days=1):test_end_date]
    logging.info("Train length: %d, Test length: %d", len(train), len(test))
    # If non-stationary use differencing -> auto_arima handles d
    model = auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', trace=True)
    logging.info("Auto-ARIMA selected: %s", model.summary())
    # Forecast
    n_periods = len(test)
    fc, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)
    fc_index = test.index
    fc_series = pd.Series(fc, index=fc_index)
    # metrics on prices (Adj Close)
    mae = mean_absolute_error(test, fc_series)
    rmse = np.sqrt(mean_squared_error(test, fc_series))
    mape = np.mean(np.abs((test - fc_series) / test)) * 100
    logging.info("ARIMA MAE=%f RMSE=%f MAPE=%f", mae, rmse, mape)
    # Save model & forecast
    joblib.dump(model, OUT_DIR / "arima_model.joblib")
    pd.DataFrame({"y_true": test, "y_pred": fc_series, "lower": conf_int[:,0], "upper": conf_int[:,1]}).to_csv(OUT_DIR / "arima_forecast_test.csv")
    return {"model": model, "mae": mae, "rmse": rmse, "mape": mape, "forecast": fc_series, "conf_int": conf_int}

def load_series():
    df = pd.read_csv(PROC_DIR / "combined_adj_and_returns.csv", index_col=0, parse_dates=True)
    # Use TSLA_adj
    return df["TSLA_adj"].dropna()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_end", default="2023-12-31")
    parser.add_argument("--test_end", default="2025-07-31")
    args = parser.parse_args()
    ts = load_series()
    res = train_and_evaluate(ts, train_end_date=args.train_end, test_end_date=args.test_end)
    print("Done. Metrics:", {k:res[k] for k in ["mae","rmse","mape"]})
