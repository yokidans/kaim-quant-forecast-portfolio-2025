# src/models/lstm_model.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
PROC_DIR = Path("data/processed")
OUT_DIR = Path("models/lstm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def create_windows(series, lookback=60):
    X, y = [], []
    for i in range(len(series)-lookback):
        X.append(series[i:i+lookback])
        y.append(series[i+lookback])
    return np.array(X), np.array(y)

def train_lstm(series, lookback=60, epochs=50, batch_size=32, train_end="2023-12-31", test_end="2025-07-31"):
    dates = series.index
    ts = series.values.reshape(-1,1)
    scaler = MinMaxScaler()
    ts_scaled = scaler.fit_transform(ts)
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    # create windows
    X, y = create_windows(ts_scaled.flatten(), lookback=lookback)
    # recreate date index for y: y corresponds to dates[lookback:]
    y_dates = dates[lookback:]
    # split by date
    train_mask = y_dates <= pd.to_datetime(train_end)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]
    logging.info("LSTM train size %d test size %d", len(X_train), len(X_test))
    # reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    # model
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ckpt = ModelCheckpoint(OUT_DIR / "best_lstm.h5", save_best_only=True, monitor='val_loss')
    model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[es, ckpt], verbose=2)
    # predict test
    y_pred_scaled = model.predict(X_test).flatten()
    # inverse scale
    scaler = joblib.load(OUT_DIR / "scaler.joblib")
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100
    logging.info("LSTM MAE=%f RMSE=%f MAPE=%f", mae, rmse, mape)
    model.save(OUT_DIR / "lstm_model.h5")
    # save test predictions with dates
    dates_test = y_dates[~train_mask]
    pd.DataFrame({"y_true": y_test_inv, "y_pred": y_pred_inv}, index=dates_test).to_csv(OUT_DIR / "lstm_forecast_test.csv")
    return {"mae": mae, "rmse": rmse, "mape": mape, "model": model}

def load_series():
    df = pd.read_csv(PROC_DIR / "combined_adj_and_returns.csv", index_col=0, parse_dates=True)
    return df["TSLA_adj"].dropna()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    series = load_series()
    res = train_lstm(series, lookback=args.lookback, epochs=args.epochs)
    print("LSTM results:", {k:res[k] for k in ["mae","rmse","mape"]})
