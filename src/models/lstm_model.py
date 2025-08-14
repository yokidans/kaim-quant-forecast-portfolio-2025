import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------- Data Loader ----------------
def load_data(file_path, lookback=60):
    df = pd.read_csv(file_path)
    if 'Adj Close' not in df.columns:
        raise ValueError(f"CSV missing 'Adj Close' column. Found: {list(df.columns)}")

    data = df['Adj Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    x, y = [], []
    for i in range(lookback, len(scaled_data)):
        x.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])

    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # [samples, time_steps, features]
    return x, y, scaler, data

# ---------------- Model Builder ----------------
def build_lstm_model(lookback):
    model = Sequential()
    model.add(Input(shape=(lookback, 1)))  # No warning now
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ---------------- Main Training ----------------
def main(args):
    # Load dataset
    file_path = os.path.join("data", "processed", f"{args.ticker}_processed.csv")
    logging.info(f"Loading data from {file_path}")
    x, y, scaler, raw_data = load_data(file_path, lookback=args.lookback)

    # Train/test split
    train_size = int(len(x) * 0.84)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    logging.info(f"LSTM train size {len(x_train)} test size {len(x_test)}")

    # Build model
    model = build_lstm_model(args.lookback)

    # Callbacks
    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join("models", f"{args.ticker}_best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True)
    ]

    # Train
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=2
    )

    # Predict
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Metrics
    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
    mape = np.mean(np.abs((y_test_actual - predictions) / y_test_actual)) * 100

    logging.info(f"LSTM MAE={mae:.6f} RMSE={rmse:.6f} MAPE={mape:.6f}")
    logging.info(f"Best model saved at: {checkpoint_path}")

    # Save final model (optional)
    final_model_path = os.path.join("models", f"{args.ticker}_final_model.keras")
    model.save(final_model_path)
    logging.info(f"Final model saved at: {final_model_path}")

    # Return metrics
    return {"mae": mae, "rmse": rmse, "mape": mape}

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="TSLA", help="Ticker symbol")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    args = parser.parse_args()

    results = main(args)
    print(f"LSTM results: {results}")
