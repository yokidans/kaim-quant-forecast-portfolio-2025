import sys
import os
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import joblib
import yfinance as yf
import talib
import pickle
import json
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from scipy.stats import norm, t

# Use importlib.metadata to avoid pkg_resources deprecation
try:
    from importlib.metadata import version as get_version
except ImportError:
    from importlib_metadata import version as get_version

# Ensure directories exist
required_dirs = ["logs", "models", "reports/figures", "data/processed"]
for d in required_dirs:
    os.makedirs(d, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(Path("logs/quant_arima.log")), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None


class QuantitativeARIMAModel:
    def __init__(self, ticker="TSLA"):
        self.ticker = ticker
        self.arima_model = None
        self.garch_model = None
        self.prophet_model = None
        self.lstm_model = None
        self.xgb_model = None
        self.results = {}
        self.feature_importance = None
        self.xgb_n_features = None
        self.xgb_lookback = None

    def load_data(self, fallback_days=365 * 5):
        """Load data from local or Yahoo Finance."""
        try:
            local_path = Path("data/processed/combined_adj_and_returns.csv")
            if local_path.exists():
                df = pd.read_csv(local_path, index_col=0, parse_dates=True)
                if f"{self.ticker}_adj" in df.columns:
                    ts = df[[f"{self.ticker}_adj"]].copy()
                    return self._add_technical_indicators(ts)

            logger.warning("Local data not found, using Yahoo Finance...")
            data = yf.download(self.ticker, period=f"{fallback_days}d", progress=False)
            if data.empty:
                raise ValueError("No data from Yahoo Finance")

            price_col = data["Adj Close"] if "Adj Close" in data.columns else data["Close"]
            ts = pd.DataFrame(price_col.rename(f"{self.ticker}_adj"))
            return self._add_technical_indicators(ts)

        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def _add_technical_indicators(self, ts):
        """Add technical indicators."""
        if isinstance(ts, pd.Series):
            ts = ts.to_frame(name="price")

        ts["returns"] = np.log(ts.iloc[:, 0] / ts.iloc[:, 0].shift(1))
        ts["volatility"] = ts["returns"].rolling(21).std() * np.sqrt(252)

        if "rsi" not in ts.columns:
            delta = ts.iloc[:, 0].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            ts["rsi"] = 100 - (100 / (1 + (avg_gain / avg_loss)))

        if "macd" not in ts.columns:
            ema12 = ts.iloc[:, 0].ewm(span=12).mean()
            ema26 = ts.iloc[:, 0].ewm(span=26).mean()
            ts["macd"] = ema12 - ema26

        if "bollinger_middle" not in ts.columns:
            ts["bollinger_middle"] = ts.iloc[:, 0].rolling(20).mean()
            ts["bollinger_upper"] = ts["bollinger_middle"] + (ts.iloc[:, 0].rolling(20).std() * 2)
            ts["bollinger_lower"] = ts["bollinger_middle"] - (ts.iloc[:, 0].rolling(20).std() * 2)

        ts = ts.dropna()
        return ts

    def prepare_data(self, df, train_end="2023-12-31", test_end="2025-07-31"):
        """Prepare train/test data."""
        train_end = pd.to_datetime(train_end)
        test_end = pd.to_datetime(test_end)

        df["returns"] = np.log(df.iloc[:, 0] / df.iloc[:, 0].shift(1))
        df["volatility"] = df["returns"].rolling(21).std() * np.sqrt(252)
        df["momentum"] = df.iloc[:, 0] / df.iloc[:, 0].shift(21) - 1

        if "rsi" not in df.columns:
            delta = df.iloc[:, 0].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            df["rsi"] = 100 - (100 / (1 + (avg_gain / avg_loss)))

        df = df.dropna()

        train = df[df.index <= train_end]
        test = df[(df.index > train_end) & (df.index <= test_end)]

        self._plot_features(df)
        self.train_data = train
        self.test_data = test
        return train, test

    def _plot_features(self, df):
        """Plot indicators."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(f"{self.ticker} Technical Indicators", y=1.02)

        axes[0, 0].plot(df.iloc[:, 0])
        axes[0, 0].set_title("Price")

        if "rsi" in df.columns:
            axes[0, 1].plot(df["rsi"])
            axes[0, 1].axhline(30, color="r", linestyle="--")
            axes[0, 1].axhline(70, color="r", linestyle="--")
            axes[0, 1].set_title("RSI")

        if "macd" in df.columns:
            axes[1, 0].plot(df["macd"])
            axes[1, 0].set_title("MACD")

        bb_cols = ["bollinger_upper", "bollinger_middle", "bollinger_lower"]
        if all(col in df.columns for col in bb_cols):
            axes[1, 1].plot(df[bb_cols])
            axes[1, 1].set_title("Bollinger Bands")

        if "volatility" in df.columns:
            axes[2, 1].plot(df["volatility"])
            axes[2, 1].set_title("Volatility")

        plt.tight_layout()
        plt.savefig(Path("reports/figures/technical_indicators.png"))
        plt.close()

    def train_arima_garch(self, train_data):
        try:
            self.arima_model = auto_arima(
                train_data.iloc[:, 0],
                seasonal=True,
                m=1,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=True,
                n_jobs=-1,
                information_criterion="aic",
                test="adf",
                max_order=10,
            )

            residuals = train_data.iloc[:, 0] - self.arima_model.predict_in_sample()
            self.garch_model = arch_model(residuals, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")

            logger.info(f"ARIMA Model Summary:\n{self.arima_model.summary()}")
            logger.info(f"GARCH Model Summary:\n{self.garch_model.summary()}")

            return self.arima_model, self.garch_model

        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    def train_prophet(self, train_data):
        try:
            prophet_df = pd.DataFrame({"ds": train_data.index, "y": train_data.iloc[:, 0]})
            available_indicators = ["rsi", "macd", "bollinger_upper", "bollinger_lower", "volatility"]

            self.prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

            for indicator in available_indicators:
                if indicator in train_data.columns:
                    prophet_df[indicator] = train_data[indicator].values
                    self.prophet_model.add_regressor(indicator)
                    logger.info(f"Added {indicator} as Prophet regressor")

            prophet_df = prophet_df.dropna()  # NaN guard
            self.prophet_model.fit(prophet_df)
            return self.prophet_model
            
        except Exception as e:
            logger.error(f"Prophet training failed: {str(e)}")
            raise

    def train_lstm(self, train_data, lookback=60):
        try:
            price_data = train_data.iloc[:, 0].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(price_data)

            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i - lookback : i, 0])
                y.append(scaled_data[i, 0])
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            self.lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                LSTM(50),
                Dense(1)
            ])
            self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
            early_stop = EarlyStopping(monitor="val_loss", patience=5)

            self.lstm_model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=0)
            logger.info("LSTM model trained successfully")
            return self.lstm_model

        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            raise

    def train_xgboost(self, train_data, lookback=21):
        X, y = [], []
        n_features = train_data.shape[1]

        for i in range(lookback, len(train_data)):
            X.append(train_data.iloc[i - lookback : i].values.flatten())
            y.append(train_data.iloc[i, 0])

        X, y = np.array(X), np.array(y)

        self.xgb_model = XGBRegressor(
            n_estimators=1000, learning_rate=0.01, max_depth=5, subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=20, objective="reg:squarederror"
        )
        self.xgb_model.fit(X, y, eval_set=[(X, y)], verbose=0)

        self.xgb_n_features = n_features
        self.xgb_lookback = lookback
        return self.xgb_model

    def ensemble_predict(self, test_data, train_data, lookback=60):
        try:
            arima_fc = self.arima_model.predict(n_periods=len(test_data))

            garch_var = self.garch_model.forecast(horizon=len(test_data)).variance.iloc[-1].values

            future = self.prophet_model.make_future_dataframe(
                periods=len(test_data),
                include_history=False
            )

            # Add all regressors that Prophet was trained with
            for indicator in ["rsi", "macd", "bollinger_upper", "bollinger_lower", "volatility"]:
                if indicator in test_data.columns:
                    future[indicator] = test_data[indicator].values
                else:
                    # If missing, fill with mean or 0 to avoid Prophet errors
                    logger.warning(f"{indicator} missing in test data, filling with mean value")
                    future[indicator] = test_data[indicator].mean() if indicator in test_data else 0

            prophet_fc = self.prophet_model.predict(future)["yhat"].values

            scaler = MinMaxScaler()
            full_data = pd.concat([train_data, test_data])
            scaled_data = scaler.fit_transform(full_data.iloc[:, 0].values.reshape(-1, 1))

            X_test = []
            for i in range(len(train_data), len(full_data)):
                X_test.append(scaled_data[i - lookback : i])
            X_test = np.array(X_test)
            lstm_fc = self.lstm_model.predict(X_test).flatten()
            lstm_fc = scaler.inverse_transform(lstm_fc.reshape(-1, 1)).flatten()

            X_test_xgb = []
            for i in range(len(train_data), len(full_data)):
                X_test_xgb.append(full_data.iloc[i - self.xgb_lookback : i].values.flatten())
            X_test_xgb = np.array(X_test_xgb)
            xgb_fc = self.xgb_model.predict(X_test_xgb)

            min_len = min(len(arima_fc), len(prophet_fc), len(lstm_fc), len(xgb_fc))
            arima_fc, prophet_fc, lstm_fc, xgb_fc = arima_fc[-min_len:], prophet_fc[-min_len:], lstm_fc[-min_len:], xgb_fc[-min_len:]
            test_data = test_data.iloc[-min_len:]

            weights = np.array([0.4, 0.3, 0.2, 0.1])
            ensemble_fc = (weights[0] * arima_fc + weights[1] * prophet_fc + weights[2] * lstm_fc + weights[3] * xgb_fc)

            return {
                "arima": arima_fc, "prophet": prophet_fc, "lstm": lstm_fc, "xgb": xgb_fc,
                "ensemble": ensemble_fc, "garch_var": garch_var, "weights": weights
            }

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            raise
        
    def save_all_models(self, models_dir: Path):
        """Save all trained models to disk."""
        if self.arima_model:
            joblib.dump(self.arima_model, models_dir / f"{self.ticker}_arima.pkl")
        if self.garch_model:
            with open(models_dir / f"{self.ticker}_garch.pkl", "wb") as f:
                pickle.dump(self.garch_model, f)
        if self.prophet_model:
            joblib.dump(self.prophet_model, models_dir / f"{self.ticker}_prophet.pkl")
        if self.lstm_model:
            self.lstm_model.save(models_dir / f"{self.ticker}_lstm.keras")
        if self.xgb_model:
            joblib.dump(self.xgb_model, models_dir / f"{self.ticker}_xgb.pkl")
        logger.info(f"✅ All models saved to {models_dir}")


def main(ticker, train_end, test_end):
    try:
        logger.info(f"Starting quantitative modeling for {ticker}")
        model = QuantitativeARIMAModel(ticker)
        df = model.load_data()
        train, test = model.prepare_data(df, train_end, test_end)

        if not train.columns.equals(test.columns):
            raise ValueError("Train/test data features do not match")

        model.train_arima_garch(train)
        model.train_prophet(train)
        model.train_lstm(train)
        model.train_xgboost(train)

        predictions = model.ensemble_predict(test, train, lookback=21)

        actual = test.iloc[:, 0].values[-len(predictions["ensemble"]):]
        metrics = {
            "mae": mean_absolute_error(actual, predictions["ensemble"]),
            "rmse": np.sqrt(mean_squared_error(actual, predictions["ensemble"])),
            "mape": mean_absolute_percentage_error(actual, predictions["ensemble"]) * 100,
            "direction_accuracy": np.mean(
                np.sign(np.diff(actual)) == np.sign(np.diff(predictions["ensemble"]))
            ) * 100
        }

        returns = np.log(test.iloc[:, 0] / test.iloc[:, 0].shift(1)).dropna().values
        risk = {
            "var_normal": float(norm.ppf(0.05) * np.sqrt(predictions["garch_var"][-1])),
            "var_t": float(t.ppf(0.05, df=5) * np.sqrt(predictions["garch_var"][-1])),
            "es": float(returns[returns < np.percentile(returns, 5)].mean())
        }

        # Save outputs
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        # Fix length mismatch for DataFrame
        max_len = max(len(v) for v in predictions.values())
        for k, v in predictions.items():
            # Convert to list and pad with NaN if shorter
            v_list = list(v)
            if len(v_list) < max_len:
                v_list += [np.nan] * (max_len - len(v_list))
            predictions[k] = v_list

        # Save predictions CSV
        pred_df = pd.DataFrame(predictions)
        pred_csv_path = models_dir / f"{ticker}_predictions.csv"
        pred_df.to_csv(pred_csv_path, index=False)

        # Save performance & risk metrics JSON
        metrics_json_path = models_dir / f"{ticker}_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump({"performance": metrics, "risk": risk}, f, indent=4)

        # Save all models
        model.save_all_models(models_dir)

        # Log saved files
        logger.info(f"All models saved to: {models_dir}")
        logger.info(f"Predictions file: {pred_csv_path}")
        logger.info(f"Metrics file: {metrics_json_path}")
        return {"metrics": metrics, "risk_metrics": risk, "predictions": predictions}

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantitative ARIMA Modeling Pipeline")
    parser.add_argument("--ticker", default="TSLA", help="Stock ticker symbol")
    parser.add_argument("--train_end", default="2023-12-31", help="Training end date")
    parser.add_argument("--test_end", default="2025-07-31", help="Test end date")
    args = parser.parse_args()

    try:
        results = main(args.ticker, args.train_end, args.test_end)
        print("\n=== Final Results ===")
        print(pd.DataFrame(results["metrics"], index=[0]).T)
        print("\n=== Risk Metrics ===")
        print(pd.DataFrame(results["risk_metrics"], index=[0]).T)
        print(f"\nAll outputs saved to models directory")
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1)