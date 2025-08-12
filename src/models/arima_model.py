import sys
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
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import joblib
import yfinance as yf
import talib
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor
from scipy.stats import norm, t

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path("logs/quant_arima.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
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
        
    def load_data(self, fallback_days=365*5):
        """Load data with technical indicators and fallback to Yahoo Finance"""
        try:
            df = pd.read_csv(Path("data/processed/quant_data.csv"), 
                           index_col=0, parse_dates=True)
            if f"{self.ticker}_adj" not in df.columns:
                raise ValueError(f"{self.ticker}_adj column not found")
                
            ts = df[f"{self.ticker}_adj"].dropna()
            
            # Add technical indicators
            ts_df = pd.DataFrame(ts)
            ts_df['rsi'] = talib.RSI(ts.values, timeperiod=14)
            ts_df['macd'], _, _ = talib.MACD(ts.values)
            ts_df['bollinger_upper'], ts_df['bollinger_middle'], ts_df['bollinger_lower'] = \
                talib.BBANDS(ts.values, timeperiod=20)
            ts_df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            
            return ts_df.dropna()
            
        except Exception as e:
            logger.warning(f"Local data load failed: {str(e)}. Using Yahoo Finance...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=fallback_days)
            data = yf.download(self.ticker, start=start_date, end=end_date)
            
            # Create technical indicators
            data['rsi'] = talib.RSI(data['Adj Close'], timeperiod=14)
            data['macd'], _, _ = talib.MACD(data['Adj Close'])
            data['bollinger_upper'], data['bollinger_middle'], data['bollinger_lower'] = \
                talib.BBANDS(data['Adj Close'], timeperiod=20)
            data['atr'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
            
            return data[['Adj Close', 'rsi', 'macd', 'bollinger_upper', 
                        'bollinger_middle', 'bollinger_lower', 'atr']].dropna()
    
    def prepare_data(self, df, train_end="2023-12-31", test_end="2025-07-31"):
        """Advanced data preparation with feature engineering"""
        train_end = pd.to_datetime(train_end)
        test_end = pd.to_datetime(test_end)
        
        # Create features
        df['returns'] = np.log(df.iloc[:,0]/df.iloc[:,0].shift(1))
        df['volatility'] = df['returns'].rolling(21).std() * np.sqrt(252)
        df['momentum'] = df.iloc[:,0]/df.iloc[:,0].shift(21) - 1
        
        # Train-test split
        train = df[df.index <= train_end]
        test = df[(df.index > train_end) & (df.index <= test_end)]
        
        # Generate plots
        self._plot_features(df)
        
        return train, test
    
    def _plot_features(self, df):
        """Visualize technical indicators"""
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 2, 1)
        df.iloc[:,0].plot(title=f"{self.ticker} Price")
        
        plt.subplot(3, 2, 2)
        df['rsi'].plot(title='RSI')
        plt.axhline(30, color='r', linestyle='--')
        plt.axhline(70, color='r', linestyle='--')
        
        plt.subplot(3, 2, 3)
        df['macd'].plot(title='MACD')
        
        plt.subplot(3, 2, 4)
        df[['bollinger_upper', 'bollinger_middle', 'bollinger_lower']].plot(title='Bollinger Bands')
        
        plt.subplot(3, 2, 5)
        df['atr'].plot(title='ATR')
        
        plt.subplot(3, 2, 6)
        df['volatility'].plot(title='Volatility')
        
        plt.tight_layout()
        plt.savefig(Path("reports/figures/technical_indicators.png"))
        plt.close()
    
    def train_arima_garch(self, train_data):
        """Bayesian-optimized ARIMA-GARCH training"""
        # Bayesian optimization for ARIMA
        param_space = {
            'p': Integer(0, 5),
            'd': Integer(0, 2),
            'q': Integer(0, 5),
            'seasonal': Categorical([True, False]),
            'm': Integer(1, 12)
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        opt = BayesSearchCV(
            auto_arima(
                train_data.iloc[:,0],
                stepwise=False,
                suppress_warnings=True,
                information_criterion='aic',
                n_jobs=-1
            ),
            param_space,
            n_iter=50,
            cv=tscv,
            scoring='neg_mean_squared_error'
        )
        
        opt.fit(train_data.iloc[:,0])
        self.arima_model = opt.best_estimator_
        
        # Train GARCH on residuals
        residuals = train_data.iloc[:,0] - self.arima_model.predict_in_sample()
        self.garch_model = arch_model(
            residuals,
            vol='Garch',
            p=1,
            q=1,
            dist='skewt'  # Skewed Student's t-distribution
        ).fit(disp='off')
        
        logger.info(f"ARIMA Model Summary:\n{self.arima_model.summary()}")
        logger.info(f"GARCH Model Summary:\n{self.garch_model.summary()}")
        
        return self.arima_model, self.garch_model
    
    def train_prophet(self, train_data):
        """Facebook Prophet model with custom seasonality"""
        df = train_data.iloc[:,0].reset_index()
        df.columns = ['ds', 'y']
        
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        
        # Add technical indicators as regressors
        for col in train_data.columns[1:]:
            self.prophet_model.add_regressor(col)
        
        self.prophet_model.fit(df)
        return self.prophet_model
    
    def train_lstm(self, train_data, lookback=60):
        """LSTM neural network for comparison"""
        from sklearn.preprocessing import MinMaxScaler
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(train_data)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        
        # Build LSTM
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            LSTM(50),
            Dense(1)
        ])
        
        self.lstm_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
        
        self.lstm_model.fit(
            X, y,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        return self.lstm_model
    
    def train_xgboost(self, train_data, lookback=21):
        """XGBoost model with feature importance"""
        X, y = [], []
        for i in range(lookback, len(train_data)):
            X.append(train_data.iloc[i-lookback:i].values.flatten())
            y.append(train_data.iloc[i, 0])
        X, y = np.array(X), np.array(y)
        
        self.xgb_model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
            objective='reg:squarederror'
        )
        
        self.xgb_model.fit(
            X, y,
            eval_set=[(X, y)],
            verbose=0
        )
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': [f"lag_{i}_feat_{j}" 
                       for i in range(lookback) 
                       for j in range(train_data.shape[1])],
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.xgb_model
    
    def ensemble_predict(self, test_data):
        """Weighted ensemble prediction"""
        # ARIMA prediction
        arima_fc, arima_ci = self.arima_model.predict(
            n_periods=len(test_data),
            return_conf_int=True
        )
        
        # GARCH volatility
        garch_var = self.garch_model.forecast(
            horizon=len(test_data)
        ).variance.iloc[-1].values
        
        # Prophet prediction
        future = self.prophet_model.make_future_dataframe(
            periods=len(test_data),
            include_history=False
        )
        for col in test_data.columns[1:]:
            future[col] = test_data[col].values
        prophet_fc = self.prophet_model.predict(future)['yhat'].values
        
        # LSTM prediction
        scaler = MinMaxScaler()
        full_data = pd.concat([train_data, test_data])
        scaled_data = scaler.fit_transform(full_data.iloc[:,0].values.reshape(-1, 1))
        
        X_test = []
        for i in range(len(train_data), len(full_data)):
            X_test.append(scaled_data[i-lookback:i])
        X_test = np.array(X_test)
        
        lstm_fc = self.lstm_model.predict(X_test).flatten()
        lstm_fc = scaler.inverse_transform(lstm_fc.reshape(-1, 1)).flatten()
        
        # XGBoost prediction
        X_test_xgb = []
        for i in range(len(train_data), len(full_data)):
            X_test_xgb.append(full_data.iloc[i-lookback:i].values.flatten())
        X_test_xgb = np.array(X_test_xgb)
        
        xgb_fc = self.xgb_model.predict(X_test_xgb)
        
        # Weighted ensemble (dynamic weights based on recent performance)
        weights = np.array([0.4, 0.3, 0.2, 0.1])  # ARIMA, Prophet, LSTM, XGBoost
        ensemble_fc = (weights[0]*arima_fc + weights[1]*prophet_fc + 
                      weights[2]*lstm_fc + weights[3]*xgb_fc)
        
        return {
            'arima': arima_fc,
            'prophet': prophet_fc,
            'lstm': lstm_fc,
            'xgb': xgb_fc,
            'ensemble': ensemble_fc,
            'garch_var': garch_var,
            'weights': weights
        }
    
    def risk_metrics(self, returns, var):
        """Calculate Value-at-Risk and Expected Shortfall"""
        # Parametric VaR (normal and t-dist)
        var_normal = norm.ppf(0.05) * np.sqrt(var)
        var_t = t.ppf(0.05, df=5) * np.sqrt(var)
        
        # Historical ES
        es = returns[returns < np.percentile(returns, 5)].mean()
        
        return {
            'var_normal': var_normal,
            'var_t': var_t,
            'es': es
        }
    
    def save_models(self):
        """Save all model artifacts"""
        import json
        
        artifacts = {
            'arima': joblib.dump(self.arima_model, Path("models/arima_model.joblib")),
            'garch': joblib.dump(self.garch_model, Path("models/garch_model.joblib")),
            'prophet': joblib.dump(self.prophet_model, Path("models/prophet_model.joblib")),
            'lstm': self.lstm_model.save(Path("models/lstm_model.h5")),
            'xgb': joblib.dump(self.xgb_model, Path("models/xgb_model.joblib")),
            'feature_importance': self.feature_importance.to_dict(),
            'metrics': self.results
        }
        
        with open(Path("models/model_artifacts.json"), 'w') as f:
            json.dump(artifacts, f)
    
    def backtest(self, train_data, test_data, initial_capital=100000):
        """Trading strategy backtest"""
        predictions = self.ensemble_predict(test_data)
        returns = test_data.iloc[:,0].pct_change().dropna()
        
        # Simple long/short strategy
        positions = np.where(predictions['ensemble'].diff() > 0, 1, -1)
        strategy_returns = positions * returns
        
        # Performance metrics
        sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        sortino = np.sqrt(252) * strategy_returns.mean() / \
                 strategy_returns[strategy_returns < 0].std()
        max_drawdown = (strategy_returns.cumsum().expanding().max() - 
                       strategy_returns.cumsum()).max()
        
        # Equity curve
        equity = initial_capital * (1 + strategy_returns).cumprod()
        
        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'equity': equity,
            'positions': positions,
            'returns': strategy_returns
        }

def main(ticker, train_end, test_end):
    """End-to-end quantitative modeling pipeline"""
    try:
        logger.info(f"Starting quantitative modeling for {ticker}")
        
        model = QuantitativeARIMAModel(ticker)
        
        # Data loading and preparation
        df = model.load_data()
        train, test = model.prepare_data(df, train_end, test_end)
        
        # Model training
        model.train_arima_garch(train)
        model.train_prophet(train)
        model.train_lstm(train)
        model.train_xgboost(train)
        
        # Ensemble prediction
        predictions = model.ensemble_predict(test)
        
        # Evaluation
        actual = test.iloc[:,0].values
        metrics = {
            'mae': mean_absolute_error(actual, predictions['ensemble']),
            'rmse': np.sqrt(mean_squared_error(actual, predictions['ensemble'])),
            'mape': mean_absolute_percentage_error(actual, predictions['ensemble']) * 100,
            'direction_accuracy': np.mean(np.sign(actual[1:]-actual[:-1]) == 
                                        np.sign(predictions['ensemble'][1:]-predictions['ensemble'][:-1])) * 100
        }
        
        # Risk analysis
        returns = np.log(test.iloc[:,0]/test.iloc[:,0].shift(1)).dropna().values
        risk = model.risk_metrics(returns, predictions['garch_var'])
        
        # Backtest
        backtest_results = model.backtest(train, test)
        
        # Save results
        model.results = {
            'metrics': metrics,
            'risk_metrics': risk,
            'backtest': backtest_results,
            'predictions': predictions
        }
        model.save_models()
        
        # Generate reports
        model.generate_report()
        
        logger.info("Modeling completed successfully")
        logger.info(f"Performance Metrics: {metrics}")
        logger.info(f"Risk Metrics: {risk}")
        
        return model.results
        
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
        print(pd.DataFrame(results['metrics'], index=[0]).T)
        print("\n=== Risk Metrics ===")
        print(pd.DataFrame(results['risk_metrics'], index=[0]).T)
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        sys.exit(1)