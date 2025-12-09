#!/usr/bin/env python3
"""
Hyperparameter tuning script for the XGBoost model.

This script optimizes model hyperparameters using Optuna and saves the best configuration.
Run this to find optimal hyperparameters before using them in main.py.

Usage:
    uv run tune.py --trials 100
"""

import argparse
import json

import ccxt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from hyperparameter_tuning import get_tuned_model, tune_hyperparameters


def compute_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema(prices, period):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=period, adjust=False).mean()


def compute_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = compute_ema(prices, fast)
    ema_slow = compute_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def compute_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return upper_band, lower_band, bb_position


def main():
    parser = argparse.ArgumentParser(
        description="Tune XGBoost hyperparameters for cryptocurrency prediction"
    )
    parser.add_argument(
        "--trials", type=int, default=100, help="Number of optimization trials (default: 100)"
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--save", action="store_true", help="Save best parameters to file")
    parser.add_argument(
        "--symbol",
        type=str,
        default="HYPE/USDT",
        help="Trading pair symbol (e.g., BTC/USDT, ETH/USDT)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="hyperliquid",
        help="Exchange name (e.g., binance, coinbase, kraken)",
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="Number of days of historical data (default: 200)"
    )

    args = parser.parse_args()

    symbol = args.symbol
    exchange_name = args.exchange.lower()
    limit = args.limit

    print("=" * 60)
    print(f"XGBoost Hyperparameter Tuning for {symbol}")
    print("=" * 60)
    print()

    # Fetch data with rate limiting enabled
    print(f"Fetching {symbol} data from {exchange_name}...")
    try:
        if exchange_name == "hyperliquid":
            exchange = ccxt.hyperliquid({"enableRateLimit": True})
        elif exchange_name == "binance":
            exchange = ccxt.binance({"enableRateLimit": True})
        elif exchange_name == "coinbase":
            exchange = ccxt.coinbase({"enableRateLimit": True})
        elif exchange_name == "kraken":
            exchange = ccxt.kraken({"enableRateLimit": True})
        elif exchange_name == "bybit":
            exchange = ccxt.bybit({"enableRateLimit": True})
        elif exchange_name == "okx":
            exchange = ccxt.okx({"enableRateLimit": True})
        else:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({"enableRateLimit": True})
    except AttributeError:
        print(f"Error: Exchange '{exchange_name}' not supported")
        return
    except Exception as e:
        print(f"Error initializing exchange: {e}")
        return

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, "1d", limit=limit)
        if len(ohlcv) < 50:
            print(f"Error: Insufficient data. Got {len(ohlcv)} days, need at least 50")
            return
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Calculate features
    df["rsi"] = compute_rsi(df["close"], 14)

    # MACD indicators
    macd_line, signal_line, macd_histogram = compute_macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = macd_histogram

    # Bollinger Bands
    upper_band, lower_band, bb_position = compute_bollinger_bands(df["close"])
    df["bb_upper"] = upper_band
    df["bb_lower"] = lower_band
    df["bb_position"] = bb_position

    # EMA indicators
    df["ema_9"] = compute_ema(df["close"], 9)
    df["ema_21"] = compute_ema(df["close"], 21)
    df["ema_50"] = compute_ema(df["close"], 50)

    # Price relative to EMAs
    df["price_above_ema9"] = (df["close"] > df["ema_9"]).astype(int)
    df["price_above_ema21"] = (df["close"] > df["ema_21"]).astype(int)
    df["price_above_ema50"] = (df["close"] > df["ema_50"]).astype(int)

    # Volume analysis
    df["vol_change"] = df["volume"].pct_change().fillna(0)
    df["vol_sma_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()

    # Price momentum
    df["price_change"] = df["close"].pct_change().shift(-1)
    df["price_momentum_5d"] = df["close"].pct_change(5)

    # Manual inputs
    df["macro_score"] = 0.6
    df["unlock_pressure"] = 0.15

    # Regime Classification Target (0=Bull, 1=Ranging, 2=Bear, 3=HighVol)
    def classify_regime(data):
        """Classify market regime for tuning."""
        regimes = pd.Series(index=data.index, dtype=int)
        weekly_return = data["close"].pct_change(7).shift(-7)
        volatility = data["close"].pct_change().rolling(14).std().shift(-7)
        momentum_3d = data["close"].pct_change(3).shift(-7)

        # High Vol
        high_vol = (volatility > volatility.quantile(0.75)) & (abs(weekly_return) > 0.15)
        regimes.loc[high_vol] = 3

        # Bull
        bull = (weekly_return > 0.10) & (momentum_3d > 0.03) & (regimes != 3)
        regimes.loc[bull] = 0

        # Bear
        bear = (weekly_return < -0.05) & (momentum_3d < -0.02) & (regimes != 3)
        regimes.loc[bear] = 2

        # Ranging
        ranging = (abs(weekly_return) <= 0.03) & (volatility <= volatility.quantile(0.50))
        regimes.loc[ranging & (regimes != 3)] = 1

        regimes.fillna(1, inplace=True)
        return regimes.astype(int)

    df["target"] = classify_regime(df)

    # Drop NaNs
    df = df.dropna()

    print(f"Data points: {len(df)}")
    print()

    # Prepare training data
    feature_columns = [
        "rsi",
        "macd",
        "macd_histogram",
        "bb_position",
        "price_above_ema9",
        "price_above_ema21",
        "price_above_ema50",
        "vol_change",
        "vol_sma_ratio",
        "price_momentum_5d",
        "macro_score",
        "unlock_pressure",
    ]
    X = df[feature_columns]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print()

    # Run optimization
    print(f"Starting hyperparameter optimization ({args.trials} trials, {args.folds}-fold CV)...")
    print()

    best_params = tune_hyperparameters(
        X_train, y_train, n_trials=args.trials, n_splits=args.folds, verbose=True
    )

    print()
    print("=" * 60)

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    model = get_tuned_model(X_train, y_train, best_params)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print()

    # Save if requested
    if args.save:
        safe_symbol = symbol.replace("/", "_")
        filename = f"best_hyperparameters_{safe_symbol}.json"
        with open(filename, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"✓ Best parameters saved to {filename}")
        print(f"  Optimized specifically for {symbol}")
        print()
        print("To use these parameters, run:")
        print(f"  uv run main.py --symbol {symbol} --exchange {exchange_name}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()
