import argparse
import json
import os

import ccxt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from risk_management import RiskManager, calculate_volatility, calculate_win_rate_and_ratio


def get_default_hyperparameters(symbol):
    """
    Get smart default hyperparameters based on asset class.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')

    Returns:
        Dictionary of default hyperparameters
    """
    # Major caps (BTC, ETH) - less volatile, more stable
    if any(x in symbol.upper() for x in ["BTC", "ETH"]):
        return {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0,
        }
    # Mid-caps (SOL, BNB, ADA, etc.) - medium volatility
    elif any(x in symbol.upper() for x in ["SOL", "BNB", "ADA", "AVAX", "DOT"]):
        return {
            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 2,
            "gamma": 1,
        }
    # Low-caps - high volatility
    else:
        return {
            "n_estimators": 300,
            "max_depth": 9,
            "learning_rate": 0.20,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "min_child_weight": 1,
            "gamma": 2,
        }


def load_hyperparameters(symbol="HYPE/USDT"):
    """
    Load hyperparameters from symbol-specific JSON file if it exists.
    Falls back to generic file, then to smart defaults.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')

    Returns:
        Dictionary of hyperparameters
    """
    # Try symbol-specific file first
    safe_symbol = symbol.replace("/", "_")
    symbol_filepath = f"best_hyperparameters_{safe_symbol}.json"

    if os.path.exists(symbol_filepath):
        with open(symbol_filepath, "r") as f:
            params = json.load(f)
        print(f"✓ Loaded {symbol} hyperparameters from {symbol_filepath}")
        return params

    # Fall back to generic file
    generic_filepath = "best_hyperparameters.json"
    if os.path.exists(generic_filepath):
        with open(generic_filepath, "r") as f:
            params = json.load(f)
        print(f"⚠ Using generic hyperparameters from {generic_filepath}")
        print(f"  (Run 'tune.py --symbol {symbol} --save' for optimized {symbol} parameters)")
        return params

    # Use smart defaults
    print(f"⚠ No hyperparameter files found, using smart defaults for {symbol}")
    print(f"  (Run 'tune.py --symbol {symbol} --trials 100 --save' to optimize)")
    return get_default_hyperparameters(symbol)


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
    bb_position = (prices - lower_band) / (upper_band - lower_band)  # 0-1 position in band
    return upper_band, lower_band, bb_position


def walk_forward_backtest(
    df,
    feature_columns,
    symbol="HYPE/USDT",
    initial_capital=10000,
    trading_fee_bps=5,
    risk_manager=None,
    train_window=120,  # Days to use for training
    retrain_frequency=30,  # Retrain every N days
):
    """
    Walk-forward backtest with periodic model retraining.

    Args:
        df: Full dataframe with features and target
        feature_columns: List of feature column names
        symbol: Trading pair symbol for loading hyperparameters
        initial_capital: Starting capital in USD
        trading_fee_bps: Trading fee in basis points
        risk_manager: RiskManager instance
        train_window: Number of days to use for training
        retrain_frequency: How often to retrain the model (in days)

    Returns:
        Dictionary with backtest metrics
    """
    if risk_manager is None:
        risk_manager = RiskManager()
    risk_manager.reset()
    risk_manager.peak_equity = initial_capital

    df = df.copy().reset_index(drop=True)

    # Initialize tracking variables
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_prob = 0
    position_size = 0
    entry_fee = 0
    entry_idx = 0
    trades = []
    equity_curve = [capital]
    total_fees = 0
    fee_rate = trading_fee_bps / 10000

    # Calculate volatility for position sizing
    volatility = calculate_volatility(df["close"])

    model = None
    last_train_idx = 0
    num_retrains = 0

    # Walk through the dataframe
    for i in range(train_window, len(df)):
        # Retrain model periodically
        if model is None or (i - last_train_idx) >= retrain_frequency:
            # Use sliding window for training
            train_start = max(0, i - train_window)
            train_end = i

            X_train = df.loc[train_start : train_end - 1, feature_columns]
            y_train = df.loc[train_start : train_end - 1, "target"]

            # Train new model with optimized hyperparameters
            hyperparams = load_hyperparameters(symbol)
            hyperparams["random_state"] = 42  # Always use fixed seed for reproducibility
            # Multi-class classification setup (4 regimes)
            hyperparams["objective"] = "multi:softmax"
            hyperparams["num_class"] = 4
            model = XGBClassifier(**hyperparams)
            model.fit(X_train, y_train)
            last_train_idx = i
            num_retrains += 1

        # Get current prediction and probabilities
        X_current = df.loc[i:i, feature_columns]
        regime_prediction = model.predict(X_current)[0]  # 0=Bull, 1=Ranging, 2=Bear, 3=HighVol
        regime_probs = model.predict_proba(X_current)[0]  # Probabilities for all 4 classes

        # Get confidence for the predicted regime
        regime_confidence = regime_probs[regime_prediction]

        current_price = df.loc[i, "close"]

        # Check for stop-loss or take-profit exits first
        if position == 1:
            should_exit, exit_reason = risk_manager.should_exit_trade(
                current_price, entry_price, "long"
            )
            if should_exit:
                # Exit position due to risk management
                exit_price = current_price
                position_value = position_size * exit_price
                exit_fee = position_size * exit_price * fee_rate
                total_fees += exit_fee
                position_value -= exit_fee

                gross_pnl = position_size * (exit_price - entry_price)
                net_pnl = gross_pnl - (entry_fee + exit_fee)

                capital += position_value

                trades.append(
                    {
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "fees": entry_fee + exit_fee,
                        "position_size": position_size,
                        "entry_prob": entry_prob,
                        "exit_reason": exit_reason,
                    }
                )

                position = 0

        # ===================================================================
        # REGIME-BASED TRADING LOGIC
        # ===================================================================
        # Entry signals based on predicted regime
        # 0=Bull: Long with larger size, 1=Ranging: Small long, 2=Bear: Stay cash, 3=HighVol: Small position with wide stops

        if position == 0:  # No position currently
            should_enter = False
            regime_multiplier = 1.0  # Position size multiplier based on regime

            if regime_prediction == 0:  # Bull regime
                should_enter = regime_confidence > 0.50  # Enter if confident
                regime_multiplier = 1.2  # Larger position in bull market
            elif regime_prediction == 1:  # Ranging regime
                should_enter = regime_confidence > 0.55  # Be more selective
                regime_multiplier = 0.7  # Smaller position in ranging market
            elif regime_prediction == 3:  # High volatility regime
                should_enter = regime_confidence > 0.60  # Very selective
                regime_multiplier = 0.5  # Much smaller position in high vol
            # regime_prediction == 2 (Bear): stay_cash = True, no entry

            if should_enter:
                # Calculate historical win rate and ratio from past trades
                if len(trades) > 0:
                    trades_df = pd.DataFrame(trades)
                    win_rate, avg_win_loss_ratio = calculate_win_rate_and_ratio(trades_df)
                else:
                    win_rate, avg_win_loss_ratio = 0.5, 1.0

                # Calculate position size using risk manager
                base_position_size = risk_manager.calculate_position_size(
                    capital=capital,
                    entry_price=current_price,
                    volatility=volatility,
                    win_rate=win_rate,
                    avg_win_loss_ratio=avg_win_loss_ratio,
                )

                # Adjust for regime
                position_size_dollars = base_position_size * regime_multiplier

                if position_size_dollars > 0:
                    position_size = position_size_dollars / current_price
                    position = 1
                    entry_price = current_price
                    entry_prob = regime_confidence
                    entry_idx = i

                    entry_fee = position_size * entry_price * fee_rate
                    total_fees += entry_fee
                    capital -= entry_fee

        # Exit signals based on regime change
        elif position == 1:
            should_exit = False
            exit_reason = "signal"

            # Exit conditions:
            # 1. Regime switches to Bear (2)
            # 2. Last day of backtest
            # 3. Low confidence in any regime

            if regime_prediction == 2 and regime_confidence > 0.45:  # Bear regime
                should_exit = True
                exit_reason = "bear_regime"
            elif i == len(df) - 1:  # Last day
                should_exit = True
                exit_reason = "end_of_period"
            elif regime_confidence < 0.40:  # Low confidence in any regime
                should_exit = True
                exit_reason = "low_confidence"

            if should_exit:
                exit_price = current_price
                position_value = position_size * exit_price
                exit_fee = position_size * exit_price * fee_rate
                total_fees += exit_fee
                position_value -= exit_fee

                gross_pnl = position_size * (exit_price - entry_price)
                net_pnl = gross_pnl - (entry_fee + exit_fee)

                capital += position_value

                trades.append(
                    {
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "fees": entry_fee + exit_fee,
                        "position_size": position_size,
                        "entry_prob": entry_prob,
                        "exit_reason": exit_reason,
                    }
                )

                position = 0

        # Update drawdown tracking
        current_equity = capital
        if position == 1:
            unrealized_pnl = (current_price - entry_price) * position_size
            current_equity += unrealized_pnl

        risk_manager.update_drawdown(current_equity)
        equity_curve.append(current_equity)

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    total_return = (equity_curve[-1] - initial_capital) / initial_capital

    trades_df = pd.DataFrame(trades) if len(trades) > 0 else pd.DataFrame()

    if len(trades_df) > 0:
        winning_trades = len(trades_df[trades_df["net_pnl"] > 0])
        losing_trades = len(trades_df[trades_df["net_pnl"] <= 0])
        win_rate = winning_trades / len(trades_df)
        avg_win = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].mean() if winning_trades > 0 else 0
        avg_loss = (
            trades_df[trades_df["net_pnl"] <= 0]["net_pnl"].mean() if losing_trades > 0 else 0
        )
        total_fees_paid = trades_df["fees"].sum()
        profit_factor = (
            abs(
                trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum()
                / trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum()
            )
            if losing_trades > 0
            else float("inf")
        )
    else:
        winning_trades = losing_trades = win_rate = avg_win = avg_loss = 0
        total_fees_paid = 0
        profit_factor = 0

    total_pnl = equity_curve[-1] - initial_capital

    # Calculate Sharpe Ratio
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe_ratio = (
        np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
        if len(daily_returns) > 1
        else 0
    )

    # Calculate Max Drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = np.min(drawdown)

    return {
        "final_equity": equity_curve[-1],
        "total_return_pct": total_return * 100,
        "total_pnl": total_pnl,
        "total_fees": total_fees_paid,
        "num_trades": len(trades_df),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown * 100,
        "trades": trades_df,
        "retrains": num_retrains,
    }


def backtest_strategy(
    df_test,
    model,
    initial_capital=10000,
    trading_fee_bps=5,
    risk_manager=None,
):
    """
    Backtest the trading strategy with risk management.

    Args:
        df_test: Test dataframe with features and actual close prices
        model: Trained XGBoost model
        initial_capital: Starting capital in USD
        trading_fee_bps: Trading fee in basis points (5 bps = 0.05%)
        risk_manager: RiskManager instance for position sizing and risk control

    Returns:
        Dictionary with backtest metrics
    """
    df_test = df_test.copy()

    # Initialize risk manager if not provided
    if risk_manager is None:
        risk_manager = RiskManager()
    risk_manager.reset()  # Reset for new backtest
    risk_manager.peak_equity = initial_capital  # Initialize peak equity
    df_test = df_test.copy()

    # Generate predictions and probabilities
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
    X_test = df_test[feature_columns]
    # Regime predictions (0=Bull, 1=Ranging, 2=Bear, 3=HighVol)
    df_test["regime_prediction"] = model.predict(X_test)
    # Get probabilities for all 4 regimes
    regime_probs_all = model.predict_proba(X_test)
    df_test["regime_confidence"] = [
        probs[pred] for probs, pred in zip(regime_probs_all, df_test["regime_prediction"])
    ]

    # Initialize tracking variables
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long
    entry_price = 0
    entry_prob = 0
    position_size = 0
    stop_loss_price = 0
    take_profit_price = 0
    entry_fee = 0  # Initialize entry_fee
    entry_idx = 0  # Initialize entry_idx
    trades = []
    equity_curve = [capital]
    total_fees = 0

    fee_rate = trading_fee_bps / 10000

    # Calculate volatility for position sizing
    volatility = calculate_volatility(df_test["close"])

    # Walk through test set
    for i in range(len(df_test)):
        current_price = df_test["close"].iloc[i]

        # Check for stop-loss or take-profit exits first
        if position == 1:
            should_exit, exit_reason = risk_manager.should_exit_trade(
                current_price, entry_price, "long"
            )
            if should_exit:
                # Exit position due to risk management
                exit_price = current_price
                position_value = position_size * exit_price
                exit_fee = position_size * exit_price * fee_rate
                total_fees += exit_fee
                position_value -= exit_fee

                gross_pnl = position_size * (exit_price - entry_price)
                net_pnl = gross_pnl - (entry_fee + exit_fee)

                capital += position_value

                trades.append(
                    {
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "fees": entry_fee + exit_fee,
                        "position_size": position_size,
                        "entry_prob": entry_prob,
                        "exit_reason": exit_reason,
                    }
                )

                position = 0

        # ===================================================================
        # REGIME-BASED TEST SET TRADING
        # ===================================================================
        regime = df_test["regime_prediction"].iloc[i]  # 0=Bull, 1=Ranging, 2=Bear, 3=HighVol
        confidence = df_test["regime_confidence"].iloc[i]

        # Entry signals based on regime
        if position == 0:
            should_enter = False
            regime_multiplier = 1.0

            if regime == 0:  # Bull
                should_enter = confidence > 0.50
                regime_multiplier = 1.2
            elif regime == 1:  # Ranging
                should_enter = confidence > 0.55
                regime_multiplier = 0.7
            elif regime == 3:  # High Vol
                should_enter = confidence > 0.60
                regime_multiplier = 0.5

            if should_enter:
                # Calculate historical win rate and ratio from past trades
                if len(trades) > 0:
                    trades_df = pd.DataFrame(trades)
                    win_rate, avg_win_loss_ratio = calculate_win_rate_and_ratio(trades_df)
                else:
                    win_rate, avg_win_loss_ratio = 0.5, 1.0  # Default values

                # Calculate position size using risk manager
                base_position_size = risk_manager.calculate_position_size(
                    capital=capital,
                    entry_price=current_price,
                    volatility=volatility,
                    win_rate=win_rate,
                    avg_win_loss_ratio=avg_win_loss_ratio,
                )
                position_size_dollars = base_position_size * regime_multiplier

                if position_size_dollars > 0:  # Only enter if risk manager allows
                    position_size = position_size_dollars / current_price
                    position = 1
                    entry_price = current_price
                    entry_prob = confidence
                    entry_idx = i

                    # Calculate stop-loss and take-profit levels
                    stop_loss_price, take_profit_price = risk_manager.calculate_stop_levels(
                        entry_price, "long"
                    )

                    # Calculate entry fee
                    entry_fee = position_size * entry_price * fee_rate
                    total_fees += entry_fee
                    capital -= entry_fee

        # Exit signals based on regime change
        elif position == 1:
            should_exit = False
            exit_reason = "signal"

            if regime == 2 and confidence > 0.45:  # Bear
                should_exit = True
                exit_reason = "bear_regime"
            elif i == len(df_test) - 1:  # Last day
                should_exit = True
                exit_reason = "end_of_period"
            elif confidence < 0.40:  # Low confidence
                should_exit = True
                exit_reason = "low_confidence"

            if should_exit:
                exit_price = current_price
                position_value = position_size * exit_price
                exit_fee = position_size * exit_price * fee_rate
                total_fees += exit_fee
                position_value -= exit_fee

                gross_pnl = position_size * (exit_price - entry_price)
                net_pnl = gross_pnl - (entry_fee + exit_fee)

                capital += position_value

                trades.append(
                    {
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "fees": entry_fee + exit_fee,
                        "position_size": position_size,
                        "entry_prob": entry_prob,
                        "exit_reason": "signal",
                    }
                )

            position = 0

        # Update drawdown tracking
        current_equity = capital
        if position == 1:
            unrealized_pnl = (current_price - entry_price) * position_size
            current_equity += unrealized_pnl

        risk_manager.update_drawdown(current_equity)

        # Track equity
        equity_curve.append(current_equity)

    # Calculate metrics
    equity_curve = np.array(equity_curve)
    total_return = (equity_curve[-1] - initial_capital) / initial_capital

    trades_df = pd.DataFrame(trades) if len(trades) > 0 else pd.DataFrame()

    if len(trades_df) > 0:
        winning_trades = len(trades_df[trades_df["net_pnl"] > 0])
        losing_trades = len(trades_df[trades_df["net_pnl"] <= 0])
        win_rate = winning_trades / len(trades_df)
        avg_win = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].mean() if winning_trades > 0 else 0
        avg_loss = (
            trades_df[trades_df["net_pnl"] <= 0]["net_pnl"].mean() if losing_trades > 0 else 0
        )
        total_fees_paid = trades_df["fees"].sum()
        profit_factor = (
            abs(
                trades_df[trades_df["net_pnl"] > 0]["net_pnl"].sum()
                / trades_df[trades_df["net_pnl"] < 0]["net_pnl"].sum()
            )
            if losing_trades > 0
            else float("inf")
        )
    else:
        winning_trades = losing_trades = win_rate = avg_win = avg_loss = 0
        total_fees_paid = 0
        profit_factor = 0

    # Total P&L equals final equity minus initial capital
    # This accounts for all trades, fees, and cash remaining
    total_pnl = equity_curve[-1] - initial_capital

    # Calculate Sharpe Ratio (assuming daily returns)
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe_ratio = (
        np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
        if len(daily_returns) > 1
        else 0
    )

    # Calculate Max Drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = np.min(drawdown)

    return {
        "final_equity": equity_curve[-1],
        "total_return_pct": total_return * 100,
        "total_pnl": total_pnl,
        "total_fees": total_fees_paid,
        "num_trades": len(trades_df),
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate_pct": win_rate * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown * 100,
        "trades": trades_df,
    }


def main():
    """Main application entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Cryptocurrency price movement predictor using XGBoost and technical analysis"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="HYPE/USDT",
        help="Trading pair symbol (e.g., BTC/USDT, ETH/USDT, SOL/USDT)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="hyperliquid",
        help="Exchange name (e.g., binance, coinbase, kraken, hyperliquid)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of days of historical data to fetch (default: 200)",
    )

    args = parser.parse_args()

    symbol = args.symbol
    exchange_name = args.exchange.lower()
    limit = args.limit

    print("\n" + "=" * 60)
    print(f"CRYPTO PRICE PREDICTOR - {symbol}")
    print("=" * 60)
    print(f"Exchange: {exchange_name}")
    print(f"Historical Data: {limit} days")
    print("=" * 60)
    print()

    # Initialize exchange with rate limiting enabled
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
            # Try to load exchange dynamically
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({"enableRateLimit": True})
    except AttributeError:
        print(f"✗ Error: Exchange '{exchange_name}' not supported by CCXT")
        print("   Available exchanges: binance, coinbase, kraken, bybit, okx, hyperliquid")
        return
    except Exception as e:
        print(f"❌ Error initializing exchange: {e}")
        return

    # Fetch data
    print(f"Fetching {symbol} data from {exchange_name}...")
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, "1d", limit=limit)
        if len(ohlcv) < 50:
            print(f"❌ Error: Insufficient data. Got {len(ohlcv)} days, need at least 50")
            return
        print(f"✓ Fetched {len(ohlcv)} days of data")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        print(f"   Symbol '{symbol}' may not be available on {exchange_name}")
        return

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    print()

    # Technical Indicators
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
    df["bb_position"] = bb_position  # Where price is within the bands (0=lower, 1=upper)

    # EMA indicators
    df["ema_9"] = compute_ema(df["close"], 9)
    df["ema_21"] = compute_ema(df["close"], 21)
    df["ema_50"] = compute_ema(df["close"], 50)

    # Price relative to EMAs (momentum indicators)
    df["price_above_ema9"] = (df["close"] > df["ema_9"]).astype(int)
    df["price_above_ema21"] = (df["close"] > df["ema_21"]).astype(int)
    df["price_above_ema50"] = (df["close"] > df["ema_50"]).astype(int)

    # Volume analysis
    df["vol_change"] = df["volume"].pct_change().fillna(0)
    df["vol_sma_ratio"] = df["volume"] / df["volume"].rolling(window=20).mean()  # Volume vs average

    # Price momentum
    df["price_change"] = df["close"].pct_change().shift(-1)  # Next day return
    df["price_momentum_5d"] = df["close"].pct_change(5)  # 5-day momentum

    # Manual inputs (updated Dec 7, 2025)
    df["macro_score"] = 0.6  # Fed paused, inflation sticky
    df["unlock_pressure"] = 0.15  # Post-Nov cliff absorbed

    # ===================================================================
    # REGIME CLASSIFICATION TARGET (4 classes)
    # ===================================================================
    def classify_regime(data):
        """
        Classify market regime into 4 categories based on price action.

        Classes:
        0 = Bull Run: Strong uptrend (>10% weekly gain + upward momentum)
        1 = Ranging: Sideways movement (±3% weekly, low volatility)
        2 = Bear Market: Strong downtrend (>5% weekly loss + downward momentum)
        3 = High Volatility: Large swings regardless of direction

        Args:
            data: DataFrame with price data

        Returns:
            Series with regime labels (0-3)
        """
        regimes = pd.Series(index=data.index, dtype=int)

        # Calculate indicators for regime classification
        weekly_return = data["close"].pct_change(7).shift(-7)  # Forward 7-day return
        volatility = data["close"].pct_change().rolling(14).std().shift(-7)  # Forward volatility
        momentum_3d = data["close"].pct_change(3).shift(-7)  # 3-day momentum

        # High Volatility Regime (priority - check first)
        high_vol_condition = volatility > volatility.quantile(0.75)  # Top 25% volatility
        large_moves = abs(weekly_return) > 0.15  # Absolute move > 15%
        regimes.loc[high_vol_condition & large_moves] = 3

        # Bull Run Regime
        bull_condition = (
            (weekly_return > 0.10)  # >10% weekly gain
            & (momentum_3d > 0.03)  # Positive 3-day momentum
            & (regimes != 3)  # Not already classified as high vol
        )
        regimes.loc[bull_condition] = 0

        # Bear Market Regime
        bear_condition = (
            (weekly_return < -0.05)  # >5% weekly loss
            & (momentum_3d < -0.02)  # Negative 3-day momentum
            & (regimes != 3)  # Not already classified as high vol
        )
        regimes.loc[bear_condition] = 2

        # Ranging Regime (everything else)
        ranging_condition = (
            (abs(weekly_return) <= 0.03)  # ±3% movement
            & (volatility <= volatility.quantile(0.50))  # Below median volatility
            & (regimes != 3)  # Not already classified
        )
        regimes.loc[ranging_condition] = 1

        # Default to ranging for any unclassified
        regimes.fillna(1, inplace=True)
        regimes = regimes.astype(int)

        return regimes

    # Target: Regime classification (0=Bull, 1=Ranging, 2=Bear, 3=HighVol)
    df["target"] = classify_regime(df)

    # Drop NaNAs
    df = df.dropna()

    # Features for model
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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Load optimized hyperparameters from tune.py results
    hyperparams = load_hyperparameters(symbol)
    hyperparams["random_state"] = 42  # Always use fixed seed for reproducibility

    print("\nUsing hyperparameters:")
    for key, value in hyperparams.items():
        print(f"  {key}: {value}")
    print()

    # XGBoost model with optimized hyperparameters
    model = XGBClassifier(**hyperparams)
    model.fit(X_train, y_train)

    # Get test data with original indices preserved
    df_test = df.loc[X_test.index].copy()

    # Feature importance analysis
    feature_importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    # Run comprehensive backtest with risk management
    risk_manager = RiskManager(
        max_drawdown_pct=0.15,  # 15% max drawdown
        max_position_size_pct=0.20,  # 20% of capital per trade
        stop_loss_pct=0.03,  # 3% stop loss
        take_profit_pct=0.08,  # 8% take profit
        kelly_fraction=0.3,  # Use 30% of Kelly for moderate risk
    )

    # Standard backtest (train once)
    backtest_results = backtest_strategy(
        df_test, model, initial_capital=10000, trading_fee_bps=5, risk_manager=risk_manager
    )

    # Walk-forward backtest (retrain periodically)
    risk_manager_wf = RiskManager(
        max_drawdown_pct=0.15,
        max_position_size_pct=0.20,
        stop_loss_pct=0.03,
        take_profit_pct=0.08,
        kelly_fraction=0.3,
    )

    walkforward_results = walk_forward_backtest(
        df,
        feature_columns,
        symbol=symbol,
        initial_capital=10000,
        trading_fee_bps=5,
        risk_manager=risk_manager_wf,
        train_window=80,  # 80 days training (shorter for more test data)
        retrain_frequency=20,  # Retrain every 20 days
    )

    # Print standard backtest results
    print("\n" + "=" * 50)
    print("STANDARD BACKTEST (Train Once)")
    print("=" * 50)
    print("Initial Capital: $10,000")
    print(f"Final Equity: ${backtest_results['final_equity']:.2f}")
    print(f"Total Return: {backtest_results['total_return_pct']:.2f}%")
    print(f"Total P&L (net): ${backtest_results['total_pnl']:.2f}")
    print(f"Total Fees Paid: ${backtest_results['total_fees']:.2f}")
    print()
    print(f"Number of Trades: {backtest_results['num_trades']}")
    print(f"Winning Trades: {backtest_results['winning_trades']}")
    print(f"Losing Trades: {backtest_results['losing_trades']}")
    print(f"Win Rate: {backtest_results['win_rate_pct']:.1f}%")
    print()
    print(f"Avg Win: ${backtest_results['avg_win']:.2f}")
    print(f"Avg Loss: ${backtest_results['avg_loss']:.2f}")
    print(f"Profit Factor: {backtest_results['profit_factor']:.2f}x")
    print()
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
    print("=" * 50)

    # Print walk-forward backtest results
    print("\n" + "=" * 50)
    print("WALK-FORWARD BACKTEST (Retrain Every 20 Days)")
    print("=" * 50)
    print("Initial Capital: $10,000")
    print(f"Final Equity: ${walkforward_results['final_equity']:.2f}")
    print(f"Total Return: {walkforward_results['total_return_pct']:.2f}%")
    print(f"Total P&L (net): ${walkforward_results['total_pnl']:.2f}")
    print(f"Total Fees Paid: ${walkforward_results['total_fees']:.2f}")
    print(f"Model Retrains: {walkforward_results['retrains']}")
    print()
    print(f"Number of Trades: {walkforward_results['num_trades']}")
    print(f"Winning Trades: {walkforward_results['winning_trades']}")
    print(f"Losing Trades: {walkforward_results['losing_trades']}")
    print(f"Win Rate: {walkforward_results['win_rate_pct']:.1f}%")
    print()
    print(f"Avg Win: ${walkforward_results['avg_win']:.2f}")
    print(f"Avg Loss: ${walkforward_results['avg_loss']:.2f}")
    print(f"Profit Factor: {walkforward_results['profit_factor']:.2f}x")
    print()
    print(f"Sharpe Ratio: {walkforward_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {walkforward_results['max_drawdown_pct']:.2f}%")
    print("=" * 50)

    # Comparison summary
    print("\n" + "=" * 50)
    print("BACKTEST COMPARISON")
    print("=" * 50)
    return_diff = walkforward_results["total_return_pct"] - backtest_results["total_return_pct"]
    trades_diff = walkforward_results["num_trades"] - backtest_results["num_trades"]
    print(f"Return Difference: {return_diff:+.2f}% (Walk-Forward vs Standard)")
    print(f"Trade Count Difference: {trades_diff:+d} trades")
    print()
    if walkforward_results["total_return_pct"] > backtest_results["total_return_pct"]:
        print("✓ Walk-Forward performed BETTER (more realistic)")
    elif walkforward_results["total_return_pct"] < backtest_results["total_return_pct"]:
        print("⚠ Walk-Forward performed WORSE (more realistic, less overfitting)")
    else:
        print("= Similar performance between methods")
    print()
    print("Note: Walk-forward testing is more realistic as it simulates")
    print("periodic model retraining that would occur in live trading.")
    print("=" * 50)

    # Print risk management metrics
    risk_metrics = risk_manager.get_risk_metrics()
    print("\nRISK MANAGEMENT METRICS")
    print("=" * 50)
    print(f"Max Drawdown Limit: {risk_metrics['max_drawdown_limit'] * 100:.1f}%")
    print(f"Current Drawdown: {risk_metrics['current_drawdown'] * 100:.2f}%")
    print(f"Max Position Size: {risk_metrics['max_position_size_pct'] * 100:.1f}% of capital")
    print(f"Stop Loss: {risk_metrics['stop_loss_pct'] * 100:.1f}%")
    print(f"Take Profit: {risk_metrics['take_profit_pct'] * 100:.1f}%")
    print(f"Kelly Fraction: {risk_metrics['kelly_fraction'] * 100:.1f}%")
    print(f"Trading Enabled: {risk_metrics['trading_enabled']}")
    print("=" * 50)

    # Print feature importance
    print("\nFEATURE IMPORTANCE (Top 5)")
    print("=" * 50)
    for idx, row in feature_importance.head(5).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    print("=" * 50)

    # Current real-time regime prediction - use latest values from data
    current = pd.DataFrame(
        [
            {
                "rsi": df["rsi"].iloc[-1],
                "macd": df["macd"].iloc[-1],
                "macd_histogram": df["macd_histogram"].iloc[-1],
                "bb_position": df["bb_position"].iloc[-1],
                "price_above_ema9": df["price_above_ema9"].iloc[-1],
                "price_above_ema21": df["price_above_ema21"].iloc[-1],
                "price_above_ema50": df["price_above_ema50"].iloc[-1],
                "vol_change": df["vol_change"].iloc[-1],
                "vol_sma_ratio": df["vol_sma_ratio"].iloc[-1],
                "price_momentum_5d": df["price_momentum_5d"].iloc[-1],
                "macro_score": 0.6,
                "unlock_pressure": 0.15,
            }
        ]
    )

    # Get regime prediction and probabilities
    regime_pred = model.predict(current)[0]
    regime_probs = model.predict_proba(current)[0]

    regime_names = ["Bull Run", "Ranging Market", "Bear Market", "High Volatility"]
    print(f"\n🔮 PREDICTED MARKET REGIME: {regime_names[regime_pred]}")
    print("=" * 50)
    print("Regime Probabilities:")
    # Only show probabilities for classes that exist in the model
    for i in range(len(regime_probs)):
        marker = "👉" if i == regime_pred else "  "
        print(f"{marker} {regime_names[i]}: {regime_probs[i]:.1%}")
    # Show "Not detected" for missing classes
    for i in range(len(regime_probs), 4):
        print(f"   {regime_names[i]}: Not detected (insufficient data)")
    print("=" * 50)
    print()

    # Trading recommendation based on regime
    if regime_pred == 0:  # Bull
        print("💡 Recommendation: LONG positions with larger size")
    elif regime_pred == 1:  # Ranging
        print("💡 Recommendation: Small positions, mean reversion strategy")
    elif regime_pred == 2:  # Bear
        print("💡 Recommendation: Stay in CASH or consider SHORT")
    elif regime_pred == 3:  # High Vol
        print("💡 Recommendation: Small positions with wider stop-losses")
    print()


if __name__ == "__main__":
    main()
