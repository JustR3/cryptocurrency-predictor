import ccxt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from risk_management import RiskManager, calculate_volatility, calculate_win_rate_and_ratio


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

            # Train new model
            model = XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                random_state=42,
            )
            model.fit(X_train, y_train)
            last_train_idx = i
            num_retrains += 1

        # Get current prediction
        X_current = df.loc[i:i, feature_columns]
        prediction = model.predict(X_current)[0]
        prob_up = model.predict_proba(X_current)[0][1]

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

        # Entry signal
        if position == 0 and prediction == 1:
            # Calculate historical win rate and ratio from past trades
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                win_rate, avg_win_loss_ratio = calculate_win_rate_and_ratio(trades_df)
            else:
                win_rate, avg_win_loss_ratio = 0.5, 1.0

            # Calculate position size using risk manager
            position_size_dollars = risk_manager.calculate_position_size(
                capital=capital,
                entry_price=current_price,
                volatility=volatility,
                win_rate=win_rate,
                avg_win_loss_ratio=avg_win_loss_ratio,
            )

            if position_size_dollars > 0:
                position_size = position_size_dollars / current_price
                position = 1
                entry_price = current_price
                entry_prob = prob_up
                entry_idx = i

                entry_fee = position_size * entry_price * fee_rate
                total_fees += entry_fee
                capital -= entry_fee

        # Exit signal
        elif position == 1 and (prediction == 0 or i == len(df) - 1):
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
    df_test["prediction"] = model.predict(X_test)
    df_test["prob_up"] = model.predict_proba(X_test)[:, 1]

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

        # Entry signal: prediction = 1 (expecting up move)
        if position == 0 and df_test["prediction"].iloc[i] == 1:
            # Calculate historical win rate and ratio from past trades
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                win_rate, avg_win_loss_ratio = calculate_win_rate_and_ratio(trades_df)
            else:
                win_rate, avg_win_loss_ratio = 0.5, 1.0  # Default values

            # Calculate position size using risk manager
            position_size_dollars = risk_manager.calculate_position_size(
                capital=capital,
                entry_price=current_price,
                volatility=volatility,
                win_rate=win_rate,
                avg_win_loss_ratio=avg_win_loss_ratio,
            )

            if position_size_dollars > 0:  # Only enter if risk manager allows
                position_size = position_size_dollars / current_price
                position = 1
                entry_price = current_price
                entry_prob = df_test["prob_up"].iloc[i]
                entry_idx = i

                # Calculate stop-loss and take-profit levels
                stop_loss_price, take_profit_price = risk_manager.calculate_stop_levels(
                    entry_price, "long"
                )

                # Calculate entry fee
                entry_fee = position_size * entry_price * fee_rate
                total_fees += entry_fee
                capital -= entry_fee

        # Exit signal: prediction = 0 OR we're at the end
        elif position == 1 and (df_test["prediction"].iloc[i] == 0 or i == len(df_test) - 1):
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


# Real-time HYPE/USDT data from Hyperliquid via CCXT
exchange = ccxt.hyperliquid()
ohlcv = exchange.fetch_ohlcv("HYPE/USDT", "1d", limit=200)
df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

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

# Target: 1 if next 7-day return >5%, else 0
df["target"] = (df["close"].pct_change(7).shift(-7) > 0.05).astype(int)

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

# XGBoost model - adjusted for more features
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
)
model.fit(X_train, y_train)

# Get test data with original indices preserved
X_test_with_idx = X_test.copy()
y_test_with_idx = y_test.copy()
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


# Current real-time prediction - use latest values from data
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

prob_up = model.predict_proba(current)[0][1]
print(f"Probability of >5% move up in next 1-4 weeks: {prob_up:.1%}")
