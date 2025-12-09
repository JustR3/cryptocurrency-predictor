"""
Backtesting engine with volatility targeting and realistic costs.
Implements Triple Barrier Method for trade management.
"""

import numpy as np
import pandas as pd

import config
from risk_management import RiskManager, calculate_volatility, calculate_win_rate_and_ratio


def calculate_position_size_volatility_targeting(
    capital: float,
    current_volatility: float,
    target_volatility: float = config.TARGET_VOLATILITY,
    max_position_pct: float = config.MAX_POSITION_SIZE_PCT,
) -> float:
    """
    Calculate position size using volatility targeting.

    Position size scales inversely with volatility to maintain constant risk.

    Args:
        capital: Current capital
        current_volatility: Current asset volatility (annualized)
        target_volatility: Target portfolio volatility
        max_position_pct: Maximum position size as % of capital

    Returns:
        Position size in dollars
    """
    if current_volatility == 0:
        return 0.0

    # Size inversely proportional to volatility
    volatility_scalar = target_volatility / current_volatility

    # Apply constraints
    volatility_scalar = min(volatility_scalar, max_position_pct)
    volatility_scalar = max(volatility_scalar, config.MIN_POSITION_SIZE_PCT)

    position_size = capital * volatility_scalar

    return position_size


def apply_slippage_and_fees(
    price: float,
    side: str,
    fee_bps: float = config.TRADING_FEE_BPS,
    slippage_bps: float = config.SLIPPAGE_BPS,
) -> float:
    """
    Apply slippage and fees to execution price.

    Args:
        price: Market price
        side: 'buy' or 'sell'
        fee_bps: Trading fee in basis points
        slippage_bps: Slippage in basis points

    Returns:
        Adjusted execution price
    """
    total_cost_bps = fee_bps + slippage_bps
    total_cost = total_cost_bps / 10000

    if side == "buy":
        # Pay more when buying
        return price * (1 + total_cost)
    else:  # sell
        # Receive less when selling
        return price * (1 - total_cost)


def backtest_strategy(
    df: pd.DataFrame,
    model,
    label_encoder,
    feature_columns: list,
    initial_capital: float = config.DEFAULT_INITIAL_CAPITAL,
    risk_manager: RiskManager = None,
):
    """
    Backtest strategy with volatility targeting and realistic costs.

    Args:
        df: DataFrame with features and prices
        model: Trained model
        label_encoder: Label encoder for predictions
        feature_columns: List of feature column names
        initial_capital: Starting capital
        risk_manager: RiskManager instance

    Returns:
        Dictionary with backtest results
    """
    if risk_manager is None:
        risk_manager = RiskManager()

    risk_manager.reset()
    risk_manager.peak_equity = initial_capital

    df = df.copy()

    # Generate predictions
    if model is None or label_encoder is None:
        print("Warning: No model provided for backtesting")
        return {
            "final_equity": initial_capital,
            "total_return_pct": 0.0,
            "total_pnl": 0.0,
            "total_fees": 0.0,
            "num_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": pd.DataFrame(),
        }

    X = df[feature_columns]
    predictions_encoded = model.predict(X)
    probs_encoded = model.predict_proba(X)

    # Decode predictions
    df["prediction"] = label_encoder.inverse_transform(predictions_encoded)
    df["confidence"] = [probs[pred] for probs, pred in zip(probs_encoded, predictions_encoded)]

    # Initialize tracking
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_idx = 0
    position_size = 0
    entry_confidence = 0
    trades = []
    equity_curve = [capital]
    total_fees = 0

    # Calculate volatility for position sizing
    volatility = calculate_volatility(df["close"])

    for i in range(len(df)):
        current_price = df["close"].iloc[i]

        # Check for exit signals if in position
        if position == 1:
            should_exit, exit_reason = risk_manager.should_exit_trade(
                current_price, entry_price, "long"
            )

            # Also check prediction-based exit
            prediction = df["prediction"].iloc[i]
            confidence = df["confidence"].iloc[i]

            # Exit if prediction turns bearish (0) or low confidence
            if prediction == 0 or confidence < 0.35:
                should_exit = True
                exit_reason = "signal" if prediction == 0 else "low_confidence"

            if should_exit:
                # Apply slippage and fees
                exit_price = apply_slippage_and_fees(current_price, "sell")
                position_value = position_size * exit_price
                exit_cost = position_size * current_price * (config.TRADING_FEE_BPS / 10000)

                gross_pnl = position_size * (exit_price - entry_price)
                net_pnl = position_value - (position_size * entry_price)

                capital += position_value
                total_fees += exit_cost

                trades.append(
                    {
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "fees": exit_cost,
                        "position_size": position_size,
                        "entry_confidence": entry_confidence,
                        "exit_reason": exit_reason,
                    }
                )

                position = 0

        # Entry signals
        if position == 0:
            prediction = df["prediction"].iloc[i]
            confidence = df["confidence"].iloc[i]

            # Enter long if prediction is bullish (2) and confident
            should_enter = prediction == 2 and confidence > 0.55

            if should_enter:
                # Calculate position size using volatility targeting
                position_size_dollars = calculate_position_size_volatility_targeting(
                    capital, volatility
                )

                # Calculate historical win rate for Kelly sizing
                if len(trades) > 0:
                    trades_df = pd.DataFrame(trades)
                    win_rate, avg_win_loss_ratio = calculate_win_rate_and_ratio(trades_df)
                else:
                    win_rate, avg_win_loss_ratio = 0.5, 1.0

                # Adjust size using risk manager
                base_position_size = risk_manager.calculate_position_size(
                    capital=capital,
                    entry_price=current_price,
                    volatility=volatility,
                    win_rate=win_rate,
                    avg_win_loss_ratio=avg_win_loss_ratio,
                )

                # Use minimum of volatility-targeted and risk-adjusted size
                position_size_dollars = min(position_size_dollars, base_position_size)

                if position_size_dollars > 0:
                    # Apply slippage and fees to entry
                    entry_price = apply_slippage_and_fees(current_price, "buy")
                    position_size = position_size_dollars / entry_price
                    entry_cost = position_size * current_price * (config.TRADING_FEE_BPS / 10000)

                    capital -= position_size * entry_price
                    total_fees += entry_cost

                    position = 1
                    entry_idx = i
                    entry_confidence = confidence

        # Update equity curve
        current_equity = capital
        if position == 1:
            unrealized_value = position_size * current_price
            current_equity += unrealized_value

        risk_manager.update_drawdown(current_equity)
        equity_curve.append(current_equity)

    # Close any remaining position
    if position == 1:
        exit_price = apply_slippage_and_fees(df["close"].iloc[-1], "sell")
        position_value = position_size * exit_price
        capital += position_value

        gross_pnl = position_size * (exit_price - entry_price)
        net_pnl = position_value - (position_size * entry_price)

        trades.append(
            {
                "entry_idx": entry_idx,
                "exit_idx": len(df) - 1,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "fees": position_size * df["close"].iloc[-1] * (config.TRADING_FEE_BPS / 10000),
                "position_size": position_size,
                "entry_confidence": entry_confidence,
                "exit_reason": "end_of_period",
            }
        )

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

    # Sharpe Ratio
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe_ratio = (
        np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
        if len(daily_returns) > 1
        else 0
    )

    # Max Drawdown
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


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_columns: list,
    symbol: str,
    initial_capital: float = config.DEFAULT_INITIAL_CAPITAL,
    risk_manager: RiskManager = None,
    train_window: int = config.TRAIN_WINDOW_DAYS,
    retrain_frequency: int = config.RETRAIN_FREQUENCY_DAYS,
):
    """
    Walk-forward backtest with periodic retraining.

    Args:
        df: Full dataframe with features and target
        feature_columns: List of feature column names
        symbol: Trading pair symbol
        initial_capital: Starting capital
        risk_manager: RiskManager instance
        train_window: Days to use for training
        retrain_frequency: Days between retraining

    Returns:
        Dictionary with backtest results
    """
    from strategies.xgb_strategy import train_model

    if risk_manager is None:
        risk_manager = RiskManager()

    risk_manager.reset()
    risk_manager.peak_equity = initial_capital

    df = df.copy().reset_index(drop=True)

    capital = initial_capital
    position = 0
    entry_price = 0
    entry_confidence = 0
    position_size = 0
    entry_idx = 0
    trades = []
    equity_curve = [capital]
    total_fees = 0

    volatility = calculate_volatility(df["close"])

    model = None
    label_encoder = None
    last_train_idx = 0
    num_retrains = 0

    for i in range(train_window, len(df)):
        # Retrain model periodically
        if model is None or (i - last_train_idx) >= retrain_frequency:
            train_start = max(0, i - train_window)
            train_end = i

            X_train = df.loc[train_start : train_end - 1, feature_columns]
            y_train = df.loc[train_start : train_end - 1, "target"]

            model, label_encoder = train_model(X_train, y_train, symbol)

            if model is not None:
                last_train_idx = i
                num_retrains += 1

        if model is None or label_encoder is None:
            continue

        # Get prediction
        X_current = df.loc[i:i, feature_columns]
        predictions_encoded = model.predict(X_current)
        probs_encoded = model.predict_proba(X_current)

        prediction = label_encoder.inverse_transform(predictions_encoded)[0]
        confidence = probs_encoded[0][predictions_encoded[0]]

        current_price = df.loc[i, "close"]

        # Check exits
        if position == 1:
            should_exit, exit_reason = risk_manager.should_exit_trade(
                current_price, entry_price, "long"
            )

            if prediction == -1 or confidence < 0.35:
                should_exit = True
                exit_reason = "signal" if prediction == -1 else "low_confidence"

            if should_exit:
                exit_price = apply_slippage_and_fees(current_price, "sell")
                position_value = position_size * exit_price
                exit_cost = position_size * current_price * (config.TRADING_FEE_BPS / 10000)

                gross_pnl = position_size * (exit_price - entry_price)
                net_pnl = position_value - (position_size * entry_price)

                capital += position_value
                total_fees += exit_cost

                trades.append(
                    {
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "fees": exit_cost,
                        "position_size": position_size,
                        "entry_confidence": entry_confidence,
                        "exit_reason": exit_reason,
                    }
                )

                position = 0

        # Entry signals
        if position == 0:
            should_enter = prediction == 1 and confidence > 0.55

            if should_enter:
                position_size_dollars = calculate_position_size_volatility_targeting(
                    capital, volatility
                )

                if len(trades) > 0:
                    trades_df = pd.DataFrame(trades)
                    win_rate, avg_win_loss_ratio = calculate_win_rate_and_ratio(trades_df)
                else:
                    win_rate, avg_win_loss_ratio = 0.5, 1.0

                base_position_size = risk_manager.calculate_position_size(
                    capital=capital,
                    entry_price=current_price,
                    volatility=volatility,
                    win_rate=win_rate,
                    avg_win_loss_ratio=avg_win_loss_ratio,
                )

                position_size_dollars = min(position_size_dollars, base_position_size)

                if position_size_dollars > 0:
                    entry_price = apply_slippage_and_fees(current_price, "buy")
                    position_size = position_size_dollars / entry_price
                    entry_cost = position_size * current_price * (config.TRADING_FEE_BPS / 10000)

                    capital -= position_size * entry_price
                    total_fees += entry_cost

                    position = 1
                    entry_idx = i
                    entry_confidence = confidence

        # Update equity
        current_equity = capital
        if position == 1:
            current_equity += position_size * current_price

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

    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe_ratio = (
        np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252)
        if len(daily_returns) > 1
        else 0
    )

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
