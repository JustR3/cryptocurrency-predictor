"""
Backtesting engine with volatility targeting and realistic costs.
Implements Triple Barrier Method for trade management.
Includes integrated risk management system.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

import config


class RiskManager:
    """
    Risk management system for trading strategies.

    Handles position sizing, stop-loss/take-profit levels, and drawdown limits.
    """

    def __init__(
        self,
        max_drawdown_pct: float = 0.20,  # 20% max drawdown
        max_position_size_pct: float = 0.10,  # 10% of capital per trade
        stop_loss_pct: float = 0.05,  # 5% stop loss
        take_profit_pct: float = 0.10,  # 10% take profit
        kelly_fraction: float = 0.5,  # Use half Kelly for safety
        min_position_size_pct: float = 0.01,  # 1% minimum position
    ):
        """
        Initialize risk manager with parameters.

        Args:
            max_drawdown_pct: Maximum allowed drawdown before stopping trading
            max_position_size_pct: Maximum position size as % of capital
            stop_loss_pct: Stop loss percentage from entry price
            take_profit_pct: Take profit percentage from entry price
            kelly_fraction: Fraction of Kelly criterion to use (0.5 = half Kelly)
            min_position_size_pct: Minimum position size as % of capital
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_size_pct = max_position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.kelly_fraction = kelly_fraction
        self.min_position_size_pct = min_position_size_pct

        # Trading state
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.is_trading_enabled = True

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        volatility: float,
        win_rate: float,
        avg_win_loss_ratio: float,
        risk_per_trade_pct: Optional[float] = None,
    ) -> float:
        """
        Calculate optimal position size using Kelly criterion and risk limits.

        Args:
            capital: Current capital
            entry_price: Entry price for the trade
            volatility: Price volatility (standard deviation of returns)
            win_rate: Historical win rate (0-1)
            avg_win_loss_ratio: Average win/loss ratio
            risk_per_trade_pct: Override for risk per trade (optional)

        Returns:
            Position size in dollars
        """
        if not self.is_trading_enabled:
            return 0.0

        # Use provided risk per trade or default to stop loss percentage
        risk_pct = risk_per_trade_pct or self.stop_loss_pct

        # Kelly Criterion: f = (bp - q) / b
        # where: b = odds (avg_win/avg_loss), p = win_rate, q = loss_rate
        if win_rate > 0 and avg_win_loss_ratio > 0:
            b = avg_win_loss_ratio
            p = win_rate
            q = 1 - p

            kelly_pct = (b * p - q) / b if b > 0 else 0
            kelly_pct = max(0, kelly_pct * self.kelly_fraction)  # Apply Kelly fraction
        else:
            kelly_pct = self.min_position_size_pct

        # Apply volatility adjustment (higher volatility = smaller position)
        vol_adjustment = min(1.0, 0.20 / volatility) if volatility > 0 else 1.0

        # Calculate position sizes
        kelly_size = capital * kelly_pct * vol_adjustment
        max_size = capital * self.max_position_size_pct
        min_size = capital * self.min_position_size_pct
        risk_based_size = capital * risk_pct  # Risk-based sizing

        # Take the minimum of all constraints
        position_size = min(kelly_size, max_size, risk_based_size)
        position_size = max(position_size, min_size)

        return position_size

    def calculate_stop_levels(
        self, entry_price: float, position_type: str = "long"
    ) -> Tuple[float, float]:
        """
        Calculate stop-loss and take-profit levels.

        Args:
            entry_price: Entry price
            position_type: "long" or "short"

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if position_type.lower() == "long":
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
        else:  # short
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)

        return stop_loss, take_profit

    def update_drawdown(self, current_equity: float) -> bool:
        """
        Update drawdown tracking and check if trading should be stopped.

        Args:
            current_equity: Current portfolio equity

        Returns:
            True if trading should continue, False if stopped due to drawdown
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if self.current_drawdown >= self.max_drawdown_pct:
            self.is_trading_enabled = False
            return False

        return True

    def should_exit_trade(
        self, current_price: float, entry_price: float, position_type: str = "long"
    ) -> Tuple[bool, str]:
        """
        Check if a trade should be exited based on stop-loss/take-profit.

        Args:
            current_price: Current market price
            entry_price: Entry price of the position
            position_type: "long" or "short"

        Returns:
            Tuple of (should_exit, exit_reason)
        """
        stop_loss, take_profit = self.calculate_stop_levels(entry_price, position_type)

        if position_type.lower() == "long":
            if current_price <= stop_loss:
                return True, "stop_loss"
            elif current_price >= take_profit:
                return True, "take_profit"
        else:  # short
            if current_price >= stop_loss:
                return True, "stop_loss"
            elif current_price <= take_profit:
                return True, "take_profit"

        return False, ""

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk management metrics.

        Returns:
            Dictionary with risk metrics
        """
        return {
            "max_drawdown_limit": self.max_drawdown_pct,
            "current_drawdown": self.current_drawdown,
            "max_position_size_pct": self.max_position_size_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "kelly_fraction": self.kelly_fraction,
            "trading_enabled": self.is_trading_enabled,
        }

    def reset(self):
        """Reset risk manager state for new backtest/run."""
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.is_trading_enabled = True


def calculate_volatility(prices: pd.Series, window: int = 20) -> float:
    """
    Calculate price volatility using standard deviation of returns.

    Args:
        prices: Price series
        window: Rolling window for volatility calculation

    Returns:
        Current volatility as fraction (e.g., 0.02 = 2%)
    """
    if len(prices) < window:
        return 0.02  # Default 2% volatility

    returns = prices.pct_change().dropna()
    volatility = returns.rolling(window=window).std().iloc[-1]

    return max(volatility, 0.005)  # Minimum 0.5% volatility


def calculate_win_rate_and_ratio(trades_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate historical win rate and average win/loss ratio.

    Args:
        trades_df: DataFrame with trade results (must have 'net_pnl' column)

    Returns:
        Tuple of (win_rate, avg_win_loss_ratio)
    """
    if len(trades_df) == 0:
        return 0.5, 1.0  # Default values

    winning_trades = trades_df[trades_df["net_pnl"] > 0]
    losing_trades = trades_df[trades_df["net_pnl"] <= 0]

    win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0.5

    avg_win = winning_trades["net_pnl"].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades["net_pnl"].mean()) if len(losing_trades) > 0 else 1

    avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

    return win_rate, avg_win_loss_ratio


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

    # Add probability columns for each class (0=loss, 1=neutral, 2=profit)
    # Handle case where not all classes are present in label_encoder
    df["prob_loss"] = 0.0
    df["prob_neutral"] = 0.0
    df["prob_profit"] = 0.0

    for i, class_label in enumerate(label_encoder.classes_):
        if class_label == 0:
            df["prob_loss"] = probs_encoded[:, i]
        elif class_label == 1:
            df["prob_neutral"] = probs_encoded[:, i]
        elif class_label == 2:
            df["prob_profit"] = probs_encoded[:, i]

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

            # Also check prediction-based exit using probabilities
            prob_profit = df["prob_profit"].iloc[i]
            prob_loss = df["prob_loss"].iloc[i]

            # Exit if profit probability drops below threshold OR loss probability dominates
            if prob_profit < config.EXIT_PROB_THRESHOLD or prob_loss > prob_profit:
                should_exit = True
                exit_reason = (
                    "low_profit_probability"
                    if prob_profit < config.EXIT_PROB_THRESHOLD
                    else "loss_probability_high"
                )

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
            prob_profit = df["prob_profit"].iloc[i]
            prob_loss = df["prob_loss"].iloc[i]

            # Enter long using probabilistic approach:
            # 1. Profit probability above threshold
            # 2. Profit probability is significantly higher than loss probability
            should_enter = (
                prob_profit > config.ENTRY_PROB_THRESHOLD
                and prob_profit > prob_loss * config.ENTRY_EDGE_RATIO
            )

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
                    entry_confidence = prob_profit  # Track profit probability at entry

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
    from strategies.xgb_strategy import load_hyperparameters, train_model

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

    # Load hyperparameters once at the start
    hyperparams = load_hyperparameters(symbol, verbose=True)

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

            model, label_encoder = train_model(
                X_train, y_train, symbol, hyperparams=hyperparams, verbose=False
            )

            if model is not None:
                last_train_idx = i
                num_retrains += 1

        if model is None or label_encoder is None:
            continue

        # Get prediction
        X_current = df.loc[i:i, feature_columns]
        predictions_encoded = model.predict(X_current)
        probs_encoded = model.predict_proba(X_current)

        # Ensure predictions_encoded is 1D
        if predictions_encoded.ndim > 1:
            predictions_encoded = predictions_encoded.flatten()

        prediction = label_encoder.inverse_transform(predictions_encoded)[0]
        confidence = probs_encoded[0][predictions_encoded[0]]

        # Extract probabilities for each class
        prob_loss = 0.0
        prob_neutral = 0.0
        prob_profit = 0.0
        for j, class_label in enumerate(label_encoder.classes_):
            if class_label == 0:
                prob_loss = probs_encoded[0][j]
            elif class_label == 1:
                prob_neutral = probs_encoded[0][j]
            elif class_label == 2:
                prob_profit = probs_encoded[0][j]

        current_price = df.loc[i, "close"]

        # Check exits
        if position == 1:
            should_exit, exit_reason = risk_manager.should_exit_trade(
                current_price, entry_price, "long"
            )

            # Exit if profit probability drops below threshold OR loss probability dominates
            if prob_profit < config.EXIT_PROB_THRESHOLD or prob_loss > prob_profit:
                should_exit = True
                exit_reason = (
                    "low_profit_probability"
                    if prob_profit < config.EXIT_PROB_THRESHOLD
                    else "loss_probability_high"
                )

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
            # Enter long using probabilistic approach
            should_enter = (
                prob_profit > config.ENTRY_PROB_THRESHOLD
                and prob_profit > prob_loss * config.ENTRY_EDGE_RATIO
            )

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
                    entry_confidence = prob_profit  # Track profit probability at entry

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
