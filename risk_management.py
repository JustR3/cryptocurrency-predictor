"""
Risk Management Module for Hyperliquid Trading Strategy.

This module provides comprehensive risk management functionality including:
- Stop-loss and take-profit calculations
- Position sizing based on Kelly criterion
- Maximum drawdown limits
- Risk-adjusted position sizing
"""

from typing import Dict, Optional, Tuple

import pandas as pd


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
