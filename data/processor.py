"""
Feature engineering and data processing.
Calculates technical indicators, returns, and other features.
"""

import pandas as pd

import config


def compute_rsi(prices: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Calculate RSI (Relative Strength Index)."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def compute_macd(
    prices: pd.Series,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = compute_ema(prices, fast)
    ema_slow = compute_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def compute_bollinger_bands(
    prices: pd.Series, period: int = config.BB_PERIOD, num_std: int = config.BB_STD
):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return upper_band, lower_band, bb_position


def compute_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.Series:
    """Calculate Average True Range (volatility indicator)."""
    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def compute_btc_beta(
    asset_returns: pd.Series, btc_returns: pd.Series, window: int = 60
) -> pd.Series:
    """
    Calculate rolling beta relative to BTC.

    Args:
        asset_returns: Asset return series
        btc_returns: BTC return series
        window: Rolling window for beta calculation

    Returns:
        Series with beta values
    """
    # Align the two series by timestamp
    combined = pd.DataFrame({"asset": asset_returns, "btc": btc_returns}).dropna()

    if len(combined) < window:
        return pd.Series(1.0, index=asset_returns.index)  # Default beta = 1

    # Calculate rolling beta using cov and var
    rolling_cov = combined["asset"].rolling(window=window).cov(combined["btc"])
    rolling_var = combined["btc"].rolling(window=window).var()

    beta = rolling_cov / rolling_var
    beta = beta.fillna(1.0)  # Fill NaN with default beta = 1

    # Return with proper index
    result = pd.Series(1.0, index=asset_returns.index)
    result.update(beta)

    return result


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to dataframe.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with technical indicators added
    """
    df = df.copy()

    # RSI
    df["rsi"] = compute_rsi(df["close"])

    # MACD
    macd_line, signal_line, macd_histogram = compute_macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = macd_histogram

    # Bollinger Bands
    upper_band, lower_band, bb_position = compute_bollinger_bands(df["close"])
    df["bb_upper"] = upper_band
    df["bb_lower"] = lower_band
    df["bb_position"] = bb_position

    # ATR (volatility)
    df["atr"] = compute_atr(df)
    df["atr_pct"] = df["atr"] / df["close"]  # ATR as percentage of price

    # EMAs
    for period in config.EMA_PERIODS:
        df[f"ema_{period}"] = compute_ema(df["close"], period)

    # Price relative to EMAs
    df["price_above_ema9"] = (df["close"] > df["ema_9"]).astype(int)
    df["price_above_ema21"] = (df["close"] > df["ema_21"]).astype(int)
    df["price_above_ema50"] = (df["close"] > df["ema_50"]).astype(int)

    # Volume
    df["vol_change"] = df["volume"].pct_change().fillna(0)
    df["vol_sma_ratio"] = df["volume"] / df["volume"].rolling(window=config.VOLUME_WINDOW).mean()

    # Price momentum
    for period in config.MOMENTUM_PERIODS:
        df[f"price_momentum_{period}d"] = df["close"].pct_change(period)

    return df


def add_btc_beta(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add BTC beta feature to dataframe.

    Args:
        df: Asset dataframe
        btc_df: BTC dataframe

    Returns:
        DataFrame with btc_beta column
    """
    df = df.copy()

    # Calculate returns
    asset_returns = df["close"].pct_change()
    btc_returns = btc_df["close"].pct_change()

    # Align by timestamp
    asset_returns.index = df["timestamp"]
    btc_returns.index = btc_df["timestamp"]

    # Compute beta
    df["btc_beta"] = compute_btc_beta(asset_returns, btc_returns).values

    return df


def add_funding_rate(df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add funding rate feature to dataframe.

    Args:
        df: Asset dataframe
        funding_df: Funding rate dataframe

    Returns:
        DataFrame with funding_rate column
    """
    df = df.copy()

    if funding_df.empty:
        # No funding data available, use neutral value
        df["funding_rate"] = 0.0
        return df

    # Merge funding rates with main dataframe
    df = df.merge(funding_df, on="timestamp", how="left")

    # Forward fill missing values and fill remaining with 0
    df["funding_rate"] = df["funding_rate"].fillna(method="ffill").fillna(0.0)

    return df


def create_features(
    df: pd.DataFrame, btc_df: pd.DataFrame = None, funding_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Create all features for modeling.

    Args:
        df: Raw OHLCV dataframe
        btc_df: BTC OHLCV dataframe (optional, for beta calculation)
        funding_df: Funding rate dataframe (optional)

    Returns:
        DataFrame with all features
    """
    # Add technical indicators
    df = add_technical_indicators(df)

    # Add BTC beta if BTC data provided
    if btc_df is not None and not btc_df.empty:
        df = add_btc_beta(df, btc_df)
    else:
        df["btc_beta"] = 1.0  # Default beta

    # Add funding rate if provided
    if funding_df is not None and not funding_df.empty:
        df = add_funding_rate(df, funding_df)
    else:
        df["funding_rate"] = 0.0  # Default funding rate

    return df


def create_triple_barrier_labels(
    df: pd.DataFrame,
    profit_pct: float = config.BARRIER_PROFIT_PCT,
    loss_pct: float = config.BARRIER_LOSS_PCT,
    time_horizon: int = config.BARRIER_TIME_HORIZON,
) -> pd.Series:
    """
    Create labels using Triple Barrier Method.

    For each bar, look forward to see which barrier is hit first:
    - Upper barrier (profit_pct): Label = 1 (Long win)
    - Lower barrier (loss_pct): Label = -1 (Long loss)
    - Time barrier (time_horizon): Label = 0 (Neutral/timeout) or sign of return

    Args:
        df: DataFrame with close prices
        profit_pct: Upper barrier profit percentage
        loss_pct: Lower barrier loss percentage
        time_horizon: Maximum days to hold

    Returns:
        Series with labels (-1, 0, 1)
    """
    labels = pd.Series(index=df.index, dtype=int)

    for i in range(len(df) - time_horizon):
        entry_price = df["close"].iloc[i]
        upper_barrier = entry_price * (1 + profit_pct)
        lower_barrier = entry_price * (1 - loss_pct)

        # Look forward up to time_horizon days
        future_prices = df["close"].iloc[i + 1 : i + 1 + time_horizon]

        # Check which barrier is hit first
        hit_upper = future_prices >= upper_barrier
        hit_lower = future_prices <= lower_barrier

        if hit_upper.any() and hit_lower.any():
            # Both hit, use whichever came first
            upper_idx = hit_upper.idxmax() if hit_upper.any() else float("inf")
            lower_idx = hit_lower.idxmax() if hit_lower.any() else float("inf")

            if upper_idx < lower_idx:
                labels.iloc[i] = 1  # Profit barrier hit first
            else:
                labels.iloc[i] = -1  # Loss barrier hit first

        elif hit_upper.any():
            labels.iloc[i] = 1  # Only profit barrier hit

        elif hit_lower.any():
            labels.iloc[i] = -1  # Only loss barrier hit

        else:
            # Time barrier hit, use sign of return
            exit_price = future_prices.iloc[-1] if len(future_prices) > 0 else entry_price
            return_pct = (exit_price - entry_price) / entry_price

            if return_pct > 0.01:  # >1% gain
                labels.iloc[i] = 1
            elif return_pct < -0.01:  # >1% loss
                labels.iloc[i] = -1
            else:
                labels.iloc[i] = 0  # Neutral

    # Fill remaining NaN with 0
    labels.fillna(0, inplace=True)

    return labels.astype(int)
