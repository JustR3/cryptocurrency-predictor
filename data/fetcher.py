"""
Data fetching module using CCXT.
Handles exchange connections and OHLCV data retrieval.
"""

import ccxt
import pandas as pd

import config


def get_exchange(exchange_name: str):
    """
    Initialize exchange with rate limiting.

    Args:
        exchange_name: Exchange name (e.g., 'binance', 'hyperliquid')

    Returns:
        CCXT exchange instance

    Raises:
        ValueError: If exchange is not supported
    """
    exchange_name = exchange_name.lower()

    try:
        if exchange_name == "hyperliquid":
            return ccxt.hyperliquid({"enableRateLimit": True})
        elif exchange_name == "binance":
            return ccxt.binance({"enableRateLimit": True})
        elif exchange_name == "coinbase":
            return ccxt.coinbase({"enableRateLimit": True})
        elif exchange_name == "kraken":
            return ccxt.kraken({"enableRateLimit": True})
        elif exchange_name == "bybit":
            return ccxt.bybit({"enableRateLimit": True})
        elif exchange_name == "okx":
            return ccxt.okx({"enableRateLimit": True})
        else:
            # Try to load exchange dynamically
            exchange_class = getattr(ccxt, exchange_name)
            return exchange_class({"enableRateLimit": True})
    except AttributeError:
        raise ValueError(
            f"Exchange '{exchange_name}' not supported. "
            "Available: binance, coinbase, kraken, bybit, okx, hyperliquid"
        )


def fetch_ohlcv(
    symbol: str, exchange_name: str = config.DEFAULT_EXCHANGE, limit: int = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data from exchange.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        exchange_name: Exchange name
        limit: Number of days to fetch (None = use config default)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume

    Raises:
        ValueError: If insufficient data or symbol not available
    """
    if limit is None:
        limit = config.get_data_limit(symbol)

    exchange = get_exchange(exchange_name)

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, "1d", limit=limit)
    except Exception as e:
        # Try USDC if USDT fails
        if "/USDT" in symbol:
            usdc_symbol = symbol.replace("/USDT", "/USDC")
            print(f"⚠ {symbol} not available, trying {usdc_symbol}")
            try:
                ohlcv = exchange.fetch_ohlcv(usdc_symbol, "1d", limit=limit)
            except Exception:
                raise ValueError(
                    f"Error fetching {symbol} or {usdc_symbol} from {exchange_name}: {e}"
                )
        else:
            raise ValueError(f"Error fetching {symbol} from {exchange_name}: {e}")

    if len(ohlcv) < 50:
        raise ValueError(f"Insufficient data: got {len(ohlcv)} days, need at least 50")

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df


def fetch_funding_rate(
    symbol: str, exchange_name: str = config.DEFAULT_EXCHANGE, limit: int = None
) -> pd.DataFrame:
    """
    Fetch funding rate data for perpetual futures.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        exchange_name: Exchange name
        limit: Number of records to fetch

    Returns:
        DataFrame with funding rate data (returns empty df if not available)
    """
    if limit is None:
        limit = config.get_data_limit(symbol)

    try:
        exchange = get_exchange(exchange_name)

        # Check if exchange supports funding rates
        if not hasattr(exchange, "fetch_funding_rate_history"):
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

        funding_history = exchange.fetch_funding_rate_history(symbol, limit=limit)

        if not funding_history:
            return pd.DataFrame(columns=["timestamp", "funding_rate"])

        df = pd.DataFrame(funding_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "fundingRate"]].rename(columns={"fundingRate": "funding_rate"})

        return df

    except Exception as e:
        print(f"Warning: Could not fetch funding rates for {symbol}: {e}")
        return pd.DataFrame(columns=["timestamp", "funding_rate"])


def fetch_btc_data(exchange_name: str = config.DEFAULT_EXCHANGE, limit: int = None) -> pd.DataFrame:
    """
    Fetch BTC/USDT data for beta calculation.
    Falls back to Binance if the primary exchange doesn't support BTC/USDT.

    Args:
        exchange_name: Exchange name
        limit: Number of days to fetch

    Returns:
        DataFrame with BTC OHLCV data
    """
    # Try primary exchange first
    try:
        return fetch_ohlcv("BTC/USDT", exchange_name, limit)
    except ValueError:
        # Fall back to Binance (most reliable for BTC data)
        if exchange_name != "binance":
            print(
                f"⚠ {exchange_name} doesn't support BTC/USDT, "
                "using Binance for BTC Beta calculation"
            )
            return fetch_ohlcv("BTC/USDT", "binance", limit)
        raise
