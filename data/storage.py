"""
Storage module for caching data to disk.
Uses Parquet format for efficient storage.
"""

from pathlib import Path

import pandas as pd

import config


def get_cache_path(symbol: str, data_type: str = "ohlcv") -> Path:
    """
    Get cache file path for a symbol.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        data_type: Type of data ('ohlcv', 'funding', 'features')

    Returns:
        Path to cache file
    """
    cache_dir = Path(config.DATA_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)

    safe_symbol = symbol.replace("/", "_")
    filename = f"{safe_symbol}_{data_type}.parquet"

    return cache_dir / filename


def save_to_cache(df: pd.DataFrame, symbol: str, data_type: str = "ohlcv"):
    """
    Save dataframe to cache.

    Args:
        df: DataFrame to save
        symbol: Trading pair
        data_type: Type of data
    """
    cache_path = get_cache_path(symbol, data_type)
    df.to_parquet(cache_path, index=False)


def load_from_cache(symbol: str, data_type: str = "ohlcv") -> pd.DataFrame:
    """
    Load dataframe from cache.

    Args:
        symbol: Trading pair
        data_type: Type of data

    Returns:
        Cached dataframe, or None if not found
    """
    cache_path = get_cache_path(symbol, data_type)

    if not cache_path.exists():
        return None

    try:
        df = pd.read_parquet(cache_path)
        return df
    except Exception:
        return None


def clear_cache(symbol: str = None):
    """
    Clear cache files.

    Args:
        symbol: If provided, only clear this symbol's cache. Otherwise clear all.
    """
    cache_dir = Path(config.DATA_CACHE_DIR)

    if not cache_dir.exists():
        return

    if symbol:
        safe_symbol = symbol.replace("/", "_")
        pattern = f"{safe_symbol}_*.parquet"
    else:
        pattern = "*.parquet"

    for file in cache_dir.glob(pattern):
        file.unlink()
