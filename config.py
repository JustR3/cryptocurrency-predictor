"""
Centralized configuration for Hyperliquid Predictor.
All settings in one place.
"""

# Trading Settings
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_EXCHANGE = "binance"
DEFAULT_INITIAL_CAPITAL = 10000
TRADING_FEE_BPS = 5  # 5 basis points = 0.05%
SLIPPAGE_BPS = 10  # 10 basis points = 0.1%

# Data Settings
DATA_CACHE_DIR = "data_cache"
DEFAULT_DATA_LIMIT_DAYS = {
    "BTC": 730,  # 2 years for major caps
    "ETH": 730,
    "SOL": 365,  # 1 year for mid caps
    "BNB": 365,
    "ADA": 365,
    "AVAX": 365,
    "DOT": 365,
    "POL": 365,
    "LINK": 365,
    "default": 180,  # 6 months for low caps
}

# Technical Indicator Parameters
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
EMA_PERIODS = [9, 21, 50]
VOLUME_WINDOW = 20
ATR_PERIOD = 14

# Feature Engineering
MOMENTUM_PERIODS = [3, 5, 7]  # Days for momentum calculation
VOLATILITY_WINDOW = 14

# Prediction Mode Configuration
# Determines the prediction horizon and Triple Barrier parameters
PREDICTION_MODES = {
    "short": {
        "name": "Short-term (1-3 days)",
        "time_horizon": 3,
        "profit_pct": 0.03,  # 3% profit target
        "loss_pct": 0.015,  # 1.5% stop loss (2:1 ratio)
        "description": "Quick trades, tight targets",
    },
    "medium": {
        "name": "Medium-term (3-5 days)",
        "time_horizon": 5,
        "profit_pct": 0.05,  # 5% profit target
        "loss_pct": 0.025,  # 2.5% stop loss (2:1 ratio)
        "description": "Balanced swing trades",
    },
    "long": {
        "name": "Long-term (5-10 days)",
        "time_horizon": 10,
        "profit_pct": 0.08,  # 8% profit target
        "loss_pct": 0.04,  # 4% stop loss (2:1 ratio)
        "description": "Position trades, wider targets",
    },
}

DEFAULT_PREDICTION_MODE = "medium"  # Default to medium-term

# Triple Barrier Method Parameters (Legacy - kept for backward compatibility)
# Use PREDICTION_MODES for multi-horizon support
BARRIER_PROFIT_PCT = 0.05  # 5% take profit
BARRIER_LOSS_PCT = 0.025  # 2.5% stop loss (2:1 ratio)
BARRIER_TIME_HORIZON = 5  # 5 days max holding period

# Risk Management
MAX_DRAWDOWN_PCT = 0.15  # 15% max drawdown
MAX_POSITION_SIZE_PCT = 0.20  # 20% of capital per trade
STOP_LOSS_PCT = 0.03  # 3% stop loss
TAKE_PROFIT_PCT = 0.08  # 8% take profit
KELLY_FRACTION = 0.30  # Use 30% of Kelly criterion
MIN_POSITION_SIZE_PCT = 0.01  # 1% minimum position
TARGET_VOLATILITY = 0.15  # 15% annualized volatility target

# Model Prediction Thresholds
# Using probabilistic approach instead of hard predictions
ENTRY_PROB_THRESHOLD = 0.40  # Minimum probability for profit class
ENTRY_EDGE_RATIO = 1.3  # P(profit) must be 1.3x > P(loss)
EXIT_PROB_THRESHOLD = 0.35  # Exit if profit probability drops below this

# Model Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_SAMPLES_FOR_TRAINING = 30

# XGBoost Default Hyperparameters
XGBOOST_DEFAULTS = {
    "major_caps": {  # BTC, ETH
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0,
    },
    "mid_caps": {  # SOL, BNB, ADA, etc.
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 2,
        "gamma": 1,
    },
    "low_caps": {  # Others
        "n_estimators": 300,
        "max_depth": 9,
        "learning_rate": 0.20,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "min_child_weight": 1,
        "gamma": 2,
    },
}

# Walk-Forward Backtest Settings
TRAIN_WINDOW_DAYS = 80
RETRAIN_FREQUENCY_DAYS = 20

# Hyperparameter Tuning Settings
OPTUNA_N_TRIALS = 100
OPTUNA_N_FOLDS = 5
OPTUNA_METRIC = "sharpe_ratio"  # Changed from accuracy to sharpe_ratio

# Feature Columns (Optimized based on comprehensive feature analysis)
# Removed: funding_rate (0% importance across all symbols & modes)
FEATURE_COLUMNS = [
    "rsi",
    "macd",
    "macd_histogram",
    "bb_position",
    "atr",
    "atr_pct",
    "price_above_ema9",
    "price_above_ema21",
    "price_above_ema50",
    "vol_change",
    "vol_sma_ratio",
    "price_momentum_3d",
    "price_momentum_5d",
    "price_momentum_7d",
    "btc_beta",
]


def get_asset_class(symbol: str) -> str:
    """Determine asset class from symbol."""
    symbol_upper = symbol.upper()
    if any(x in symbol_upper for x in ["BTC", "ETH"]):
        return "major_caps"
    elif any(x in symbol_upper for x in ["SOL", "BNB", "ADA", "AVAX", "DOT", "POL", "LINK"]):
        return "mid_caps"
    else:
        return "low_caps"


def get_data_limit(symbol: str, user_limit: int = None) -> int:
    """Get data limit based on asset class or user override."""
    if user_limit:
        return user_limit

    for key, days in DEFAULT_DATA_LIMIT_DAYS.items():
        if key.upper() in symbol.upper():
            return days

    return DEFAULT_DATA_LIMIT_DAYS["default"]


def get_default_hyperparameters(symbol: str) -> dict:
    """Get default hyperparameters based on asset class."""
    asset_class = get_asset_class(symbol)
    params = XGBOOST_DEFAULTS[asset_class].copy()
    params["random_state"] = RANDOM_STATE
    return params


def get_prediction_mode_params(mode: str = DEFAULT_PREDICTION_MODE) -> dict:
    """
    Get Triple Barrier parameters for a specific prediction mode.

    Args:
        mode: Prediction mode ('short', 'medium', or 'long')

    Returns:
        Dictionary with time_horizon, profit_pct, and loss_pct

    Raises:
        ValueError: If mode is not recognized
    """
    if mode not in PREDICTION_MODES:
        raise ValueError(
            f"Invalid prediction mode '{mode}'. Must be one of: {list(PREDICTION_MODES.keys())}"
        )
    return PREDICTION_MODES[mode]
