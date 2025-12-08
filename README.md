# Hyperliquid HYPE/USDT Price Predictor

A machine learning-based trading strategy for predicting HYPE/USDT price movements on the Hyperliquid exchange.

## Overview

This project uses XGBoost machine learning to predict whether HYPE/USDT will experience a >5% price increase over a 7-day period. The model combines technical indicators (RSI), volume analysis, and macro-sentiment inputs to generate trading signals.

## Features

- **Real-time Data Fetching**: Retrieves 200 days of OHLCV data from Hyperliquid via CCXT
- **Technical Indicators**: Calculates RSI (Relative Strength Index) for momentum analysis
- **XGBoost Classification**: Machine learning model trained on historical data
- **Comprehensive Backtesting**: Full P&L simulation with realistic trading mechanics
  - Position sizing (configurable % of capital)
  - Trading fees (basis points)
  - Equity curve tracking
  - Win rate, profit factor, Sharpe ratio
  - Maximum drawdown analysis
- **Real-time Predictions**: Live probability forecasts for upcoming market moves

## Installation

### Prerequisites
- Python 3.10+
- [UV](https://astral.sh/blog/uv/) package manager
- macOS: `libomp` for XGBoost support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/justra/hyperliquid-predictor.git
cd hyperliquid-predictor
```

2. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. (macOS only) Install OpenMP runtime:
```bash
brew install libomp
```

## Running the Application

### Quick Start
```bash
# Run the complete pipeline
uv run main.py
```

### Main Predictor
```bash
uv run main.py
```

This will:
1. Fetch 200 days of HYPE/USDT data from Hyperliquid
2. Calculate technical indicators (RSI, volume change)
3. Train XGBoost model on 80% of historical data
4. Backtest on 20% of data and display performance metrics
5. Generate live prediction for next market move

### Hyperparameter Tuning (Optional)

To optimize the model's hyperparameters using Optuna's Bayesian optimization:

```bash
# Quick tuning (20 trials, ~1 minute)
uv run tune.py --trials 20

# Standard tuning (100 trials, ~6 minutes)
uv run tune.py --trials 100

# Comprehensive tuning with custom CV folds
uv run tune.py --trials 100 --folds 5

# Save best parameters to file for reuse
uv run tune.py --trials 100 --save
```

**How tuning works:**
- Uses Tree-structured Parzen Estimator (TPE) - a Bayesian optimization algorithm
- Intelligently explores hyperparameter space, learning from each trial
- Focuses computational budget on promising regions
- ~50-100x faster than random or grid search
- Optimizes: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, gamma

**Using tuned parameters:**
After running `tune.py --save`, a `best_hyperparameters.json` file is created. You can then:
1. Copy the parameters from the JSON file
2. Update the model creation in `main.py` with the optimized values
3. Re-run `main.py` with the new parameters

### Typical Workflow

```bash
# 1. Initial exploration
uv run main.py  # See current performance

# 2. Optimize model
uv run tune.py --trials 50 --save

# 3. Update main.py with best_hyperparameters.json values

# 4. Evaluate improved model
uv run main.py  # See improved backtest results
```

## Output Example

```
==================================================
BACKTEST RESULTS
==================================================
Initial Capital: $10,000
Final Equity: $13235.48
Total Return: 32.35%
Total P&L (net): $3235.48
Total Fees Paid: $3.27

Number of Trades: 3
Winning Trades: 1
Losing Trades: 2
Win Rate: 33.3%

Avg Win: $76.31
Avg Loss: $-68.17
Profit Factor: 0.56x

Sharpe Ratio: 6.29
Max Drawdown: -2.17%
==================================================

Probability of >5% move up in next 1-4 weeks: 42.1%
```

## Model Architecture

### Features
- **RSI (14-period)**: Relative Strength Index for momentum
- **Volume Change**: Percentage change in trading volume
- **Macro Score**: Manual sentiment score (0-1 scale)
- **Unlock Pressure**: Token unlock/dilution pressure estimate

### Target
Binary classification: 1 if next 7-day return > 5%, else 0

### XGBoost Hyperparameters
- Estimators: 200
- Max Depth: 4
- Learning Rate: 0.05
- Subsample: 0.8

## Backtest Explanation

The backtest simulates real trading to evaluate strategy performance:

**How it works:**
1. **Training (80% of data)**: Model learns patterns from 80 days of data
2. **Testing (20% of data)**: Model makes predictions on unseen 20 days
3. **Trade Simulation**: 
   - **Entry**: When model predicts 1 (expects >5% move up)
   - **Position Size**: 10% of available capital per trade
   - **Fees**: 5 basis points on entry and exit
   - **Exit**: When model predicts 0 or at end of period

**Metrics calculated:**
- **Total Return %**: Final equity vs initial capital
- **Total P&L (net)**: Total profit/loss after all fees (= Final Equity - Initial Capital)
- **Total Fees Paid**: Trading fees deducted from capital
- **Win Rate**: % of trades that were profitable
- **Profit Factor**: Winning trade value / Losing trade value
- **Sharpe Ratio**: Risk-adjusted returns (higher = better)
- **Max Drawdown %**: Largest peak-to-trough decline

**Important notes:**
- Uses chronological split (not random) - tests on future data
- Includes realistic trading fees
- Tracks equity curve for drawdown analysis
- Results show historical performance only - not guaranteed future results

## Backtest Parameters

- **Initial Capital**: $10,000
- **Position Size**: 10% of capital per trade
- **Trading Fee**: 5 basis points (0.05%) on entry and exit
- **Entry Signal**: Model prediction = 1 (bullish)
- **Exit Signal**: Model prediction = 0 (bearish) or end of test period
- **Data Split**: 80% training, 20% testing (chronological order)

## Key Metrics

| Metric | Description |
|--------|-------------|
| **Total Return %** | Overall profit/loss as percentage of initial capital |
| **Total P&L (net)** | Actual dollar profit/loss after all fees |
| **Total Fees Paid** | Trading fees deducted from positions |
| **Win Rate** | % of closed trades that were profitable |
| **Avg Win / Avg Loss** | Average profit and loss per winning/losing trade |
| **Profit Factor** | Ratio of total winning trades to losing trades (>1 is profitable) |
| **Sharpe Ratio** | Risk-adjusted return (higher is better, >1 is good) |
| **Max Drawdown %** | Largest peak-to-trough decline during backtest |

## Project Structure

```
hyperliquid-predictor/
├── main.py                    # Main application - data fetching, training, backtesting, predictions
├── tune.py                    # CLI tool for hyperparameter tuning via Optuna
├── hyperparameter_tuning.py   # Reusable module for Bayesian hyperparameter optimization
├── pyproject.toml             # Project dependencies (UV)
├── uv.lock                    # Locked dependency versions
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## File Descriptions

- **main.py** (226 lines)
  - Fetches 200 days of OHLCV data from Hyperliquid
  - Computes technical indicators (RSI, volume change)
  - Trains XGBoost classifier on historical data
  - Runs comprehensive backtest with P&L simulation
  - Generates real-time predictions
  - Reports detailed backtest metrics

- **tune.py** (128 lines)
  - CLI interface for hyperparameter optimization
  - Uses Optuna's Bayesian optimization (TPE sampler)
  - Configurable trials and cross-validation folds
  - Optional parameter export to JSON
  - Full accuracy reporting and best parameter display

- **hyperparameter_tuning.py** (104 lines)
  - Core optimization logic with Optuna integration
  - Objective function for cross-validation scoring
  - Model training with optimized parameters
  - Reusable for future ML experiments

## Dependencies

- **ccxt** ≥4.0.0: Cryptocurrency exchange APIs
- **pandas** ≥2.0.0: Data manipulation and analysis
- **numpy** ≥1.24.0: Numerical computing
- **xgboost** ≥2.0.0: Gradient boosting machine learning
- **scikit-learn** ≥1.3.0: Cross-validation and metrics
- **optuna** ≥3.0.0: Bayesian hyperparameter optimization

## Disclaimer

This is an educational project for learning purposes. Cryptocurrency trading carries significant risk. Past performance does not guarantee future results. Never trade with capital you cannot afford to lose.

## Future Enhancements

- [ ] Real-time trading integration with Hyperliquid API
- [ ] Risk management features (stop-loss, take-profit levels)
- [ ] Position sizing based on volatility (Kelly Criterion, ATR-based)
- [ ] Additional technical indicators (MACD, Bollinger Bands, Volume Profile)
- [ ] Walk-forward backtesting for more realistic evaluation
- [ ] Multi-timeframe analysis (1h, 4h, 1d)
- [ ] Ensemble methods combining multiple models
- [ ] Feature importance analysis and visualization
- [ ] Sentiment/macro data integration
- [ ] Trade history export (CSV/JSON)
- [ ] Equity curve visualization
- [ ] Monthly/weekly P&L reporting
- [ ] Model performance metrics (precision, recall, F1-score)
- [ ] Correlation analysis with other assets

## License

MIT License
