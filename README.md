# Cryptocurrency Price Predictor

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active%20Development-yellow)

A machine learning trading system using XGBoost with the **Triple Barrier Method** for realistic labeling, multi-horizon prediction, and sophisticated risk management across 100+ cryptocurrency exchanges.

## 🎯 Key Features

- **Multi-Horizon Prediction**: Short (1-3d), Medium (3-5d), Long (5-10d) with adaptive targets
- **Triple Barrier Labeling**: Realistic trade outcomes vs. arbitrary thresholds
- **Feature Analysis Tools**: Identify important features and detect multicollinearity
- **Volatility Targeting**: Dynamic position sizing for consistent risk
- **Sharpe Ratio Optimization**: Tune for profitability, not just accuracy
- **Multi-Exchange Support**: Binance, Coinbase, Kraken, Bybit, OKX, Hyperliquid + 100 more
- **Realistic Costs**: 5 bps fees + 10 bps slippage modeling

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone https://github.com/JustR3/hyperliquid-predictor.git
cd hyperliquid-predictor

# Install UV package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync

# macOS only: Install OpenMP for XGBoost
brew install libomp
```

### Basic Usage

```bash
# Run backtest (default: BTC/USDT, medium-term)
uv run main.py

# Different prediction horizons
uv run main.py --symbol BTC/USDT --mode short   # 1-3 days
uv run main.py --symbol ETH/USDT --mode medium  # 3-5 days (default)
uv run main.py --symbol SOL/USDT --mode long    # 5-10 days

# Optimize hyperparameters
uv run tune.py --symbol BTC/USDT --mode short --trials 100 --save

# Analyze feature importance
uv run feature_analysis.py --symbol BTC/USDT --mode medium

# Check for multicollinearity
uv run correlation_analysis.py --symbol BTC/USDT --plot
```

## 📁 Project Structure

```
├── config.py                    # All settings and prediction modes
├── main.py                      # Main backtest orchestrator
├── tune.py                      # Hyperparameter optimization
├── feature_analysis.py          # Feature importance analysis
├── correlation_analysis.py      # Multicollinearity detection
├── data/
│   ├── fetcher.py              # CCXT data fetching
│   ├── processor.py            # Features + Triple Barrier labels
│   └── storage.py              # Parquet caching
├── strategies/
│   └── xgb_strategy.py         # XGBoost model
└── backtest/
    └── engine.py               # Backtesting + risk management
```

## 🔬 How It Works

### Multi-Horizon Prediction Modes

| Mode | Horizon | Profit | Loss | Risk:Reward |
|------|---------|--------|------|-------------|
| Short | 3 days | 3% | 1.5% | 2:1 |
| Medium | 5 days | 5% | 2.5% | 2:1 |
| Long | 10 days | 8% | 4% | 2:1 |

### Triple Barrier Method

For each timestamp, observe which happens first:
1. **Profit target hit** → Label = 1 (Long win)
2. **Stop loss hit** → Label = -1 (Long loss)
3. **Time expires** → Label = sign of return (Neutral)

More realistic than "will price go up tomorrow?" binary labels.

### Position Sizing

Combines volatility targeting + Kelly Criterion:
- Scale inversely with volatility (target: 15% annual)
- Kelly sizing based on historical win rate and avg win/loss
- Take minimum of both for conservative approach

### Features (15 total)

- **Momentum**: RSI, MACD, MACD histogram, momentum (3d/5d/7d)
- **Trend**: EMA flags (9/21/50), Bollinger Band position
- **Volatility**: ATR, ATR percentage
- **Volume**: Change, volume/SMA ratio
- **Market**: BTC beta (rolling correlation)

## 📊 Sample Output

```
BACKTEST RESULTS
Return: 23.4% | P&L: $2,345
Trades: 42 | Win Rate: 61.9%
Sharpe: 1.82 | Max DD: -8.3%

TOP FEATURES
atr_pct: 7.8%
macd_histogram: 7.7%
price_momentum_5d: 7.2%
```

## 🛠️ Advanced Usage

### Correlation Analysis

Check for feature redundancy:

```bash
# Analyze feature correlations
uv run correlation_analysis.py --symbol BTC/USDT

# Generate heatmap
uv run correlation_analysis.py --symbol ETH/USDT --plot --save
```

**Findings**: No critical multicollinearity (all |r| ≤ 0.90). Moderate correlations (0.70-0.86) in momentum and trend indicators suggest room for optimization.

### Custom Configuration

Edit `config.py` to modify:
- Barrier percentages per mode
- Risk management parameters (Kelly fraction, max position size)
- Technical indicator periods
- Feature set selection

## 🎯 Roadmap

### Current Focus
- [ ] VIF analysis for multicollinearity
- [ ] Permutation importance and SHAP values
- [ ] Feature subset optimization via backtesting

### Future Plans
- [ ] Probabilistic targets (quantile regression)
- [ ] Alternative models (LightGBM, ensemble methods)
- [ ] Cross-asset features (SPY, DXY, Gold)
- [ ] Online learning for incremental updates


## ⚠️ Disclaimer

**Educational and research purposes only. Not financial advice.**

- Cryptocurrency trading involves substantial risk of loss
- Past performance does not guarantee future results
- Only trade with capital you can afford to lose

## 📝 License

MIT License - see LICENSE file for details.

---

**Star ⭐ this repo if you find it useful!**
