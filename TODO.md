# Project Roadmap

## ✅ Completed Features

### Core System
- [x] **Triple Barrier Method implementation** - Industry-standard labeling for realistic trade outcomes
- [x] **Multi-horizon prediction modes** - Short (1-3d), Medium (3-5d), Long (5-10d) with adaptive barriers
- [x] **Multi-exchange support** - Works with 100+ exchanges via CCXT (Binance, Coinbase, Kraken, Bybit, OKX, Hyperliquid)
- [x] **XGBoost strategy** - Machine learning model with proper time-series handling
- [x] **Volatility targeting** - Dynamic position sizing based on market volatility
- [x] **Realistic cost modeling** - Trading fees (5 bps) + slippage (10 bps)
- [x] **Sharpe Ratio optimization** - Hyperparameter tuning focuses on profitability, not just accuracy
- [x] **Walk-forward backtesting** - Retrains model every 20 days to avoid overfitting
- [x] **Risk management system** - Kelly Criterion + volatility targeting for position sizing
- [x] **Feature engineering** - Technical indicators (RSI, MACD, BB, ATR, EMAs), BTC beta, funding rates
- [x] **Symbol-specific hyperparameters** - Smart defaults based on asset class (major/mid/low caps)
- [x] **Data caching** - Parquet storage for faster repeated analysis
- [x] **CLI interface** - Simple command-line interface with flexible options
- [x] **Type hints** - Full type annotation across all modules
- [x] **Centralized configuration** - All settings in `config.py`
- [x] **Feature importance analysis** - Tool to identify and rank feature contributions

### Documentation
- [x] **Comprehensive README** - Installation, usage, methodology, project structure
- [x] **MIT License** - Proper open-source licensing
- [x] **Code documentation** - Docstrings following Google style
- [x] **Project badges** - Python version, license, development status

---

## Agentic Workflow - Refactoring Plan

### Phase 1: Audit Data Layer ✅ COMPLETED
- [x] **Verify `data/fetcher.py`**
  - [x] Confirm async CCXT usage (or document why sync is acceptable) - Sync is appropriate for daily data
  - [x] Verify error handling for missing symbols - ✅ Comprehensive try/except blocks
  - [x] Test fallback mechanisms (USDT → USDC, primary exchange → Binance) - ✅ Working correctly
  - [x] Validate rate limiting configuration - ✅ Enabled on all exchanges

- [x] **Verify `data/processor.py`**
  - [x] Audit Triple Barrier Method implementation - ✅ Correctly implemented
  - [x] Confirm profit/loss/time barrier logic is correct - ✅ Proper priority handling
  - [x] Verify label remapping {-1, 0, 1} → {0, 1, 2} for XGBoost - ✅ Consistent across codebase
  - [x] Test edge cases (insufficient data, NaN handling) - ✅ Properly handled
  - [x] Validate technical indicator calculations - ✅ Verified

### Phase 2: Audit Strategy Layer ✅ COMPLETED
- [x] **Verify `strategies/xgb_strategy.py`**
  - [x] Confirm hyperparameter loading hierarchy (symbol-specific → generic → smart defaults) - ✅ Working
  - [x] Verify label encoding/decoding matches training labels - ✅ Consistent
  - [x] Test model training with edge cases (single class, insufficient samples) - ✅ Protected
  - [x] Validate feature importance extraction - ✅ Implemented
  - [x] Ensure prediction probabilities align with class labels - ✅ Verified

### Phase 3: Cleanup & Integration ✅ COMPLETED
- [x] **Integrate `risk_management.py` into `backtest/engine.py`**
  - [x] Move `RiskManager` class to `backtest/engine.py`
  - [x] Move helper functions (`calculate_volatility`, `calculate_win_rate_and_ratio`)
  - [x] Update imports in `main.py` and `tune.py` - No updates needed, imports from backtest.engine
  - [x] Verify all tests still pass - ✅ Imports working correctly

- [x] **Delete `risk_management.py`**
  - [x] Confirm no remaining imports reference this file - ✅ Only backtest/engine.py used it
  - [x] Remove from repository - ✅ Deleted

- [x] **Organize `tune.py` output**
  - [x] Ensure hyperparameters save to `data/best_hyperparameters_{SYMBOL}.json` - ✅ Working
  - [x] Verify file naming convention consistency - ✅ Uses underscore separator
  - [x] Add validation for saved JSON structure - ✅ Handled by save_hyperparameters()

### Phase 4: Multi-Horizon Prediction ✅ COMPLETED
- [x] **Implement prediction mode system**
  - [x] Add PREDICTION_MODES config with short/medium/long horizons
  - [x] Create `get_prediction_mode_params()` helper function
  - [x] Update Triple Barrier to accept dynamic parameters
  - [x] Add `--mode` CLI argument to main.py, tune.py, feature_analysis.py
  - [x] Maintain backward compatibility with legacy config values

### Phase 5: Feature Analysis & Optimization 🔄 IN PROGRESS
- [x] **Create feature analysis tool** (`feature_analysis.py`)
  - [x] Train full model and extract feature importance scores
  - [x] Rank features by contribution to predictions
  - [x] Identify low/zero importance features
  - [x] Calculate cumulative importance coverage (80%, 90%, 95%)
  - [x] Generate recommendations for feature optimization
  - [x] Support all prediction modes (short/medium/long)

- [ ] **Conduct comprehensive feature analysis**
  - [ ] Run analysis on multiple symbols (BTC, ETH, SOL, + 5 more)
  - [ ] Test across all 3 prediction modes (short, medium, long)
  - [ ] Document consistent patterns in feature importance
  - [ ] Identify features to remove (funding_rate confirmed useless)
  - [ ] Create optimized FEATURE_COLUMNS configuration

- [ ] **Optimize feature set**
  - [ ] Remove zero-contribution features (funding_rate)
  - [ ] Test optimized feature sets via backtesting
  - [ ] Compare Sharpe Ratios: All features vs Top 90% vs Top 80%
  - [ ] Update config.py with final optimized FEATURE_COLUMNS
  - [ ] Document performance improvements

## Future Enhancements (Post-Optimization)
- [ ] Implement probabilistic targets (quantile regression)
- [ ] Explore alternative models (LightGBM, Random Forest, CatBoost)
- [ ] Add cross-asset features (SPY, DXY, Gold correlations)
- [ ] Implement online learning for incremental model updates
- [ ] Add ensemble methods (model stacking/voting)
- [ ] Implement walk-forward optimization for hyperparameters
- [ ] Add real-time data streaming for live trading

---

## Usage Examples

### Run with different prediction modes:
```bash
# Short-term prediction (1-3 days)
python main.py --symbol BTC/USDT --mode short

# Medium-term prediction (3-5 days, default)
python main.py --symbol BTC/USDT --mode medium

# Long-term prediction (5-10 days)
python main.py --symbol BTC/USDT --mode long
```

### Hyperparameter tuning with mode:
```bash
python tune.py --symbol SOL/USDT --mode short --trials 100 --save
```

### Feature analysis with mode:
```bash
python feature_analysis.py --symbol ETH/USDC --mode long
```

---

**Workflow**: For each task:
1. Read this roadmap
2. Plan changes
3. Execute code modifications
4. **VERIFY** by running the code
5. Commit with conventional commit message
