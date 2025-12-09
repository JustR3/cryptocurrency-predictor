# GitHub Copilot Instructions

## Agentic Workflow Rules

### 0. Package Management & Script Execution
**ALWAYS use UV for package management and script execution:**

**Package Management:**
- Use `uv add <package>` to install packages
- Use `uv remove <package>` to uninstall packages
- Use `uv sync` to synchronize dependencies
- Never use `pip install`, `pip uninstall`, or similar pip commands

**Script Execution:**
- Use `uv run <script.py>` to run Python scripts
- Never use `python <script.py>` or `python3 <script.py>`
- Examples: `uv run main.py`, `uv run tune.py --symbol BTC/USDT`

### 1. Task Execution Process
For every task in [`TODO.md`](../TODO.md):

1. **READ** the roadmap and understand the task
2. **PLAN** the changes (explain what you'll do)
3. **EXECUTE** the code modifications
4. **VERIFY** by running the code:
   - Run relevant tests
   - Execute affected scripts
   - Confirm no errors or regressions
5. **COMMIT** with a conventional commit message

### 2. Commit Message Standards
Use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code restructuring without behavior change
- `test:` Adding or updating tests
- `docs:` Documentation changes
- `chore:` Maintenance tasks

**Examples**:
```
feat: add probabilistic target prediction
fix: correct Triple Barrier label calculation
refactor: integrate RiskManager into backtest engine
test: add unit tests for data processor
docs: update README with new features
chore: remove deprecated risk_management.py
```

### 3. Code Standards

#### Type Hints (MANDATORY)
All functions must have type hints:

```python
def fetch_ohlcv(
    symbol: str,
    exchange_name: str = config.DEFAULT_EXCHANGE,
    limit: int | None = None
) -> pd.DataFrame:
    """Docstring here."""
    pass
```

#### Configuration Management
- **NO** hardcoded values in code
- All settings go in [`config.py`](../config.py)
- Use `config.CONSTANT_NAME` everywhere

**Bad**:
```python
if volatility > 0.15:  # What is 0.15?
    pass
```

**Good**:
```python
if volatility > config.TARGET_VOLATILITY:
    pass
```

#### Error Handling
- Use specific exceptions, not bare `except:`
- Provide helpful error messages
- Log warnings for recoverable issues

```python
try:
    data = fetch_data(symbol)
except ccxt.NetworkError as e:
    raise ValueError(f"Network error fetching {symbol}: {e}")
except ccxt.ExchangeError as e:
    print(f"Warning: Exchange error for {symbol}, trying fallback: {e}")
    data = fetch_fallback(symbol)
```

#### Documentation
- All public functions have docstrings
- Docstrings follow Google style:

```python
def calculate_position_size(
    capital: float,
    volatility: float,
    win_rate: float
) -> float:
    """
    Calculate optimal position size using Kelly criterion.

    Args:
        capital: Current account capital in USD
        volatility: Annualized volatility (e.g., 0.15 = 15%)
        win_rate: Historical win rate (0-1)

    Returns:
        Position size in dollars

    Raises:
        ValueError: If volatility is negative or win_rate not in [0, 1]
    """
    pass
```

### 4. Testing Requirements
Before marking a task complete:

- [ ] Code runs without errors
- [ ] No new warnings introduced
- [ ] Existing functionality unchanged (unless intentional)
- [ ] Type hints pass mypy validation (if configured)
- [ ] Ruff linting passes

### 5. File Organization
Current structure (DO NOT modify without discussion):

```
.
├── config.py              # All settings
├── main.py                # Main orchestration
├── tune.py                # Hyperparameter optimization
├── data/
│   ├── fetcher.py        # CCXT data fetching
│   ├── processor.py      # Feature engineering
│   └── storage.py        # Caching
├── strategies/
│   └── xgb_strategy.py   # XGBoost model
├── backtest/
│   └── engine.py         # Backtesting logic
└── risk_management.py    # (TO BE INTEGRATED & DELETED)
```

### 6. Priorities
1. **Correctness** > Speed > Style
2. **Readability** > Cleverness
3. **Type Safety** > Dynamic typing
4. **Explicit** > Implicit

---

## Current Focus
Refer to [`TODO.md`](../TODO.md) for active tasks. Start with **Phase 1: Audit Data Layer**.
