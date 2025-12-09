"""
Hyperliquid Predictor - Main Entry Point
Orchestrates data fetching, model training, and backtesting.
"""

import argparse

import config
from backtest.engine import backtest_strategy, walk_forward_backtest
from data.fetcher import fetch_btc_data, fetch_funding_rate, fetch_ohlcv
from data.processor import create_features, create_triple_barrier_labels
from risk_management import RiskManager
from strategies.xgb_strategy import get_feature_importance, train_model


def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency price predictor")
    parser.add_argument("--symbol", type=str, default=config.DEFAULT_SYMBOL, help="Trading pair")
    parser.add_argument("--exchange", type=str, default=config.DEFAULT_EXCHANGE, help="Exchange")
    parser.add_argument("--limit", type=int, default=None, help="Days of historical data")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"CRYPTO PREDICTOR - {args.symbol}")
    print(f"{'=' * 60}")
    print(f"Exchange: {args.exchange}")
    print(f"Data: {config.get_data_limit(args.symbol, args.limit)} days")
    print(f"{'=' * 60}\n")

    # Fetch data
    print(f"Fetching {args.symbol} data...")
    df = fetch_ohlcv(args.symbol, args.exchange, args.limit)
    btc_df = fetch_btc_data(args.exchange, args.limit) if "BTC" not in args.symbol else df
    funding_df = fetch_funding_rate(args.symbol, args.exchange, args.limit)
    print(f"✓ Fetched {len(df)} days\n")

    # Create features
    print("Creating features...")
    df = create_features(df, btc_df, funding_df)
    df["target"] = create_triple_barrier_labels(df)
    # Remap labels from {-1, 0, 1} to {0, 1, 2} for XGBoost
    df["target"] = df["target"].map({-1: 0, 0: 1, 1: 2})
    df = df.dropna()
    print(f"✓ Created {len(config.FEATURE_COLUMNS)} features\n")

    # Train model
    print("Training model...")
    split_idx = int(len(df) * (1 - config.TEST_SIZE))
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    model, label_encoder = train_model(
        train_df[config.FEATURE_COLUMNS], train_df["target"], args.symbol
    )
    print(f"✓ Model trained\n")

    # Backtests
    risk_mgr = RiskManager()
    print("Running backtests...\n")

    standard_results = backtest_strategy(
        test_df, model, label_encoder, config.FEATURE_COLUMNS, risk_manager=risk_mgr
    )
    wf_results = walk_forward_backtest(df, config.FEATURE_COLUMNS, args.symbol)

    # Print results
    print_results("STANDARD BACKTEST", standard_results)
    print_results("WALK-FORWARD BACKTEST", wf_results)
    print_comparison(standard_results, wf_results)
    print_feature_importance(model, config.FEATURE_COLUMNS)


def print_results(title, results):
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")
    print(f"Return: {results['total_return_pct']:.2f}% | P&L: ${results['total_pnl']:.2f}")
    print(f"Trades: {results['num_trades']} | Win Rate: {results['win_rate_pct']:.1f}%")
    print(f"Sharpe: {results['sharpe_ratio']:.2f} | Max DD: {results['max_drawdown_pct']:.2f}%")


def print_comparison(std, wf):
    print(f"\n{'=' * 50}\nCOMPARISON\n{'=' * 50}")
    print(f"Return Diff: {wf['total_return_pct'] - std['total_return_pct']:+.2f}%")


def print_feature_importance(model, features):
    importance = get_feature_importance(model, features)
    print(f"\n{'=' * 50}\nTOP 5 FEATURES\n{'=' * 50}")
    for _, row in importance.head(5).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
