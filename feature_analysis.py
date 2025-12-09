"""
Feature Importance Analysis Tool

Trains a model and extracts feature importance scores to understand
which features contribute most to predictions.
"""

import argparse

import pandas as pd
from xgboost import XGBClassifier

import config
from data.fetcher import fetch_btc_data, fetch_funding_rate, fetch_ohlcv
from data.processor import create_features, create_triple_barrier_labels
from strategies.xgb_strategy import get_feature_importance, load_hyperparameters


def analyze_features(
    symbol: str,
    exchange: str = config.DEFAULT_EXCHANGE,
    limit: int | None = None,
    mode: str = config.DEFAULT_PREDICTION_MODE,
) -> pd.DataFrame:
    """
    Analyze feature importance for a given symbol.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        exchange: Exchange name
        limit: Data limit in days
        mode: Prediction mode ('short', 'medium', or 'long')

    Returns:
        DataFrame with feature importance scores
    """
    # Get prediction mode parameters
    mode_params = config.get_prediction_mode_params(mode)

    print(f"\n{'=' * 70}")
    print(f"Feature Importance Analysis: {symbol}")
    print(f"Prediction Mode: {mode_params['name']}")
    print(f"{'=' * 70}\n")

    # Fetch data
    print(f"Fetching data for {symbol}...")
    df = fetch_ohlcv(symbol, exchange, limit or 365)

    if df.empty:
        print(f"Error: No data fetched for {symbol}")
        return pd.DataFrame()

    # Fetch BTC and funding data
    btc_df = fetch_btc_data(exchange, limit or 365) if "BTC" not in symbol.upper() else df
    funding_df = fetch_funding_rate(symbol, exchange, limit or 365)

    # Create features
    print("Creating features...")
    df = create_features(df, btc_df, funding_df)
    df["target"] = create_triple_barrier_labels(
        df,
        profit_pct=mode_params["profit_pct"],
        loss_pct=mode_params["loss_pct"],
        time_horizon=mode_params["time_horizon"],
    )
    # Remap labels from {-1, 0, 1} to {0, 1, 2} for XGBoost
    df["target"] = df["target"].map({-1: 0, 0: 1, 1: 2})
    df = df.dropna()

    # Prepare features and labels
    X = df[config.FEATURE_COLUMNS].copy()
    y = df["target"].copy()

    if len(y) < config.MIN_SAMPLES_FOR_TRAINING:
        print(f"Error: Not enough samples ({len(y)}) for training")
        return pd.DataFrame()

    # Load hyperparameters
    hyperparams = load_hyperparameters(symbol, verbose=True)
    hyperparams["random_state"] = config.RANDOM_STATE
    hyperparams["objective"] = "multi:softprob"
    hyperparams["num_class"] = 3

    # Train full model
    print(f"\nTraining model on {len(X):,} samples...")
    model = XGBClassifier(**hyperparams)
    model.fit(X, y)

    # Get feature importance
    importance_df = get_feature_importance(model, config.FEATURE_COLUMNS)

    # Display results
    print(f"\n{'=' * 70}")
    print("Feature Importance Rankings")
    print(f"{'=' * 70}")
    print(f"{'Feature':<25} {'Importance':>12}  {'% of Total':>10}")
    print(f"{'-' * 70}")

    total_importance = importance_df["importance"].sum()
    for idx, row in importance_df.iterrows():
        pct = (row["importance"] / total_importance) * 100
        print(f"{row['feature']:<25} {row['importance']:>12.6f}  {pct:>9.1f}%")

    print(f"{'-' * 70}")
    print(f"{'TOTAL':<25} {total_importance:>12.6f}  {100.0:>9.1f}%")
    print(f"{'=' * 70}\n")

    # Save to file
    safe_symbol = symbol.replace("/", "_")
    output_file = f"data/feature_importance_{safe_symbol}.csv"
    importance_df.to_csv(output_file, index=False)
    print(f"✓ Saved feature importance to {output_file}\n")

    # Provide recommendations
    print_recommendations(importance_df)

    return importance_df


def print_recommendations(importance_df: pd.DataFrame):
    """Print feature optimization recommendations."""
    print(f"{'=' * 70}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 70}\n")

    # Calculate cumulative importance
    importance_df["cumulative_pct"] = (
        importance_df["importance"].cumsum() / importance_df["importance"].sum() * 100
    )

    # Find how many features cover 80%, 90%, 95%
    n_80 = (importance_df["cumulative_pct"] <= 80).sum()
    n_90 = (importance_df["cumulative_pct"] <= 90).sum()
    n_95 = (importance_df["cumulative_pct"] <= 95).sum()

    print("Feature Coverage Analysis:")
    print(f"  • Top {n_80} features cover 80% of total importance")
    print(f"  • Top {n_90} features cover 90% of total importance")
    print(f"  • Top {n_95} features cover 95% of total importance")
    print(f"  • Total features: {len(importance_df)}\n")

    # Identify low-importance features
    threshold = 0.01  # 1% of total
    total_importance = importance_df["importance"].sum()
    low_importance = importance_df[importance_df["importance"] < (threshold * total_importance)]

    if not low_importance.empty:
        print("Low-importance features (< 1% each):")
        for _, row in low_importance.iterrows():
            pct = (row["importance"] / total_importance) * 100
            print(f"  • {row['feature']:<25} ({pct:.2f}%)")
        print()

    # Recommendations
    print("Suggested next steps:")
    print(f"  1. Consider using top {n_90} features (covers 90%)")
    print("  2. Run backtests comparing:")
    print(f"     - All {len(importance_df)} features (current)")
    print(f"     - Top {n_90} features (simplified)")
    print(f"     - Top {n_80} features (minimal)")
    print("  3. Compare Sharpe Ratio and total returns")
    print("  4. Update FEATURE_COLUMNS in config.py with optimal set")
    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze feature importance")
    parser.add_argument(
        "--symbol",
        type=str,
        default=config.DEFAULT_SYMBOL,
        help="Trading pair (e.g., BTC/USDT)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=config.DEFAULT_EXCHANGE,
        help="Exchange name",
    )
    parser.add_argument("--limit", type=int, help="Data limit in days")
    parser.add_argument(
        "--mode",
        type=str,
        default=config.DEFAULT_PREDICTION_MODE,
        choices=["short", "medium", "long"],
        help="Prediction mode: short (1-3d), medium (3-5d), long (5-10d)",
    )

    args = parser.parse_args()

    # Determine data limit
    limit = config.get_data_limit(args.symbol, args.limit)

    # Run analysis
    # Determine data limit
    limit = config.get_data_limit(args.symbol, args.limit)

    # Run analysis
    analyze_features(
        symbol=args.symbol,
        exchange=args.exchange,
        limit=limit,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
