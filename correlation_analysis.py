"""
Correlation Analysis Tool for Hyperliquid Predictor.
Identifies multicollinearity and redundant features.
"""

import argparse

import numpy as np
import pandas as pd

import config
from data.fetcher import fetch_ohlcv
from data.processor import create_features

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


def analyze_correlation(
    symbol: str = config.DEFAULT_SYMBOL,
    exchange_name: str = config.DEFAULT_EXCHANGE,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Calculate correlation matrix for all features.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        exchange_name: Exchange name
        limit: Number of days of data (None = use config default)

    Returns:
        Correlation matrix DataFrame
    """
    print(f"📊 Correlation Analysis for {symbol}")
    print("=" * 50)

    # Fetch and process data
    print(f"\n1. Fetching data from {exchange_name}...")
    limit = limit if limit is not None else config.get_data_limit(symbol)
    df = fetch_ohlcv(symbol, exchange_name, limit)
    print(f"   ✓ Loaded {len(df)} rows")

    print("\n2. Adding technical indicators and features...")
    df = create_features(df, btc_df=None, funding_df=None)
    print(f"   ✓ {len(config.FEATURE_COLUMNS)} features calculated")

    # Calculate correlation matrix
    print("\n3. Calculating correlation matrix...")
    feature_data = df[config.FEATURE_COLUMNS].dropna()
    corr_matrix = feature_data.corr()

    # Find highly correlated pairs
    print("\n4. Identifying highly correlated feature pairs (|r| > 0.90):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = float(corr_matrix.iloc[i, j])
            if abs(corr_value) > 0.90:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                high_corr_pairs.append((feat1, feat2, corr_value))
                print(f"   • {feat1:20s} ↔ {feat2:20s}  r = {corr_value:+.3f}")

    if not high_corr_pairs:
        print("   ✓ No highly correlated pairs found (all |r| ≤ 0.90)")

    # Find moderately correlated pairs (0.70 - 0.90)
    print("\n5. Moderately correlated feature pairs (0.70 < |r| ≤ 0.90):")
    moderate_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = float(corr_matrix.iloc[i, j])
            if 0.70 < abs(corr_value) <= 0.90:
                feat1 = corr_matrix.columns[i]
                feat2 = corr_matrix.columns[j]
                moderate_corr_pairs.append((feat1, feat2, corr_value))
                print(f"   • {feat1:20s} ↔ {feat2:20s}  r = {corr_value:+.3f}")

    if not moderate_corr_pairs:
        print("   ✓ No moderately correlated pairs found")

    # Summary statistics
    print("\n6. Correlation Summary Statistics:")
    upper_triangle = np.triu(np.abs(corr_matrix.values), k=1)
    correlations = upper_triangle[(upper_triangle != 0) & ~np.isnan(upper_triangle)]
    if len(correlations) > 0:
        print(f"   • Mean correlation:   {np.mean(correlations):.3f}")
        print(f"   • Median correlation: {np.median(correlations):.3f}")
        print(f"   • Max correlation:    {np.max(correlations):.3f}")
        print(f"   • Min correlation:    {np.min(correlations):.3f}")
    else:
        print("   ✓ No correlations to analyze")

    # Recommendations
    print("\n7. Recommendations:")
    if high_corr_pairs:
        print("   ⚠️  High multicollinearity detected!")
        print("   → Consider removing one feature from each highly correlated pair")
        print("   → Use VIF analysis to identify which features to drop")
    elif moderate_corr_pairs:
        print("   ⚠️  Moderate correlation present")
        print("   → Features may be redundant but not critically")
        print("   → Consider testing feature removal impact via backtesting")
    else:
        print("   ✓ Low multicollinearity - features are relatively independent")

    print("\n" + "=" * 50)
    return corr_matrix


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    symbol: str,
    save: bool = False,
) -> None:
    """
    Create and display correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame
        symbol: Trading pair for title
        save: Whether to save the plot to file
    """
    plt.figure(figsize=(12, 10))

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1,
        vmax=1,
    )

    plt.title(f"Feature Correlation Matrix - {symbol}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        filename = f"correlation_heatmap_{symbol.replace('/', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\n📊 Heatmap saved to: {filename}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature correlations and identify multicollinearity"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=config.DEFAULT_SYMBOL,
        help=f"Trading pair (default: {config.DEFAULT_SYMBOL})",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=config.DEFAULT_EXCHANGE,
        help=f"Exchange name (default: {config.DEFAULT_EXCHANGE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of days of data (default: auto based on symbol)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display correlation heatmap",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save heatmap to file",
    )

    args = parser.parse_args()

    # Run correlation analysis
    corr_matrix = analyze_correlation(
        symbol=args.symbol,
        exchange_name=args.exchange,
        limit=args.limit,
    )

    # Plot if requested
    if args.plot or args.save:
        plot_correlation_heatmap(corr_matrix, args.symbol, args.save)


if __name__ == "__main__":
    main()
