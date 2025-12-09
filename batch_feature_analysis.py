#!/usr/bin/env python3
"""
Batch Feature Analysis Script

Analyzes feature importance across multiple symbols and prediction modes.
Generates comprehensive reports for feature optimization decisions.
"""

import argparse

import pandas as pd

import config
from feature_analysis import analyze_features


def batch_analyze(
    symbols: list[str],
    modes: list[str],
    exchange: str = config.DEFAULT_EXCHANGE,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Run feature analysis on multiple symbols and modes.

    Args:
        symbols: List of trading pairs
        modes: List of prediction modes ('short', 'medium', 'long')
        exchange: Exchange name
        limit: Data limit in days

    Returns:
        DataFrame with aggregated results
    """
    print(f"\n{'=' * 70}")
    print(f"BATCH FEATURE ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Symbols: {len(symbols)}")
    print(f"Modes: {modes}")
    print(f"Total runs: {len(symbols) * len(modes)}")
    print(f"{'=' * 70}\n")

    all_results = []

    for symbol in symbols:
        for mode in modes:
            try:
                print(f"\n{'=' * 70}")
                print(f"Processing: {symbol} - {mode.upper()} mode")
                print(f"{'=' * 70}")

                # Determine data limit per symbol
                symbol_limit = limit if limit is not None else config.get_data_limit(symbol)

                # Run analysis
                importance_df = analyze_features(
                    symbol=symbol,
                    exchange=exchange,
                    limit=symbol_limit,
                    mode=mode,
                )

                if not importance_df.empty:
                    # Add metadata
                    importance_df["symbol"] = symbol
                    importance_df["mode"] = mode
                    all_results.append(importance_df)

                    print(f"✓ Completed: {symbol} - {mode}")
                else:
                    print(f"⚠ Skipped: {symbol} - {mode} (no data)")

            except Exception as e:
                print(f"❌ Error: {symbol} - {mode}: {e}")
                continue

    # Combine all results
    if not all_results:
        print("\n❌ No results collected!")
        return pd.DataFrame()

    combined_df = pd.concat(all_results, ignore_index=True)

    return combined_df


def generate_summary_report(results_df: pd.DataFrame) -> None:
    """
    Generate summary report from batch analysis results.

    Args:
        results_df: Combined results from all analyses
    """
    if results_df.empty:
        print("\n⚠ No data to summarize")
        return

    print(f"\n{'=' * 70}")
    print("COMPREHENSIVE FEATURE ANALYSIS SUMMARY")
    print(f"{'=' * 70}\n")

    # Overall feature importance (averaged across all symbols/modes)
    avg_importance = results_df.groupby("feature")["importance"].mean().sort_values(ascending=False)

    print("Average Feature Importance (across all symbols & modes):")
    print(f"{'=' * 70}")
    print(f"{'Feature':<25} {'Avg Importance':>15}  {'Rank':>8}")
    print(f"{'-' * 70}")
    for rank, (feature, importance) in enumerate(avg_importance.items(), 1):
        print(f"{feature:<25} {importance:>15.6f}  {rank:>8}")
    print(f"{'=' * 70}\n")

    # Identify consistently low-importance features
    print("Features with <1% importance in ALL runs:")
    print(f"{'-' * 70}")
    low_features = set()
    for feature in results_df["feature"].unique():
        feature_data = results_df[results_df["feature"] == feature]
        max_importance = feature_data["importance"].max()
        if max_importance < 0.01:
            low_features.add(feature)
            count = len(feature_data)
            print(f"  • {feature:<25} (max: {max_importance:.4f} across {count} runs)")

    if not low_features:
        print("  None - all features have >1% importance in at least one run")
    print(f"{'-' * 70}\n")

    # Features to consider removing
    print("RECOMMENDATIONS:")
    print(f"{'=' * 70}")

    if low_features:
        print(f"✓ Remove these {len(low_features)} features (consistently useless):")
        for feature in sorted(low_features):
            print(f"  • {feature}")
        print()

    # Top features (>5% avg importance)
    top_features = avg_importance[avg_importance > 0.05]
    print(f"✓ Keep these {len(top_features)} core features (>5% avg importance):")
    for feature in top_features.index:
        print(f"  • {feature:<25} ({avg_importance[feature]:.1%})")
    print()

    # Calculate optimal feature count
    cumsum = avg_importance.cumsum() / avg_importance.sum()
    n_80 = (cumsum <= 0.80).sum()
    n_90 = (cumsum <= 0.90).sum()

    print(f"Feature Coverage:")
    print(f"  • Top {n_80} features cover 80% of importance")
    print(f"  • Top {n_90} features cover 90% of importance")
    print(f"  • Total features: {len(avg_importance)}")
    print()

    print("Next Steps:")
    print(f"  1. Remove {len(low_features)} low-importance features from config.py")
    print(f"  2. Test optimized set (top {n_90} features) via backtesting")
    print("  3. Compare Sharpe Ratios: All vs Optimized")
    print(f"{'=' * 70}\n")

    # Save summary
    summary_file = "data/feature_analysis_summary.csv"
    avg_importance.to_csv(summary_file)
    print(f"✓ Saved summary to {summary_file}")

    # Save full results
    results_file = "data/feature_analysis_full_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"✓ Saved full results to {results_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch feature analysis across multiple symbols")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=[
            "BTC/USDT",
            "ETH/USDC",
            "SOL/USDT",
            "BNB/USDT",
            "ADA/USDT",
            "AVAX/USDT",
            "POL/USDT",
            "LINK/USDT",
        ],
        help="List of trading pairs to analyze",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["short", "medium", "long"],
        choices=["short", "medium", "long"],
        help="Prediction modes to test",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=config.DEFAULT_EXCHANGE,
        help="Exchange name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Override data limit (uses smart defaults if not specified)",
    )

    args = parser.parse_args()

    # Run batch analysis
    results_df = batch_analyze(
        symbols=args.symbols,
        modes=args.modes,
        exchange=args.exchange,
        limit=args.limit,
    )

    # Generate summary report
    if not results_df.empty:
        generate_summary_report(results_df)
    else:
        print("\n❌ Batch analysis failed - no results to report")


if __name__ == "__main__":
    main()
