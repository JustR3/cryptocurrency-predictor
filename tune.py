#!/usr/bin/env python3
"""
Hyperparameter tuning script using Optuna.
Optimizes for Sharpe Ratio instead of accuracy.
"""

import argparse

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import config
from data.fetcher import fetch_btc_data, fetch_funding_rate, fetch_ohlcv
from data.processor import create_features, create_triple_barrier_labels
from strategies.xgb_strategy import save_hyperparameters


def calculate_sharpe_ratio_cv(model, X, y, n_splits=5):
    """
    Calculate Sharpe ratio via time series cross-validation.

    Simulates a simple long-only strategy based on predictions.
    Labels: 0 = loss, 1 = neutral, 2 = profit
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    sharpe_ratios = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Check if training fold has at least 2 classes
        unique_train_classes = np.unique(y_train)
        if len(unique_train_classes) < 2:
            continue  # Skip this fold

        # Use LabelEncoder to handle non-consecutive classes
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)

        # Fit model with encoded labels
        model.fit(X_train, y_train_encoded)
        predictions_encoded = model.predict(X_val)

        # Decode predictions back to original labels
        predictions = le.inverse_transform(predictions_encoded)

        # Simple strategy: long when prediction is 2 (profit)
        strategy_returns = []
        for i, pred in enumerate(predictions):
            if i < len(y_val):
                actual_label = y_val.iloc[i]
                # Simulate return based on prediction correctness
                # pred=2 (profit), actual=2 (profit) -> Win
                if pred == 2 and actual_label == 2:
                    strategy_returns.append(0.02)  # Win
                # pred=2 (profit), actual=0 (loss) -> Loss
                elif pred == 2 and actual_label == 0:
                    strategy_returns.append(-0.01)  # Loss
                else:
                    strategy_returns.append(0.0)  # No trade

        if len(strategy_returns) > 0:
            returns_array = np.array(strategy_returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            sharpe = (mean_return / (std_return + 1e-6)) * np.sqrt(252)
            sharpe_ratios.append(sharpe)

    return np.mean(sharpe_ratios) if sharpe_ratios else 0.0


def objective(trial, X_train, y_train, n_splits=5):
    """Optuna objective function optimizing Sharpe Ratio."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": config.RANDOM_STATE,
    }

    model = XGBClassifier(**params)
    sharpe = calculate_sharpe_ratio_cv(model, X_train, y_train, n_splits)

    return float(sharpe)


def main():
    parser = argparse.ArgumentParser(description="Tune hyperparameters for Sharpe Ratio")
    parser.add_argument("--symbol", type=str, default=config.DEFAULT_SYMBOL, help="Trading pair")
    parser.add_argument("--exchange", type=str, default=config.DEFAULT_EXCHANGE, help="Exchange")
    parser.add_argument("--limit", type=int, default=None, help="Days of data")
    parser.add_argument("--trials", type=int, default=config.OPTUNA_N_TRIALS, help="Trials")
    parser.add_argument("--folds", type=int, default=config.OPTUNA_N_FOLDS, help="CV folds")
    parser.add_argument("--save", action="store_true", help="Save best parameters")
    parser.add_argument(
        "--mode",
        type=str,
        default=config.DEFAULT_PREDICTION_MODE,
        choices=["short", "medium", "long"],
        help="Prediction mode: short (1-3d), medium (3-5d), long (5-10d)",
    )
    args = parser.parse_args()

    # Get prediction mode parameters
    mode_params = config.get_prediction_mode_params(args.mode)

    print(f"\n{'=' * 60}")
    print(f"HYPERPARAMETER TUNING - {args.symbol}")
    print(f"Optimizing for: {config.OPTUNA_METRIC.upper()}")
    print(f"Prediction Mode: {mode_params['name']}")
    print(f"{'=' * 60}\n")

    # Fetch and prepare data
    print(f"Fetching {args.symbol} data...")
    df = fetch_ohlcv(args.symbol, args.exchange, args.limit)
    btc_df = fetch_btc_data(args.exchange, args.limit) if "BTC" not in args.symbol else df
    funding_df = fetch_funding_rate(args.symbol, args.exchange, args.limit)

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

    X = df[config.FEATURE_COLUMNS]
    y = df["target"]

    split_idx = int(len(df) * (1 - config.TEST_SIZE))
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    # Check class distribution
    unique_classes = np.unique(y_train)
    print(f"Training set: {len(X_train)} samples")
    print(f"Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Validate we have at least 2 classes for meaningful training
    if len(unique_classes) < 2:
        print(f"❌ Error: Only {len(unique_classes)} class in training data.")
        print("Need at least 2 classes for classification. Try:")
        print("  - Increase --limit for more data")
        print("  - Adjust barrier parameters in config.py")
        return

    # If missing middle class, XGBoost will handle it but warn user
    if len(unique_classes) < 3:
        missing_classes = set([0, 1, 2]) - set(unique_classes)
        print(f"⚠ Warning: Missing classes {missing_classes} in training data")
        print("Optimization will continue but results may be suboptimal.\n")

    print(f"Starting optimization ({args.trials} trials)...\n")  # Run optimization
    sampler = TPESampler(seed=config.RANDOM_STATE)
    pruner = MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, args.folds),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    print(f"\n{'=' * 60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best Sharpe Ratio: {study.best_value:.4f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    if args.save:
        save_hyperparameters(study.best_params, args.symbol)
        print(f"\nTo use: python main.py --symbol {args.symbol} --exchange {args.exchange}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
