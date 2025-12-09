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
from xgboost import XGBClassifier

import config
from data.fetcher import fetch_btc_data, fetch_funding_rate, fetch_ohlcv
from data.processor import create_features, create_triple_barrier_labels
from strategies.xgb_strategy import save_hyperparameters


def calculate_sharpe_ratio_cv(model, X, y, n_splits=5):
    """
    Calculate Sharpe ratio via time series cross-validation.

    Simulates a simple long-only strategy based on predictions.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    sharpe_ratios = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        # Simple strategy: long when prediction is positive
        strategy_returns = []
        for i, pred in enumerate(predictions):
            if i < len(y_val) - 1:
                actual_label = y_val.iloc[i]
                # Simulate return based on prediction correctness
                if pred == 1 and actual_label == 1:
                    strategy_returns.append(0.02)  # Win
                elif pred == 1 and actual_label == -1:
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
        "objective": "multi:softmax",
        "num_class": 3,  # -1, 0, 1
    }

    model = XGBClassifier(**params)
    sharpe = calculate_sharpe_ratio_cv(model, X_train, y_train, n_splits)

    return sharpe


def main():
    parser = argparse.ArgumentParser(description="Tune hyperparameters for Sharpe Ratio")
    parser.add_argument("--symbol", type=str, default=config.DEFAULT_SYMBOL, help="Trading pair")
    parser.add_argument("--exchange", type=str, default=config.DEFAULT_EXCHANGE, help="Exchange")
    parser.add_argument("--limit", type=int, default=None, help="Days of data")
    parser.add_argument("--trials", type=int, default=config.OPTUNA_N_TRIALS, help="Trials")
    parser.add_argument("--folds", type=int, default=config.OPTUNA_N_FOLDS, help="CV folds")
    parser.add_argument("--save", action="store_true", help="Save best parameters")
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"HYPERPARAMETER TUNING - {args.symbol}")
    print(f"Optimizing for: {config.OPTUNA_METRIC.upper()}")
    print(f"{'=' * 60}\n")

    # Fetch and prepare data
    print(f"Fetching {args.symbol} data...")
    df = fetch_ohlcv(args.symbol, args.exchange, args.limit)
    btc_df = fetch_btc_data(args.exchange, args.limit) if "BTC" not in args.symbol else df
    funding_df = fetch_funding_rate(args.symbol, args.exchange, args.limit)

    df = create_features(df, btc_df, funding_df)
    df["target"] = create_triple_barrier_labels(df)
    # Remap labels from {-1, 0, 1} to {0, 1, 2} for XGBoost
    df["target"] = df["target"].map({-1: 0, 0: 1, 1: 2})
    df = df.dropna()

    X = df[config.FEATURE_COLUMNS]
    y = df["target"]

    split_idx = int(len(df) * (1 - config.TEST_SIZE))
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]

    print(f"Training set: {len(X_train)} samples")
    print(f"Starting optimization ({args.trials} trials)...\n")

    # Run optimization
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
