"""
Hyperparameter tuning module for XGBoost model optimization.

This module uses Optuna for efficient hyperparameter search.
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


def objective(trial, X_train, y_train, n_splits=5):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training target
        n_splits: Number of CV folds

    Returns:
        Cross-validation score
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42,
        # Multi-class classification (4 regime classes)
        "objective": "multi:softmax",
        "num_class": 4,
    }

    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=n_splits, scoring="accuracy")

    return scores.mean()


def tune_hyperparameters(X_train, y_train, n_trials=100, n_splits=5, verbose=True):
    """
    Tune XGBoost hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials
        n_splits: Number of cross-validation folds
        verbose: Print optimization progress

    Returns:
        Dictionary with best hyperparameters
    """
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_warmup_steps=10)

    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    # Disable Optuna logging if not verbose
    optuna_logger = optuna.logging.get_logger("optuna")
    if not verbose:
        optuna_logger.setLevel(optuna.logging.WARNING)

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, n_splits),
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    if verbose:
        print("\nOptimization complete!")
        print(f"Best accuracy: {study.best_value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

    return study.best_params


def get_tuned_model(X_train, y_train, best_params):
    """
    Create and train XGBoost model with tuned hyperparameters.

    Args:
        X_train: Training features
        y_train: Training target
        best_params: Best hyperparameters from tuning

    Returns:
        Trained XGBClassifier model
    """
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model
