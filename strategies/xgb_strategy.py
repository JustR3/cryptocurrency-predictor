"""
XGBoost strategy for cryptocurrency trading.
Handles model training, prediction, and hyperparameter management.
"""

import json
import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import config


def load_hyperparameters(symbol: str):
    """
    Load hyperparameters from symbol-specific JSON file.
    Falls back to smart defaults based on asset class.

    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')

    Returns:
        Dictionary of hyperparameters
    """
    # Try symbol-specific file first
    safe_symbol = symbol.replace("/", "_")
    symbol_filepath = f"data/best_hyperparameters_{safe_symbol}.json"

    if os.path.exists(symbol_filepath):
        with open(symbol_filepath, "r") as f:
            params = json.load(f)
        print(f"✓ Loaded {symbol} hyperparameters from {symbol_filepath}")
        return params

    # Fall back to generic file
    generic_filepath = "data/best_hyperparameters.json"
    if os.path.exists(generic_filepath):
        with open(generic_filepath, "r") as f:
            params = json.load(f)
        print(f"⚠ Using generic hyperparameters from {generic_filepath}")
        print(f"  (Run 'tune.py --symbol {symbol} --save' for optimized parameters)")
        return params

    # Use smart defaults
    print(f"⚠ No hyperparameter files found, using smart defaults for {symbol}")
    print(f"  (Run 'tune.py --symbol {symbol} --trials 100 --save' to optimize)")
    return config.get_default_hyperparameters(symbol)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, symbol: str):
    """
    Train XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        symbol: Trading pair (for hyperparameter loading)

    Returns:
        Tuple of (model, label_encoder) or (None, None) if training failed
    """
    # Skip training if too few samples
    if len(y_train) < config.MIN_SAMPLES_FOR_TRAINING:
        return None, None

    # Use LabelEncoder to ensure classes are consecutive starting from 0
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    n_classes = len(label_encoder.classes_)

    # Skip if only one class
    if n_classes < 2:
        return None, None

    # Load hyperparameters
    hyperparams = load_hyperparameters(symbol)
    hyperparams["random_state"] = config.RANDOM_STATE

    # Classification setup
    hyperparams["objective"] = "multi:softprob"
    hyperparams["num_class"] = n_classes

    try:
        model = XGBClassifier(**hyperparams)
        model.fit(X_train, y_train_encoded)
        return model, label_encoder
    except ValueError as e:
        print(f"Warning: Training failed: {e}")
        return None, None


def predict(model, label_encoder, X: pd.DataFrame):
    """
    Make predictions with trained model.

    Args:
        model: Trained XGBoost model
        label_encoder: Label encoder used during training
        X: Features to predict on

    Returns:
        Tuple of (predictions, probabilities)
        - predictions: Decoded class labels
        - probabilities: Probability for each class
    """
    if model is None or label_encoder is None:
        return None, None

    # Get encoded predictions and probabilities
    predictions_encoded = model.predict(X)
    probs_encoded = model.predict_proba(X)

    # Decode predictions back to original labels
    predictions = label_encoder.inverse_transform(predictions_encoded)

    return predictions, probs_encoded


def get_feature_importance(model, feature_columns: list):
    """
    Get feature importance from trained model.

    Args:
        model: Trained XGBoost model
        feature_columns: List of feature names

    Returns:
        DataFrame with feature importance sorted
    """
    if model is None:
        return pd.DataFrame()

    importance_df = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return importance_df


def save_hyperparameters(params: dict, symbol: str):
    """
    Save hyperparameters to JSON file.

    Args:
        params: Hyperparameter dictionary
        symbol: Trading pair
    """
    safe_symbol = symbol.replace("/", "_")
    filename = f"data/best_hyperparameters_{safe_symbol}.json"

    with open(filename, "w") as f:
        json.dump(params, f, indent=2)

    print(f"✓ Saved hyperparameters to {filename}")
