"""Training and persistence utilities for the Burundi agriculture models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_decision_tree(
    X_train, y_train, max_depth: int = 4, criterion: str = "gini"
) -> DecisionTreeClassifier:
    """Train and return a decision tree classifier with reproducible settings."""
    model = DecisionTreeClassifier(
        max_depth=max_depth, criterion=criterion, random_state=42
    )
    return model.fit(X_train, y_train)


def train_random_forest(
    X_train, y_train, n_estimators: int = 100, random_state: int = 42
) -> RandomForestClassifier:
    """Train and return a random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    return model.fit(X_train, y_train)


def train_logistic_regression(
    X_train, y_train, max_iter: int = 1000, random_state: int = 42
) -> LogisticRegression:
    """Train and return a logistic regression classifier."""
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    return model.fit(X_train, y_train)


def save_model(model: Any, path: str | Path) -> None:
    """Persist a fitted model with joblib, creating the parent directory first."""
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(path: str | Path) -> Any:
    """Load and return a joblib-serialized model or preprocessing artifact."""
    return joblib.load(path)


def save_scaler(scaler: Any, path: str | Path = "models/scaler.pkl") -> None:
    """Save the fitted scaler required for consistent future inference."""
    save_model(scaler, path)


def save_feature_columns(
    feature_columns: list[str], path: str | Path = "models/feature_columns.pkl"
) -> None:
    """Save the training feature column order required for inference alignment."""
    save_model(feature_columns, path)


def train_and_save_all(X_train, y_train, scaler, feature_columns) -> dict[str, Any]:
    """Train the three required models and save all deployment artifacts."""
    models = {
        "decision_tree": train_decision_tree(X_train, y_train),
        "random_forest": train_random_forest(X_train, y_train),
        "logistic_regression": train_logistic_regression(X_train, y_train),
    }

    save_model(models["decision_tree"], "models/decision_tree.pkl")
    save_model(models["random_forest"], "models/random_forest.pkl")
    save_model(models["logistic_regression"], "models/logistic_regression.pkl")
    save_scaler(scaler)
    save_feature_columns(feature_columns)

    return models
