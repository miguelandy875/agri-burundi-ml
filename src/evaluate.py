"""Evaluation helpers for model comparison and reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def get_metrics(model: Any, X_test, y_test) -> dict[str, Any]:
    """Return standard weighted classification metrics and the confusion matrix."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def plot_confusion_matrix(model: Any, X_test, y_test, title: str) -> None:
    """Plot a French-labeled confusion matrix heatmap for a fitted classifier."""
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Mauvaise récolte", "Bonne récolte"],
        yticklabels=["Mauvaise récolte", "Bonne récolte"],
    )
    plt.title(title)
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model: Any, feature_names: list[str], title: str, top_n: int = 15
) -> None:
    """Plot the top feature importances for trees or logistic regression.

    Tree-based models use ``feature_importances_``. Logistic regression uses the
    absolute value of coefficients as an interpretable importance proxy.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()


def _positive_scores(model: Any, X_test) -> np.ndarray:
    """Return continuous positive-class scores for ROC computation."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    raise ValueError("Model must expose predict_proba or decision_function for ROC.")


def plot_roc_curves(models_dict: dict[str, Any], X_test, y_test) -> dict[str, float]:
    """Plot ROC curves for several models and return AUC scores by model name."""
    auc_scores: dict[str, float] = {}
    plt.figure(figsize=(8, 6))

    for model_name, model in models_dict.items():
        scores = _positive_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        model_auc = auc(fpr, tpr)
        auc_scores[model_name] = model_auc
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard")
    plt.title("Courbes ROC des modèles")
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return auc_scores


def cross_validate_model(model: Any, X, y, cv: int = 5) -> dict[str, Any]:
    """Return mean, standard deviation, and all accuracy scores from CV."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
        "all_scores": scores,
    }


def overfitting_analysis(
    X_train,
    y_train,
    X_test,
    y_test,
    max_depths=range(1, 21),
) -> dict[str, list[float]]:
    """Train decision trees across depths and return train/test accuracies."""
    depths = list(max_depths)
    train_scores: list[float] = []
    test_scores: list[float] = []

    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))

    return {
        "depths": depths,
        "train_scores": train_scores,
        "test_scores": test_scores,
    }


def _json_safe(value: Any) -> Any:
    """Convert numpy/scikit-learn values to JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_metrics(metrics_dict: dict[str, Any], path: str | Path = "metrics.json") -> None:
    """Save compact model metrics to JSON at the requested path."""
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(metrics_dict), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
