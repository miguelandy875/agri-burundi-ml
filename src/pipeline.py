"""Executable end-to-end ML pipeline for the Burundi agriculture project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import auc, roc_curve

from preprocess import (
    encode_categoricals,
    get_X_y,
    impute_missing,
    load_data,
    normalize_features,
    report_missing,
    split_data,
)
from train import train_and_save_all
from evaluate import get_metrics, save_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "agriculture_burundi.csv"
METRICS_PATH = PROJECT_ROOT / "metrics.json"


def _model_auc(model, X_test, y_test) -> float:
    """Calculate positive-class AUC for a fitted classifier."""
    scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, scores)
    return auc(fpr, tpr)


def run_pipeline() -> pd.DataFrame:
    """Run the full pipeline from raw data to saved model and metric artifacts."""
    print("1. Loading raw data...")
    raw_df = load_data(DATA_PATH)
    print(f"Raw shape: {raw_df.shape}")

    print("\n2. Missing values before imputation:")
    missing_report = report_missing(raw_df)
    if missing_report.empty:
        print("No missing values found.")
    else:
        print(missing_report.to_string())

    clean_df = impute_missing(raw_df)
    print(f"Missing values after imputation: {int(clean_df.isna().sum().sum())}")

    print("\n3. Encoding categorical variables...")
    encoded_df, feature_columns = encode_categoricals(clean_df)
    print(f"Encoded shape: {encoded_df.shape}")
    print(f"Feature count: {len(feature_columns)}")

    print("\n4. Creating X and y...")
    X, y = get_X_y(encoded_df)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\n5. Splitting train/test data...")
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    print("\n6. Normalizing features...")
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    print("\n7-8. Training and saving models, scaler, and feature columns...")
    models = train_and_save_all(X_train_scaled, y_train, scaler, feature_columns)

    print("\n9-10. Evaluating models and saving metrics...")
    display_names = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
    }
    metrics_for_json = {}
    summary_rows = []

    for model_key, model in models.items():
        metrics = get_metrics(model, X_test_scaled, y_test)
        model_auc = _model_auc(model, X_test_scaled, y_test)
        metrics_for_json[model_key] = {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "auc": model_auc,
        }
        summary_rows.append(
            {
                "Model": display_names[model_key],
                "Accuracy": metrics["accuracy"],
                "F1": metrics["f1"],
                "AUC": model_auc,
            }
        )

    save_metrics(metrics_for_json, METRICS_PATH)

    summary_table = pd.DataFrame(summary_rows)
    print("\n11. Summary table:")
    print(summary_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    return summary_table


if __name__ == "__main__":
    run_pipeline()
