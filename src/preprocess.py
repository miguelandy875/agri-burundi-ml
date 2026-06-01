"""Reusable preprocessing utilities for the Burundi agriculture ML project."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CATEGORICAL_COLS = ["province", "culture", "saison"]
BINARY_COLS = ["utilisation_engrais", "acces_irrigation"]
TARGET_COL = "bonne_recolte"
LEAKAGE_COLS = ["rendement_t_ha", "production_totale_t"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw agriculture CSV and return it unchanged."""
    return pd.read_csv(path)


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return count and percentage of missing values for columns with missing data."""
    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percentage": df.isna().mean() * 100,
        }
    )
    return missing[missing["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values using leakage-aware, reproducible imputations.

    Strategy:
    - ``bonne_recolte`` is the target, so rows with a missing target are dropped.
      Imputing labels would invent ground truth and bias model evaluation.
    - Continuous numerical columns are imputed with the median by
      ``(culture, province)`` because yields and climate conditions vary strongly
      by crop and location. If a group median is unavailable, the global median is
      used as a stable fallback.
    - Binary operational columns (fertilizer and irrigation) are imputed with the
      mode because their valid values are categories encoded as 0/1.

    The returned dataframe is guaranteed to contain no missing values.
    """
    cleaned = df.copy()

    if TARGET_COL in cleaned.columns:
        cleaned = cleaned.dropna(subset=[TARGET_COL]).copy()

    for col in BINARY_COLS:
        if col in cleaned.columns and cleaned[col].isna().any():
            mode = cleaned[col].mode(dropna=True)
            if mode.empty:
                raise ValueError(f"Cannot impute binary column {col}: no valid mode.")
            cleaned[col] = cleaned[col].fillna(mode.iloc[0])

    numeric_cols = cleaned.select_dtypes(include="number").columns.tolist()
    numeric_cols = [
        col for col in numeric_cols if col not in BINARY_COLS + [TARGET_COL]
    ]

    for col in numeric_cols:
        if cleaned[col].isna().any():
            group_medians = cleaned.groupby(["culture", "province"])[col].transform(
                "median"
            )
            cleaned[col] = cleaned[col].fillna(group_medians)
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    remaining_missing = cleaned.isna().sum()
    if remaining_missing.any():
        raise ValueError(
            "Missing values remain after imputation: "
            f"{remaining_missing[remaining_missing > 0].to_dict()}"
        )

    return cleaned


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode nominal categorical columns and return final feature names.

    ``drop_first=True`` prevents perfect multicollinearity for linear models while
    preserving interpretable binary indicators for tree-based models.
    """
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    feature_columns = [
        col for col in df_encoded.columns if col not in LEAKAGE_COLS + [TARGET_COL]
    ]
    return df_encoded, feature_columns


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a preprocessed dataframe into model features ``X`` and target ``y``."""
    # rendement_t_ha and production_totale_t are direct linear derivatives of the
    # target variable. Including them causes data leakage and would produce
    # artificially inflated accuracy scores.
    X = df.drop(columns=LEAKAGE_COLS + [TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def _continuous_numerical_columns(X: pd.DataFrame) -> list[str]:
    """Identify numerical columns to scale, excluding binary and OHE indicators."""
    numerical_cols = X.select_dtypes(include="number").columns.tolist()
    return [
        col
        for col in numerical_cols
        if col not in BINARY_COLS and X[col].dropna().nunique() > 2
    ]


def normalize_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale continuous numerical features using statistics from ``X_train`` only.

    Binary columns and one-hot encoded columns are intentionally left unchanged so
    their 0/1 meaning is preserved. Fitting only on training data avoids leaking
    test-set distribution information into preprocessing.
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    numerical_cols = _continuous_numerical_columns(X_train_scaled)

    scaler = StandardScaler()
    if numerical_cols:
        X_train_scaled[numerical_cols] = scaler.fit_transform(
            X_train_scaled[numerical_cols]
        )
        X_test_scaled[numerical_cols] = scaler.transform(X_test_scaled[numerical_cols])
    else:
        scaler.fit(pd.DataFrame(index=X_train_scaled.index))

    return X_train_scaled, X_test_scaled, scaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split and save it under ``artifacts/``.

    Stratification keeps the good/bad harvest proportion stable across the split,
    while ``random_state`` makes reported metrics reproducible.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = project_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    X_train.to_csv(artifacts_dir / "X_train.csv", index=False)
    X_test.to_csv(artifacts_dir / "X_test.csv", index=False)
    y_train.to_csv(artifacts_dir / "y_train.csv", index=False)
    y_test.to_csv(artifacts_dir / "y_test.csv", index=False)

    return X_train, X_test, y_train, y_test


def build_inference_input(
    raw_input_dict: dict[str, Any],
    feature_columns: list[str],
    scaler: StandardScaler,
    numerical_cols: list[str],
) -> pd.DataFrame:
    """Prepare one raw user input row with the exact training feature schema.

    The function applies the same categorical encoding pattern used during
    training, inserts any missing dummy columns as 0, drops unexpected columns,
    orders columns exactly like ``feature_columns``, and scales the requested
    continuous numerical columns with the fitted training scaler.
    """
    row = pd.DataFrame([raw_input_dict])
    encoded = pd.get_dummies(row, columns=CATEGORICAL_COLS, drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)

    for categorical_col in CATEGORICAL_COLS:
        value = raw_input_dict.get(categorical_col)
        dummy_col = f"{categorical_col}_{value}"
        if dummy_col in encoded.columns:
            encoded.loc[:, dummy_col] = 1

    cols_to_scale = [col for col in numerical_cols if col in encoded.columns]
    if cols_to_scale:
        encoded[cols_to_scale] = scaler.transform(encoded[cols_to_scale])

    return encoded
