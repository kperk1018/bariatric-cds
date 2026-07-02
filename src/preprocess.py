"""Centralized preprocessing: OHE encoding + inference alignment.

Both reproduce_models.py (train) and predict.py (inference) must use this
module so the feature matrix is identical at train and inference time.
Divergence here is the #1 silent bug in deployed ML systems.
"""
import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["Sex", "Race", "Surgery_Type"]


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """OHE-encode categoricals; return encoded df and stable column list.

    Uses drop_first=False so all category levels are explicit (required for SVR
    which is distance-based, and cleaner for interpretation).
    """
    sub = df[feature_cols].copy()
    cats_present = [c for c in CATEGORICAL_COLS if c in feature_cols]
    encoded = pd.get_dummies(sub, columns=cats_present, drop_first=False, dummy_na=False)
    # Ensure bool dummy columns are cast to int (sklearn models expect numeric)
    bool_cols = encoded.select_dtypes(include="bool").columns
    encoded[bool_cols] = encoded[bool_cols].astype(int)
    columns = list(encoded.columns)
    return encoded, columns


def encode_patient(patient: dict, feature_columns: list[str]) -> np.ndarray:
    """Convert a single-patient dict to a (1, n_features) array aligned to training columns.

    Applies the same OHE as build_feature_matrix, then reindexes to the persisted
    column list so any unseen category levels become zero rather than raising an error.
    """
    row = pd.DataFrame([patient])
    cats_present = [c for c in CATEGORICAL_COLS if c in row.columns]
    encoded = pd.get_dummies(row, columns=cats_present, drop_first=False, dummy_na=False)
    bool_cols = encoded.select_dtypes(include="bool").columns
    encoded[bool_cols] = encoded[bool_cols].astype(int)
    # Align to training column order; fill missing OHE columns with 0
    aligned = encoded.reindex(columns=feature_columns, fill_value=0)
    return aligned.values.astype(float)


def get_numeric_cols(feature_cols: list[str]) -> list[str]:
    """Return the non-categorical features (targets for median imputation)."""
    return [c for c in feature_cols if c not in CATEGORICAL_COLS]
