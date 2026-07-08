"""Preprocessing helpers used across phases."""

from __future__ import annotations

import pandas as pd


def coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to numeric when possible and keep original for non-convertible columns."""
    converted = frame.copy()
    for col in converted.columns:
        try:
            converted[col] = pd.to_numeric(converted[col])
        except Exception:
            pass
    return converted


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, X_val: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, StandardScaler]:
    """Scale features using StandardScaler fit on X_train."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    if scaler.scale_ is not None:
        scaler.scale_ = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    scaled_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    scaled_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    scaled_val = None
    if X_val is not None:
        scaled_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    return scaled_train, scaled_test, scaled_val, scaler


def scale_targets(
    y_train: pd.Series, y_test: pd.Series, y_val: pd.Series | None = None
) -> tuple[pd.Series, pd.Series, pd.Series | None, StandardScaler]:
    """Scale targets using StandardScaler fit on y_train."""
    scaler = StandardScaler()
    y_train_np = y_train.to_numpy().reshape(-1, 1) if hasattr(y_train, "to_numpy") else np.asarray(y_train).reshape(-1, 1)
    y_test_np = y_test.to_numpy().reshape(-1, 1) if hasattr(y_test, "to_numpy") else np.asarray(y_test).reshape(-1, 1)

    scaler.fit(y_train_np)
    if scaler.scale_ is not None:
        scaler.scale_ = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    scaled_train = pd.Series(scaler.transform(y_train_np).ravel(), name=y_train.name, index=y_train.index)
    scaled_test = pd.Series(scaler.transform(y_test_np).ravel(), name=y_test.name, index=y_test.index)
    scaled_val = None
    if y_val is not None:
        y_val_np = y_val.to_numpy().reshape(-1, 1) if hasattr(y_val, "to_numpy") else np.asarray(y_val).reshape(-1, 1)
        scaled_val = pd.Series(scaler.transform(y_val_np).ravel(), name=y_val.name, index=y_val.index)
    return scaled_train, scaled_test, scaled_val, scaler


def create_validation_split(
    X: pd.DataFrame, y: pd.Series, val_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into train and validation sets."""
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_size, random_state=random_state)
    return X_tr, X_val, y_tr, y_val
