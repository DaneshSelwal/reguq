"""Data loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import coerce_split_config
from .types import DataBundle, SplitConfig


def _read_dataframe(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    raise TypeError(f"Unsupported data source type: {type(source)}")


def _extract_train_test_from_mapping(data: Mapping[str, Any]) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    train = data.get("train") or data.get("train_df") or data.get("train_path")
    test = data.get("test") or data.get("test_df") or data.get("test_path")
    if train is None and test is None:
        return None, None
    if train is None or test is None:
        raise ValueError("Both train and test must be provided together.")
    return _read_dataframe(train), _read_dataframe(test)


def _extract_single_data(data: Any) -> pd.DataFrame:
    if isinstance(data, Mapping):
        single = data.get("data") or data.get("data_df") or data.get("data_path")
        if single is None:
            raise ValueError("Mapping data input must provide train/test or a single data source.")
        return _read_dataframe(single)
    return _read_dataframe(data)


def _split_single_dataset(dataset: pd.DataFrame, split_config: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        dataset,
        test_size=split_config.test_size,
        shuffle=split_config.shuffle,
        random_state=split_config.random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _validate_target(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in columns: {list(df.columns)}")


def _to_bundle(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> DataBundle:
    _validate_target(train_df, target_col)
    _validate_target(test_df, target_col)

    feature_columns = [c for c in train_df.columns if c != target_col]
    if not feature_columns:
        raise ValueError("No feature columns found. Dataset must contain at least one feature column.")

    missing_in_test = [c for c in feature_columns if c not in test_df.columns]
    if missing_in_test:
        raise ValueError(f"Test dataset is missing feature columns: {missing_in_test}")

    X_train = train_df[feature_columns].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[feature_columns].copy()
    y_test = test_df[target_col].copy()

    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_columns=feature_columns,
    )


def prepare_data_bundle(
    data: Any,
    target_col: str,
    split_config: Mapping[str, Any] | SplitConfig | None = None,
) -> DataBundle:
    split_cfg = coerce_split_config(split_config)

    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError("Tuple data input must be (train_df, test_df).")
        train_df = _read_dataframe(data[0])
        test_df = _read_dataframe(data[1])
        return _to_bundle(train_df, test_df, target_col)

    if isinstance(data, Mapping):
        train_df, test_df = _extract_train_test_from_mapping(data)
        if train_df is not None and test_df is not None:
            return _to_bundle(train_df, test_df, target_col)

    full_df = _extract_single_data(data)
    train_df, test_df = _split_single_dataset(full_df, split_cfg)
    return _to_bundle(train_df, test_df, target_col)
