from __future__ import annotations

import pandas as pd

from reguq.data import prepare_data_bundle


def test_prepare_data_bundle_from_train_test_paths(synthetic_paths):
    train_path, test_path = synthetic_paths
    bundle = prepare_data_bundle(
        data={"train_path": str(train_path), "test_path": str(test_path)},
        target_col="target",
    )

    assert list(bundle.X_train.columns) == ["f1", "f2", "f3"]
    assert bundle.y_train.name == "target"
    assert len(bundle.X_test) > 0


def test_prepare_data_bundle_from_single_dataframe_split(synthetic_dataframes):
    train_df, test_df = synthetic_dataframes
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    bundle = prepare_data_bundle(
        data=full_df,
        target_col="target",
        split_config={"test_size": 0.25, "shuffle": False, "random_state": 42},
    )

    assert len(bundle.X_train) + len(bundle.X_test) == len(full_df)
    assert bundle.feature_columns == ["f1", "f2", "f3"]
