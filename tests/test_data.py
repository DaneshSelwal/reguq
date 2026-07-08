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


def test_standard_scaler_zero_variance_guard():
    import numpy as np
    from reguq.preprocess import scale_features, scale_targets
    
    # Create constant features and target
    X_train = pd.DataFrame({"constant_feat": [5.0, 5.0, 5.0]})
    X_test = pd.DataFrame({"constant_feat": [5.0, 5.0, 5.0]})
    y_train = pd.Series([10.0, 10.0, 10.0], name="target")
    y_test = pd.Series([10.0, 10.0, 10.0], name="target")
    
    # Check feature scaling
    scaled_train, scaled_test, _, scaler = scale_features(X_train, X_test)
    assert not np.isnan(scaled_train.to_numpy()).any()
    assert not np.isnan(scaled_test.to_numpy()).any()
    # Means should be subtracted (5.0 - 5.0 = 0.0), scale should be 1.0 (so no NaN)
    assert np.allclose(scaled_train.to_numpy(), 0.0)
    
    # Check target scaling
    scaled_y_train, scaled_y_test, _, y_scaler = scale_targets(y_train, y_test)
    assert not np.isnan(scaled_y_train.to_numpy()).any()
    assert not np.isnan(scaled_y_test.to_numpy()).any()
    assert np.allclose(scaled_y_train.to_numpy(), 0.0)

