import numpy as np
import pandas as pd
import pytest

from reguq.splits import kfold_indices, split_indices, train_calib_split


def test_time_ordered_calibration_is_last_block():
    tr, cal = split_indices(10, 0.3, time_ordered=True)
    assert tr.tolist() == list(range(7))
    assert cal.tolist() == [7, 8, 9]


def test_random_split_is_disjoint_and_seeded():
    tr, cal = split_indices(100, 0.25, random_state=0)
    assert len(cal) == 25 and len(tr) == 75
    assert set(tr).isdisjoint(cal)
    tr2, cal2 = split_indices(100, 0.25, random_state=0)
    assert np.array_equal(cal, cal2)


def test_train_calib_split_pandas_positional():
    X = pd.DataFrame({"a": range(10)}, index=range(100, 110))
    y = pd.Series(np.arange(10.0), index=X.index)
    out = train_calib_split(X, y, 3, time_ordered=True)
    assert out.X_calib["a"].tolist() == [7, 8, 9]
    assert out.y_calib.tolist() == [7.0, 8.0, 9.0]


def test_kfold_partition():
    folds = kfold_indices(23, 5, random_state=1)
    allidx = np.concatenate(folds)
    assert sorted(allidx.tolist()) == list(range(23))


def test_bad_sizes():
    with pytest.raises(ValueError):
        split_indices(10, 1.5)
    with pytest.raises(ValueError):
        split_indices(10, 10)
