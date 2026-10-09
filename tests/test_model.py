
import pandas as pd
import sys
import pytest
from sklearn.linear_model import LogisticRegression

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train 


def test_dataset_loads_with_expected_features():
    X, y = train.load_data()

    assert list(X.columns) == train.FEATURES
    assert len(X) == len(y)
    assert len(X) > 0


def test_dataset_has_valid_target_values():
    X, y = train.load_data()

    assert set(y.unique()).issubset({0, 1})
    assert not X.isnull().any().any()
    assert not y.isnull().any()


def test_missing_dataset_columns_are_rejected(tmp_path, monkeypatch):
    bad_file = tmp_path / "invalid_students.csv"

    pd.DataFrame({
        "hours_studied": [4, 6],
        "passed": [0, 1],
    }).to_csv(bad_file, index=False)

    monkeypatch.setattr(train, "DATA_PATH", bad_file)

    with pytest.raises(ValueError, match="missing columns"):
        train.load_data()


def test_data_split_sizes_and_disjoint_indices():
    X, y = train.load_data()

    X_train, X_val, X_test, y_train, y_val, y_test = (
        train.split_data(X, y)
    )

    total = len(X)

    assert len(X_train) == pytest.approx(total * 0.6, abs=1)
    assert len(X_val) == pytest.approx(total * 0.2, abs=1)
    assert len(X_test) == pytest.approx(total * 0.2, abs=1)

    # Splits must contain different rows.
    assert set(X_train.index).isdisjoint(X_val.index)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_val.index).isdisjoint(X_test.index)

    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)
    assert len(y_test) == len(X_test)


def test_model_trains_and_predicts_binary_classes():
    X, y = train.load_data()

    X_train, X_val, _, y_train, _, _ = train.split_data(X, y)

    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    assert len(predictions) == len(X_val)
    assert set(predictions).issubset({0, 1})
