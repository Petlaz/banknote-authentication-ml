import pytest
import pandas as pd
from banknote_auth.features import load_clean_data, split_features_targets

def test_load_clean_data():
    df = load_clean_data()
    # Check type
    assert isinstance(df, pd.DataFrame)
    # Check columns
    expected_cols = {"Variance_Wavelet", "Skewness_Wavelet", "Curtosis_Wavelet", "Image_Entropy", "Class"}
    assert expected_cols.issubset(set(df.columns))
    # Check no missing values
    assert df.isnull().sum().sum() == 0
    # Check at least 1000 rows (should match your dataset size)
    assert len(df) > 1000

def test_split_features_targets():
    df = load_clean_data()
    X, y = split_features_targets(df)
    # X should be a DataFrame, y should be a Series or 1D array
    assert isinstance(X, pd.DataFrame)
    assert hasattr(y, "shape")
    # X and y should have the same number of rows
    assert len(X) == len(y)
    # y should only contain 0 and 1
    assert set(y.unique()).issubset({0, 1})

    # To check if your tests in test_data.py run successfully: pytest tests/test_data.py