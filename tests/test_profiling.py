
import pandas as pd
import pytest

from app.ml.profiling import profile_dataset


def test_profile_dataset():
    df = pd.DataFrame({
        "Age": [25, 42, 31, 50, 25],
        "Salary": [30000, 45000, None, 60000, 30000],
        "Churn": ["Yes", "No", "Yes", "No", "Yes"],
    })

    profile = profile_dataset(df, "Churn")

    assert profile["rows"] == 5
    assert profile["columns"] == 3
    assert profile["duplicate_rows"] == 1

    assert profile["missing_values"] == {
        "Age": 0,
        "Salary": 1,
        "Churn": 0,
    }

    assert profile["target_distribution"] == {
        "Yes": 3,
        "No": 2,
    }

def test_profile_dataset_rejects_empty_dataframe():
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="DataFrame is empty"):
        profile_dataset(df, "Churn")


def test_profile_dataset_rejects_empty_target():
    df = pd.DataFrame({"Churn": ["Yes", "No"]})

    with pytest.raises(ValueError, match="Target column is empty"):
        profile_dataset(df, "   ")


def test_profile_dataset_rejects_missing_target():
    df = pd.DataFrame({"Age": [25, 30]})

    with pytest.raises(ValueError, match="Target column doesn't exist"):
        profile_dataset(df, "Churn")

def test_missing_target_values():
    df = pd.DataFrame({
            "Age": [25, 42, 31, 50, 25],
            "Salary": [30000, 45000, None, 60000, 30000],
            "Churn": ["Yes", "No",None, "No", "Yes"],
        })
    profile = profile_dataset(df, "Churn")
    assert profile["missing_target_values"] == 1
