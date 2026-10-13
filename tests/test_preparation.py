
import pandas as pd
import pytest

from app.ml.preparation import detect_task_type
from app.ml.preparation import validate_task_choice


def test_text_target_is_classification():
    df = pd.DataFrame({"target": ["Yes", "No", "Yes"]})
    assert detect_task_type(df, "target") == "classification"


def test_binary_numeric_target_is_classification():
    df = pd.DataFrame({"target": [0, 1, 0, 1]})
    assert detect_task_type(df, "target") == "classification"


def test_numeric_target_with_few_unique_values_is_uncertain():
    df = pd.DataFrame({"target": [0, 1, 2, 0, 1, 2]})
    assert detect_task_type(df, "target") == "uncertain"


def test_numeric_target_with_30_unique_values_is_regression():
    df = pd.DataFrame({"target": list(range(30))})
    assert detect_task_type(df, "target") == "regression"


def test_missing_target_values_raise_error():
    df = pd.DataFrame({"target": ["Yes", None, "No"]})

    with pytest.raises(ValueError, match="Target column contains missing values"):
        detect_task_type(df, "target")


def test_valid_classification_choice():
    assert validate_task_choice("classification") == "classification"


def test_valid_regression_choice():
    assert validate_task_choice("regression") == "regression"


def test_invalid_task_choice():
    with pytest.raises(ValueError):
        validate_task_choice("classify")


def test_numeric_target_with_fractional_values_is_regression():
    df = pd.DataFrame({"target": [0.6, 1.3, 2.7, 0.0, 1.1, 2.5]})
    assert detect_task_type(df, "target") == "regression"