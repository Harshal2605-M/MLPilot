
import pandas as pd
import pytest

from app.ml.preparation import detect_task_type, split_features_target, split_train_test
from app.ml.preparation import validate_task_choice
from app.ml.preparation import analyze_numeric_features, analyze_categorical_features


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

def test_split_features_target():
    df = pd.DataFrame({
        "Age": [22, 30, 25],
        "Salary": [25000, 45000, 35000],
        "Purchased": ["No", "Yes", "No"]
    })

    X, y = split_features_target(df, "Purchased")

    assert list(X.columns) == ["Age", "Salary"]
    assert y.name == "Purchased"
    assert len(X) == len(y) == 3

def test_split_train_test_sizes():
    X = pd.DataFrame({"Age": list(range(100))})
    y = pd.Series([value * 2 for value in range(100)])

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_split_train_test_preserves_alignment():
    X = pd.DataFrame({"feature": list(range(50))})
    y = pd.Series([value * 10 for value in range(50)])

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    assert (y_train == X_train["feature"] * 10).all()
    assert (y_test == X_test["feature"] * 10).all()


def test_split_train_test_is_reproducible():
    X = pd.DataFrame({"feature": list(range(50))})
    y = pd.Series(list(range(50)))

    first = split_train_test(X, y, random_state=42)
    second = split_train_test(X, y, random_state=42)

    for first_part, second_part in zip(first, second):
        assert first_part.equals(second_part)

def test_analyze_numeric_features():
    X_train = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40],
        "Salary": [20000, 25000, 30000, 35000, 500000]
    })

    result = analyze_numeric_features(X_train)

    assert "Age" in result
    assert "Salary" in result

    assert result["Age"]["imputation_strategy"] == "mean"
    assert result["Salary"]["imputation_strategy"] == "median"

def test_analyze_categorical_features():
    X_train = pd.DataFrame({
        "City": ["Pune", "Mumbai", "Pune", "Delhi"],
        "Department": ["IT", "HR", "IT", "IT"],
    })

    result = analyze_categorical_features(X_train)

    assert result["City"]["unique_categories"] == 3
    assert result["City"]["encoding_strategy"] == "onehot"
    assert result["Department"]["imputation_strategy"] == "most_frequent"


import pytest

def test_ordinal_order_missing_observed_category():
    X_train = pd.DataFrame({
        "Education_Level": [
            "Graduate",
            "Doctorate",
            "Undergraduate",
        ]
    })

    ordinal_categories = {
        "Education_Level": [
            "Undergraduate",
            "Graduate",
        ]
    }

    with pytest.raises(ValueError, match="missing from the approved order"):
        analyze_categorical_features(
            X_train,
            ordinal_categories=ordinal_categories,
        )
