
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression, LinearRegression

from app.ml.engine import run_baseline_experiment


def test_engine_trains_classification_model():
    df = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
        "City": [
            "Pune", "Mumbai", "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai", "Pune", "Mumbai",
        ],
        "Purchased": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    result = run_baseline_experiment(
        df=df,
        target_column="Purchased",
    )

    assert result["task_type"] == "classification"
    assert isinstance(
        result["model"].named_steps["model"],
        LogisticRegression,
    )
    assert len(result["X_test"]) == 2
    assert len(result["y_test"]) == 2


def test_engine_trains_regression_model():
    df = pd.DataFrame({
        "Area": [
            500, 600, 700, 800, 900,
            1000, 1100, 1200, 1300, 1400,
        ],
        "City": [
            "Pune", "Mumbai", "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai", "Pune", "Mumbai",
        ],
        "Price": [
            20.5, 25.5, 30.5, 35.5, 40.5,
            45.5, 50.5, 55.5, 60.5, 65.5,
        ],
    })

    result = run_baseline_experiment(
        df=df,
        target_column="Price",
    )

    assert result["task_type"] == "regression"
    assert isinstance(
        result["model"].named_steps["model"],
        LinearRegression,
    )
    assert len(result["X_test"]) == 2
    assert len(result["y_test"]) == 2


def test_engine_requires_confirmation_for_uncertain_task():
    df = pd.DataFrame({
        "Feature": list(range(10)),
        "Target": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    })

    with pytest.raises(
        ValueError,
        match="Task type is uncertain",
    ):
        run_baseline_experiment(
            df=df,
            target_column="Target",
        )


def test_engine_accepts_explicit_uncertain_task():
    df = pd.DataFrame({
        "Feature": list(range(10)),
        "Target": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    })

    result = run_baseline_experiment(
        df=df,
        target_column="Target",
        selected_task_type="classification",
    )

    assert result["task_type"] == "classification"
    assert isinstance(
        result["model"].named_steps["model"],
        LogisticRegression,
    )


def test_engine_rejects_conflicting_task_choice():
    df = pd.DataFrame({
        "Feature": list(range(10)),
        "Target": [
            10.5, 20.5, 30.5, 40.5, 50.5,
            60.5, 70.5, 80.5, 90.5, 100.5,
        ],
    })

    with pytest.raises(
        ValueError,
        match="Detected task type is 'regression'",
    ):
        run_baseline_experiment(
            df=df,
            target_column="Target",
            selected_task_type="classification",
        )
