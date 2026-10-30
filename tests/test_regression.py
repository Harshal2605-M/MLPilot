
import pandas as pd
import pytest
from unittest.mock import patch

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app.ml.regression import (
    train_baseline_regressor,
    train_regression_models,
)


def test_baseline_regressor_trains_successfully():
    X_train = pd.DataFrame({
        "Area": [500, 600, 700, 800, 900, 1000],
        "City": [
            "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai",
        ],
    })

    y_train = pd.Series([20, 25, 30, 35, 40, 45])

    pipeline = train_baseline_regressor(
        X_train=X_train,
        y_train=y_train,
        numerical_columns=["Area"],
        nominal_columns=["City"],
        numerical_analysis={
            "Area": {
                "imputation_strategy": "none",
                "scaling_strategy": "standard",
            }
        },
        categorical_analysis={
            "City": {
                "imputation_strategy": "none",
                "encoding_strategy": "onehot",
            }
        },
    )

    assert isinstance(pipeline, Pipeline)
    assert isinstance(
        pipeline.named_steps["model"],
        LinearRegression,
    )


def test_baseline_regressor_predicts_values():
    X_train = pd.DataFrame({
        "Area": [500, 600, 700, 800, 900, 1000],
        "City": [
            "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai",
        ],
    })

    y_train = pd.Series([20, 25, 30, 35, 40, 45])

    pipeline = train_baseline_regressor(
        X_train=X_train,
        y_train=y_train,
        numerical_columns=["Area"],
        nominal_columns=["City"],
        numerical_analysis={
            "Area": {
                "imputation_strategy": "none",
                "scaling_strategy": "standard",
            }
        },
        categorical_analysis={
            "City": {
                "imputation_strategy": "none",
                "encoding_strategy": "onehot",
            }
        },
    )

    predictions = pipeline.predict(X_train)

    assert len(predictions) == len(y_train)
    assert all(isinstance(value, float) for value in predictions)


def test_baseline_regressor_rejects_missing_target():
    X_train = pd.DataFrame({
        "Area": [500, 600, 700],
    })

    y_train = pd.Series([20.0, float("nan"), 30.0])

    with pytest.raises(
        ValueError,
        match="missing values",
    ):
        train_baseline_regressor(
            X_train=X_train,
            y_train=y_train,
            numerical_columns=["Area"],
            nominal_columns=[],
            numerical_analysis={
                "Area": {
                    "imputation_strategy": "none",
                    "scaling_strategy": "standard",
                }
            },
            categorical_analysis={},
        )


def test_baseline_regressor_rejects_constant_target():
    X_train = pd.DataFrame({
        "Area": [500, 600, 700],
    })

    y_train = pd.Series([25, 25, 25])

    with pytest.raises(
        ValueError,
        match="at least two distinct target values",
    ):
        train_baseline_regressor(
            X_train=X_train,
            y_train=y_train,
            numerical_columns=["Area"],
            nominal_columns=[],
            numerical_analysis={
                "Area": {
                    "imputation_strategy": "none",
                    "scaling_strategy": "standard",
                }
            },
            categorical_analysis={},
        )


# Multi-model regression tests

def test_train_regression_models_returns_five_models():
    X_train = pd.DataFrame({
        "Area": [500, 600, 700, 800, 900, 1000, 1100, 1200],
        "Age": [2, 3, 4, 5, 6, 7, 8, 9],
    })

    y_train = pd.Series([20, 25, 30, 35, 40, 45, 50, 55])

    results = train_regression_models(
        X_train=X_train,
        y_train=y_train,
        numerical_columns=["Area", "Age"],
        nominal_columns=[],
        numerical_analysis={
            "Area": {
                "imputation_strategy": "none",
                "scaling_strategy": "standard",
            },
            "Age": {
                "imputation_strategy": "none",
                "scaling_strategy": "standard",
            },
        },
        categorical_analysis={},
    )

    assert len(results) == 5


def test_regression_models_have_valid_status():
    X_train = pd.DataFrame({
        "Area": [500, 600, 700, 800, 900, 1000, 1100, 1200],
        "Age": [2, 3, 4, 5, 6, 7, 8, 9],
    })

    y_train = pd.Series([20, 25, 30, 35, 40, 45, 50, 55])

    results = train_regression_models(
        X_train=X_train,
        y_train=y_train,
        numerical_columns=["Area", "Age"],
        nominal_columns=[],
        numerical_analysis={
            "Area": {
                "imputation_strategy": "none",
                "scaling_strategy": "standard",
            },
            "Age": {
                "imputation_strategy": "none",
                "scaling_strategy": "standard",
            },
        },
        categorical_analysis={},
    )

    for result in results.values():

        assert result["status"] in ["completed", "failed"]

        if result["status"] == "completed":
            assert result["model"] is not None
            assert result["error"] is None
        else:
            assert result["model"] is None
            assert result["error"] is not None



def test_regression_continues_when_one_model_fails():

    X_train = pd.DataFrame({
        "Area": [500, 600, 700, 800, 900, 1000, 1100, 1200],
        "Age": [2, 3, 4, 5, 6, 7, 8, 9],
    })

    y_train = pd.Series([20, 25, 30, 35, 40, 45, 50, 55])

    with patch(
        "app.ml.regression.get_regression_models"
    ) as mock_registry:

        mock_registry.return_value = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "KNN": KNeighborsRegressor(),
            "SVR": SVR(),
        }

        with patch(
            "app.ml.regression.build_model_pipeline"
        ) as mock_pipeline:

            from sklearn.pipeline import Pipeline

            def pipeline_side_effect(**kwargs):

                if kwargs["model"] is mock_registry.return_value["Random Forest"]:
                    raise ValueError("Simulated training failure")

                return Pipeline([
                    ("model", kwargs["model"])
                ])

            mock_pipeline.side_effect = pipeline_side_effect

            results = train_regression_models(
                X_train=X_train,
                y_train=y_train,
                numerical_columns=["Area", "Age"],
                nominal_columns=[],
                numerical_analysis={
                    "Area": {
                        "imputation_strategy": "none",
                        "scaling_strategy": "standard",
                    },
                    "Age": {
                        "imputation_strategy": "none",
                        "scaling_strategy": "standard",
                    },
                },
                categorical_analysis={},
            )

    assert len(results) == 5

    assert results["Random Forest"]["status"] == "failed"

    assert sum(
        result["status"] == "completed"
        for result in results.values()
    ) == 4
