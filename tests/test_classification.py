
import pandas as pd
import pytest
from unittest.mock import patch

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from app.ml.classification import (
    train_classification_models,
    train_baseline_classifier,
)


def test_baseline_classifier_trains_successfully():
    X_train = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45],
        "City": [
            "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai",
        ],
    })

    y_train = pd.Series([0, 1, 0, 1, 0, 1])

    numerical_analysis = {
        "Age": {
            "imputation_strategy": "none",
            "scaling_strategy": "standard",
        }
    }

    categorical_analysis = {
        "City": {
            "imputation_strategy": "none",
            "encoding_strategy": "onehot",
        }
    }

    pipeline = train_baseline_classifier(
        X_train=X_train,
        y_train=y_train,
        numerical_columns=["Age"],
        nominal_columns=["City"],
        numerical_analysis=numerical_analysis,
        categorical_analysis=categorical_analysis,
    )

    assert isinstance(pipeline, Pipeline)
    assert isinstance(
        pipeline.named_steps["model"],
        LogisticRegression,
    )


def test_baseline_classifier_predicts_classes():
    X_train = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45],
        "City": [
            "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai",
        ],
    })

    y_train = pd.Series([0, 1, 0, 1, 0, 1])

    pipeline = train_baseline_classifier(
        X_train=X_train,
        y_train=y_train,
        numerical_columns=["Age"],
        nominal_columns=["City"],
        numerical_analysis={
            "Age": {
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
    assert set(predictions).issubset({0, 1})


def test_baseline_classifier_rejects_single_class():
    X_train = pd.DataFrame({
        "Age": [20, 30, 40],
    })

    y_train = pd.Series([1, 1, 1])

    with pytest.raises(
        ValueError,
        match="at least two classes",
    ):
        train_baseline_classifier(
            X_train=X_train,
            y_train=y_train,
            numerical_columns=["Age"],
            nominal_columns=[],
            numerical_analysis={
                "Age": {
                    "imputation_strategy": "none",
                    "scaling_strategy": "standard",
                }
            },
            categorical_analysis={},
        )


# Multi-model classification tests

def test_train_classification_models_returns_five_models():

    X_train = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45, 50, 55],
        "Salary": [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000],
    })

    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    results = train_classification_models(
        X_train=X_train,
        y_train=y_train,
        numerical_columns=["Age", "Salary"],
        nominal_columns=[],
        numerical_analysis={
            "Age": {
                "imputation_strategy": "none",
                "scaling_strategy": "standard",
            },
            "Salary": {
                "imputation_strategy": "none",
                "scaling_strategy": "standard",
            },
        },
        categorical_analysis={},
    )

    assert len(results) == 5


def test_classification_models_have_valid_status():

    X_train = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45, 50, 55],
        "Salary": [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000],
    })

    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    results = train_classification_models(
        X_train=X_train,
        y_train=y_train,
        numerical_columns=["Age", "Salary"],
        nominal_columns=[],
        numerical_analysis={
            "Age": {
                "imputation_strategy": "none",
                "scaling_strategy": "standard",
            },
            "Salary": {
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


def test_classification_continues_when_one_model_fails():

    X_train = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45, 50, 55],
        "Salary": [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000],
    })

    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])

    numerical_analysis = {
        "Age": {
            "imputation_strategy": "none",
            "scaling_strategy": "standard",
        },
        "Salary": {
            "imputation_strategy": "none",
            "scaling_strategy": "standard",
        },
    }

    with patch(
        "app.ml.classification.get_classification_models"
    ) as mock_registry:

        mock_registry.return_value = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier(),
            "SVM": SVC(),
        }

        with patch(
            "app.ml.classification.build_model_pipeline"
        ) as mock_pipeline:

            from sklearn.pipeline import Pipeline

            def pipeline_side_effect(**kwargs):

                if kwargs["model"] is mock_registry.return_value["Random Forest"]:
                    raise ValueError("Simulated training failure")

                return Pipeline([
                    ("model", kwargs["model"])
                ])

            mock_pipeline.side_effect = pipeline_side_effect

            results = train_classification_models(
                X_train=X_train,
                y_train=y_train,
                numerical_columns=["Age", "Salary"],
                nominal_columns=[],
                numerical_analysis=numerical_analysis,
                categorical_analysis={},
            )

    assert len(results) == 5

    assert results["Random Forest"]["status"] == "failed"

    assert sum(
        result["status"] == "completed"
        for result in results.values()
    ) == 4
