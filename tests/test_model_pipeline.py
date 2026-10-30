
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app.ml.model_pipeline import build_model_pipeline


def test_build_model_pipeline_returns_complete_pipeline():
    numerical_analysis = {
        "Age": {
            "imputation_strategy": "none",
            "scaling_strategy": "standard",
        }
    }

    categorical_analysis = {
        "City": {
            "imputation_strategy": "most_frequent",
            "encoding_strategy": "onehot",
        }
    }

    model = LogisticRegression()

    pipeline = build_model_pipeline(
        numerical_columns=["Age"],
        nominal_columns=["City"],
        numerical_analysis=numerical_analysis,
        categorical_analysis=categorical_analysis,
        model=model,
    )

    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps["preprocessing"] is not None
    assert pipeline.named_steps["model"] is model


def test_model_pipeline_can_fit_and_predict():
    X_train = pd.DataFrame({
        "Age": [20, 25, 30, 35, 40, 45],
        "City": [
            "Pune", "Mumbai", "Pune",
            "Mumbai", "Pune", "Mumbai",
        ],
    })

    y_train = [0, 1, 0, 1, 0, 1]

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

    pipeline = build_model_pipeline(
        numerical_columns=["Age"],
        nominal_columns=["City"],
        numerical_analysis=numerical_analysis,
        categorical_analysis=categorical_analysis,
        model=LogisticRegression(),
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_train)

    assert len(predictions) == len(y_train)
    assert set(predictions).issubset({0, 1})


def test_model_pipeline_rejects_missing_model():
    with pytest.raises(
        ValueError,
        match="A machine learning model must be provided",
    ):
        build_model_pipeline(
            numerical_columns=[],
            nominal_columns=[],
            numerical_analysis={},
            categorical_analysis={},
            model=None,
        )
