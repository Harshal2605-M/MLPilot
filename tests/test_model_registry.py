
import pytest

from app.ml.model_registry import (
    get_models,
    get_classification_models,
    get_regression_models,
)


def test_classification_registry_returns_five_models():
    models = get_classification_models()

    assert len(models) == 5


def test_regression_registry_returns_five_models():
    models = get_regression_models()

    assert len(models) == 5


def test_classification_model_names_are_unique():
    models = get_classification_models()

    assert len(models) == len(set(models.keys()))


def test_regression_model_names_are_unique():
    models = get_regression_models()

    assert len(models) == len(set(models.keys()))


def test_get_models_returns_classification_models():
    models = get_models("classification")

    assert len(models) == 5
    assert "Logistic Regression" in models


def test_get_models_returns_regression_models():
    models = get_models("regression")

    assert len(models) == 5
    assert "Linear Regression" in models


def test_get_models_rejects_invalid_task_type():
    with pytest.raises(ValueError):
        get_models("clustering")


def test_registry_returns_fresh_instances():
    first_models = get_models("classification")
    second_models = get_models("classification")

    assert first_models is not second_models

    for name in first_models:
        assert first_models[name] is not second_models[name]
