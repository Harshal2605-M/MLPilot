import pytest
from app.ml.ranking import rank_models


def test_rank_regression_models():
    model_results = {
        "Linear Regression": {
            "status": "completed",
            "metrics": {"mae": 450, "rmse": 520, "r2": 0.89},
        },
        "Random Forest": {
            "status": "completed",
            "metrics": {"mae": 300, "rmse": 380, "r2": 0.94},
        },
        "Decision Tree": {
            "status": "completed",
            "metrics": {"mae": 600, "rmse": 710, "r2": 0.82},
        },
        "KNN": {
            "status": "failed",
            "error": "Training failed",
        },
    }

    metric_configuration = {
        "primary_metric": "mae",
        "secondary_metric": "rmse",
    }

    result = rank_models(model_results, metric_configuration)

    assert result["top_models"] == [
        "Random Forest",
        "Linear Regression",
        "Decision Tree",
    ]

    assert result["ranking"][0]["rank"] == 1
    assert result["ranking"][0]["primary_metric"] == 300

    assert "KNN" in result["excluded_models"]

def test_primary_metric_takes_priority():
    models = {
        "Model A": {
            "status": "completed",
            "metrics": {"mae": 100, "rmse": 150},
        },
        "Model B": {
            "status": "completed",
            "metrics": {"mae": 120, "rmse": 100},
        },
    }

    config = {
        "primary_metric": "mae",
        "secondary_metric": "rmse",
    }

    result = rank_models(models, config)

    assert result["top_models"][0] == "Model A"


def test_secondary_metric_breaks_tie():
    models = {
        "Model A": {
            "status": "completed",
            "metrics": {"mae": 100, "rmse": 150},
        },
        "Model B": {
            "status": "completed",
            "metrics": {"mae": 100, "rmse": 120},
        },
    }

    config = {
        "primary_metric": "mae",
        "secondary_metric": "rmse",
    }

    result = rank_models(models, config)

    assert result["top_models"][0] == "Model B"


def test_alphabetical_order_breaks_complete_tie():
    models = {
        "Zebra": {
            "status": "completed",
            "metrics": {"mae": 100, "rmse": 150},
        },
        "Apple": {
            "status": "completed",
            "metrics": {"mae": 100, "rmse": 150},
        },
    }

    config = {
        "primary_metric": "mae",
        "secondary_metric": "rmse",
    }

    result = rank_models(models, config)

    assert result["top_models"] == ["Apple", "Zebra"]


def test_missing_primary_metric_is_excluded():
    models = {
        "Model A": {
            "status": "completed",
            "metrics": {"rmse": 150},
        },
    }

    config = {
        "primary_metric": "mae",
        "secondary_metric": "rmse",
    }

    result = rank_models(models, config)

    assert result["top_models"] == []
    assert "Model A" in result["excluded_models"]


def test_invalid_metric_configuration():
    models = {}

    config = {
        "primary_metric": "unknown_metric",
        "secondary_metric": "rmse",
    }

    with pytest.raises(ValueError):
        rank_models(models, config)
