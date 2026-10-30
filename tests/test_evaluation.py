
import pytest

from app.ml.evaluation import (
    evaluate_classification,
    evaluate_regression,
)


def test_classification_metrics():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    metrics = evaluate_classification(y_true, y_pred)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision_weighted"] >= 0
    assert metrics["recall_weighted"] >= 0
    assert metrics["f1_weighted"] >= 0
    assert metrics["confusion_matrix"] == [
        [2, 0],
        [1, 1],
    ]


def test_regression_metrics():
    y_true = [100, 200, 300]
    y_pred = [110, 190, 290]

    metrics = evaluate_regression(y_true, y_pred)

    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["mse"] == pytest.approx(100.0)
    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["r2"] == pytest.approx(0.985)



def test_classification_returns_macro_metrics():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]

    result = evaluate_classification(y_true, y_pred)

    assert "precision_macro" in result
    assert "recall_macro" in result
    assert "f1_macro" in result


def test_classification_returns_class_specific_metrics():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0]

    result = evaluate_classification(y_true, y_pred)

    assert "0" in result["class_metrics"]
    assert "1" in result["class_metrics"]

    assert "precision" in result["class_metrics"]["1"]
    assert "recall" in result["class_metrics"]["1"]
    assert "f1_score" in result["class_metrics"]["1"]
    assert "support" in result["class_metrics"]["1"]


def test_classification_returns_class_labels():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = evaluate_classification(y_true, y_pred)

    assert result["class_labels"] == [0, 1]
    assert len(result["confusion_matrix"]) == 2


def test_classification_preserves_existing_metrics():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    result = evaluate_classification(y_true, y_pred)

    assert "accuracy" in result
    assert "precision_weighted" in result
    assert "recall_weighted" in result
    assert "f1_weighted" in result
    assert "confusion_matrix" in result
