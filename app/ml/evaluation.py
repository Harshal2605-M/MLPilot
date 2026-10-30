
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_classification(y_true, y_pred):
    """Calculate comprehensive classification metrics."""

    labels = sorted(set(y_true) | set(y_pred))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    class_metrics = {}

    for label in labels:
        label_key = str(label)

        class_metrics[label_key] = {
            "precision": float(report[label_key]["precision"]),
            "recall": float(report[label_key]["recall"]),
            "f1_score": float(report[label_key]["f1-score"]),
            "support": int(report[label_key]["support"]),
        }

    return {
        # Overall metric
        "accuracy": float(accuracy_score(y_true, y_pred)),

        # Weighted metrics
        "precision_weighted": float(
            precision_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true, y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        # Macro metrics
        "precision_macro": float(
            precision_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true, y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        # Individual class metrics
        "class_metrics": class_metrics,

        # Confusion matrix and corresponding class order
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        ).tolist(),

        "class_labels": labels,
    }


def evaluate_regression(y_true, y_pred):
    """Calculate regression metrics on held-out test data."""

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }
