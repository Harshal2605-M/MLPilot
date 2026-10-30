
from sklearn.linear_model import LogisticRegression

from app.ml.model_pipeline import build_model_pipeline
from app.ml.model_registry import get_classification_models


def train_baseline_classifier(
    X_train,
    y_train,
    numerical_columns,
    nominal_columns,
    numerical_analysis,
    categorical_analysis,
    ordinal_categories=None,
    approved_high_cardinality=None,
    max_onehot_categories=100,
):
    """
    Train a baseline Logistic Regression classifier.

    Returns the fitted preprocessing and classification pipeline.
    """

    if y_train.nunique() < 2:
        raise ValueError(
            "Classification requires at least two classes in the training target."
        )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    pipeline = build_model_pipeline(
        numerical_columns=numerical_columns,
        nominal_columns=nominal_columns,
        numerical_analysis=numerical_analysis,
        categorical_analysis=categorical_analysis,
        model=model,
        ordinal_categories=ordinal_categories,
        approved_high_cardinality=approved_high_cardinality,
        max_onehot_categories=max_onehot_categories,
    )

    pipeline.fit(X_train, y_train)

    return pipeline


def train_classification_models(
    X_train,
    y_train,
    numerical_columns,
    nominal_columns,
    numerical_analysis,
    categorical_analysis,
    ordinal_categories=None,
    approved_high_cardinality=None,
    max_onehot_categories=100,
):
    """
    Train all registered classification models independently.

    Returns successful models and individual model failures.
    """

    if y_train.nunique() < 2:
        raise ValueError(
            "Classification requires at least two classes in the training target."
        )

    models = get_classification_models()

    results = {}

    for model_name, estimator in models.items():

        try:
            pipeline = build_model_pipeline(
                numerical_columns=numerical_columns,
                nominal_columns=nominal_columns,
                numerical_analysis=numerical_analysis,
                categorical_analysis=categorical_analysis,
                model=estimator,
                ordinal_categories=ordinal_categories,
                approved_high_cardinality=approved_high_cardinality,
                max_onehot_categories=max_onehot_categories,
            )

            pipeline.fit(X_train, y_train)

            results[model_name] = {
                "status": "completed",
                "model": pipeline,
                "error": None,
            }

        except Exception as e:

            results[model_name] = {
                "status": "failed",
                "model": None,
                "error": str(e),
            }

    return results
