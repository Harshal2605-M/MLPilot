from sklearn.linear_model import LinearRegression

from app.ml.model_pipeline import build_model_pipeline
from app.ml.model_registry import get_regression_models


def train_baseline_regressor(
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
    Train a baseline Linear Regression model.

    Returns the fitted preprocessing and regression pipeline.
    """

    if y_train.isna().any():
        raise ValueError(
            "The training target contains missing values."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "Regression requires at least two distinct target values."
        )

    model = LinearRegression()

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


def train_regression_models(
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
    Train all registered regression models independently.

    Returns successful models and individual model failures.
    """

    if y_train.isna().any():
        raise ValueError(
            "The training target contains missing values."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "Regression requires at least two distinct target values."
        )

    models = get_regression_models()

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
