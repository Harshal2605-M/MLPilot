
from sklearn.pipeline import Pipeline

from app.ml.preprocessing.pipeline import (
    build_preprocessing_pipeline,
)


def build_model_pipeline(
    numerical_columns,
    nominal_columns,
    numerical_analysis,
    categorical_analysis,
    model,
    ordinal_categories=None,
    approved_high_cardinality=None,
    max_onehot_categories=100,
):
    """
    Build a complete preprocessing and model pipeline.

    This function constructs the pipeline but does not train it.
    """

    if model is None:
        raise ValueError("A machine learning model must be provided.")

    preprocessing_pipeline = build_preprocessing_pipeline(
        numerical_columns=numerical_columns,
        nominal_columns=nominal_columns,
        numerical_analysis=numerical_analysis,
        categorical_analysis=categorical_analysis,
        ordinal_categories=ordinal_categories,
        approved_high_cardinality=approved_high_cardinality,
        max_onehot_categories=max_onehot_categories,
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing_pipeline),
            ("model", model),
        ]
    )
