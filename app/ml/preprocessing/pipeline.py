
from sklearn.compose import ColumnTransformer

from app.ml.preprocessing.numerical import (
    build_numerical_pipeline,
)
from app.ml.preprocessing.categorical import (
    build_categorical_pipeline,
)


def build_preprocessing_pipeline(
    numerical_columns,
    nominal_columns,
    numerical_analysis,
    categorical_analysis,
    ordinal_categories=None,
    approved_high_cardinality=None,
    max_onehot_categories=100,
):
    transformers = []

    # Build numerical transformer specifications.
    numerical_transformers = build_numerical_pipeline(
        numerical_columns=numerical_columns,
        analysis=numerical_analysis,
    )

    transformers.extend(numerical_transformers)

    # Build categorical transformer specifications.
    categorical_transformers = build_categorical_pipeline(
        nominal_columns=nominal_columns,
        analysis=categorical_analysis,
        ordinal_categories=ordinal_categories,
        approved_high_cardinality=approved_high_cardinality,
        max_onehot_categories=max_onehot_categories,
    )

    transformers.extend(categorical_transformers)

    # No transformations are required.
    if not transformers:
        return "passthrough"

    # One ColumnTransformer combines every feature transformer.
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
