
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

    if numerical_columns:
        numerical_pipeline = build_numerical_pipeline(
            numerical_columns=numerical_columns,
            analysis=numerical_analysis,
        )

        if numerical_pipeline != "passthrough":
            transformers.append(
                (
                    "numerical",
                    numerical_pipeline,
                    numerical_columns,
                )
            )

    if nominal_columns or ordinal_categories:
        categorical_pipeline = build_categorical_pipeline(
            nominal_columns=nominal_columns,
            analysis=categorical_analysis,
            ordinal_categories=ordinal_categories,
            approved_high_cardinality=approved_high_cardinality,
            max_onehot_categories=max_onehot_categories,
        )

        if categorical_pipeline != "passthrough":
            transformers.append(
            (
                "categorical",
                categorical_pipeline,
                nominal_columns
                + list((ordinal_categories or {}).keys()),
            )
        )

    if not transformers:
        return "passthrough"

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
