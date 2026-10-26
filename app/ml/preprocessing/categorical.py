
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_categorical_pipeline(
    nominal_columns,
    analysis,
    ordinal_categories=None,
    approved_high_cardinality=None,
    max_onehot_categories=100,
):
    ordinal_categories = ordinal_categories or {}
    approved_high_cardinality = set(
        approved_high_cardinality or []
    )

    transformers = []

    def validate_analysis(column):
        if column not in analysis:
            raise ValueError(
                f"No analysis result found for categorical "
                f"column '{column}'."
            )

        required_keys = {
            "imputation_strategy",
            "encoding_strategy",
        }

        missing_keys = required_keys - analysis[column].keys()

        if missing_keys:
            raise ValueError(
                f"Analysis for '{column}' is missing required "
                f"fields: {sorted(missing_keys)}"
            )

    def build_imputer(column):
        strategy = analysis[column]["imputation_strategy"]

        if strategy == "none":
            return None

        if strategy == "most_frequent":
            return SimpleImputer(strategy="most_frequent")

        if strategy == "missing_category":
            return SimpleImputer(
                strategy="constant",
                fill_value="__MLPILOT_MISSING__",
            )

        raise ValueError(
            f"Column '{column}' needs manual review "
            f"or has an unsupported imputation strategy: "
            f"'{strategy}'."
        )

    def build_feature_pipeline(column, encoder):
        steps = []

        imputer = build_imputer(column)

        if imputer is not None:
            steps.append(("imputer", imputer))

        steps.append(("encoder", encoder))

        return Pipeline(steps=steps)

    # Validate nominal feature configuration.
    for column in nominal_columns:
        validate_analysis(column)

        strategy = analysis[column]["encoding_strategy"]

        if strategy == "high_cardinality_review":
            if column not in approved_high_cardinality:
                raise ValueError(
                    f"Column '{column}' requires "
                    f"high-cardinality approval."
                )

        elif strategy != "onehot":
            raise ValueError(
                f"Column '{column}' is not approved "
                f"for one-hot encoding."
            )

        if max_onehot_categories < 1:
            raise ValueError(
                "max_onehot_categories must be at least 1."
            )

        transformers.append(
            (
                f"nominal_{column}",
                build_feature_pipeline(
                    column,
                    OneHotEncoder(
                        handle_unknown="ignore",
                        max_categories=max_onehot_categories,
                    ),
                ),
                [column],
            )
        )

    # Validate and build ordinal feature pipelines.
    for column, categories in ordinal_categories.items():
        validate_analysis(column)

        if analysis[column]["encoding_strategy"] != "ordinal":
            raise ValueError(
                f"Column '{column}' has not been approved "
                f"as ordinal."
            )

        if not categories:
            raise ValueError(
                f"Category order for '{column}' cannot be empty."
            )

        if len(categories) != len(set(categories)):
            raise ValueError(
                f"Category order for '{column}' contains duplicates."
            )

        transformers.append(
            (
                f"ordinal_{column}",
                build_feature_pipeline(
                    column,
                    OrdinalEncoder(
                        categories=[categories],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                [column],
            )
        )

    if not transformers:
        return "passthrough"

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
