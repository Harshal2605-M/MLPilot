
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler


def build_numerical_pipeline(numerical_columns, analysis):
    transformers = []

    for column in numerical_columns:
        if column not in analysis:
            raise ValueError(
                f"No analysis result found for numerical column '{column}'."
            )

        column_analysis = analysis[column]

        imputation_strategy = column_analysis["imputation_strategy"]
        scaling_strategy = column_analysis["scaling_strategy"]

        if (
            imputation_strategy == "manual_review"
            or scaling_strategy == "manual_review"
        ):
            raise ValueError(
                f"Column '{column}' requires manual review "
                "before numerical preprocessing."
            )

        steps = []

        if imputation_strategy == "mean":
            steps.append(
                ("imputer", SimpleImputer(strategy="mean"))
            )
        elif imputation_strategy == "median":
            steps.append(
                ("imputer", SimpleImputer(strategy="median"))
            )
        elif imputation_strategy != "none":
            raise ValueError(
                f"Unsupported imputation strategy for '{column}': "
                f"{imputation_strategy}"
            )

        if scaling_strategy == "standard":
            steps.append(("scaler", StandardScaler()))
        elif scaling_strategy == "robust":
            steps.append(("scaler", RobustScaler()))
        else:
            raise ValueError(
                f"Unsupported scaling strategy for '{column}': "
                f"{scaling_strategy}"
            )

        transformers.append(
            (
                f"numeric_{column}",
                Pipeline(steps=steps),
                [column],
            )
        )

    return transformers
