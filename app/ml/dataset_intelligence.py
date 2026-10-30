
import pandas as pd

from app.ml.profiling import profile_dataset


def get_dataset_overview(df, target_column):
    """Collect general dataset information."""

    profile = profile_dataset(df, target_column)

    return {
        "rows": profile["rows"],
        "columns": profile["columns"],
        "numerical_columns": profile["numerical_columns"],
        "categorical_columns": profile["categorical_columns"],
        "duplicate_rows": profile["duplicate_rows"],
        "missing_values": profile["missing_values"],
        "missing_target_values": profile["missing_target_values"],
    }


def analyze_target(y, task_type):
    """Analyze classification or regression target."""

    if y.empty:
        raise ValueError("Target column is empty.")

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    if task_type == "classification":

        class_counts = y.value_counts().to_dict()

        class_percentages = (
            y.value_counts(normalize=True) * 100
        ).to_dict()

        minority_count = min(class_counts.values())
        majority_count = max(class_counts.values())

        return {
            "number_of_classes": len(class_counts),
            "class_distribution": class_counts,
            "class_percentages": class_percentages,
            "minority_class_percentage": float(
                min(class_percentages.values())
            ),
            "imbalance_ratio": float(
                majority_count / minority_count
            ),
        }

    elif task_type == "regression":

        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                "Regression target must contain numeric values."
            )

        q1 = y.quantile(0.25)
        q3 = y.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_percentage = 0.0
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = (
                (y < lower_bound) | (y > upper_bound)
            )

            outlier_percentage = float(
                outliers.mean() * 100
            )

        return {
            "mean": float(y.mean()),
            "median": float(y.median()),
            "standard_deviation": float(y.std()),
            "skewness": (
                float(y.skew())
                if pd.notna(y.skew())
                else 0.0
            ),
            "outlier_percentage": outlier_percentage,
            "minimum": float(y.min()),
            "maximum": float(y.max()),
        }

    else:
        raise ValueError(
            "Task type must be classification or regression."
        )


def analyze_numerical_intelligence(X_train):
    """Calculate descriptive statistics for numerical features."""

    numerical_data = X_train.select_dtypes(include="number")

    analysis = {}

    for column in numerical_data.columns:

        values = numerical_data[column]
        non_missing = values.dropna()

        analysis[column] = {
            "mean": (
                float(non_missing.mean())
                if not non_missing.empty else None
            ),
            "median": (
                float(non_missing.median())
                if not non_missing.empty else None
            ),
            "standard_deviation": (
                float(non_missing.std())
                if not non_missing.empty else None
            ),
            "unique_values": int(non_missing.nunique()),
            "missing_percentage": float(
                values.isna().mean() * 100
            ),
        }

    return analysis


def analyze_categorical_intelligence(X_train):
    """Analyze category counts and rare categories."""

    categorical_data = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    )

    analysis = {}

    for column in categorical_data.columns:

        values = categorical_data[column].dropna()

        category_counts = values.value_counts()

        total_values = len(values)

        if total_values == 0:
            analysis[column] = {
                "category_counts": {},
                "rare_categories": [],
                "rare_category_percentage": None,
                "unique_categories": 0,
            }
            continue

        # Categories representing less than 1% of values.
        rare_categories = category_counts[
            category_counts / total_values < 0.01
        ]

        analysis[column] = {
            "category_counts": {
                str(category): int(count)
                for category, count in category_counts.items()
            },
            "rare_categories": [
                str(category)
                for category in rare_categories.index
            ],
            "rare_category_percentage": float(
                rare_categories.sum() / total_values * 100
            ),
            "unique_categories": int(
                values.nunique()
            ),
        }

    return analysis


def analyze_feature_relationships(X_train):
    """Identify highly correlated numerical feature pairs."""

    numerical_data = X_train.select_dtypes(include="number")

    if numerical_data.shape[1] < 2:
        return {
            "highly_correlated_pairs": []
        }

    correlation_matrix = numerical_data.corr()

    correlated_pairs = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            correlation = correlation_matrix.iloc[i, j]

            if (
                pd.notna(correlation)
                and abs(correlation) >= 0.90
            ):
                correlated_pairs.append({
                    "feature_1": columns[i],
                    "feature_2": columns[j],
                    "correlation": float(correlation),
                })

    return {
        "highly_correlated_pairs": correlated_pairs
    }


def build_data_quality_warnings(
    dataset_profile,
    numerical_analysis,
    categorical_analysis,
    excluded_features,
):
    """Generate descriptive data-quality warnings."""

    warnings = []

    if dataset_profile["duplicate_rows"] > 0:
        warnings.append({
            "type": "duplicate_rows",
            "count": dataset_profile["duplicate_rows"],
            "message": "Dataset contains duplicate rows."
        })

    for column, details in numerical_analysis.items():

        if details["unique_values"] <= 1:
            warnings.append({
                "type": "constant_feature",
                "feature": column,
                "message": "Numerical feature has no variation."
            })

        if details["missing_percentage"] > 30:
            warnings.append({
                "type": "high_missing_values",
                "feature": column,
                "percentage": details["missing_percentage"],
            })

    for column, details in categorical_analysis.items():

        if details["unique_categories"] <= 1:
            warnings.append({
                "type": "constant_feature",
                "feature": column,
                "message": "Categorical feature has no variation."
            })

        if (
            details["rare_category_percentage"] is not None
            and details["rare_category_percentage"] > 10
        ):
            warnings.append({
                "type": "rare_categories",
                "feature": column,
                "percentage": details["rare_category_percentage"],
            })

    for feature in excluded_features:
        warnings.append({
            "type": "excluded_feature",
            "feature": feature["column"],
            "reason": feature["reason"],
        })

    return warnings


def build_dataset_intelligence(
    df,
    target_column,
    task_type,
    X_train,
    y_train,
    numerical_analysis,
    categorical_analysis,
    excluded_features,
):
    """
    Combine profiling, preparation and additional analysis
    into one structured dataset intelligence object.
    """

    dataset_profile = get_dataset_overview(
        df,
        target_column,
    )

    target_analysis = analyze_target(
        y_train,
        task_type,
    )

    numerical_intelligence = analyze_numerical_intelligence(
        X_train
    )

    categorical_intelligence = analyze_categorical_intelligence(
        X_train
    )

    feature_relationships = analyze_feature_relationships(
        X_train
    )

    data_quality_warnings = build_data_quality_warnings(
        dataset_profile,
        numerical_intelligence,
        categorical_intelligence,
        excluded_features,
    )

    return {
        "dataset_overview": dataset_profile,
        "target_analysis": target_analysis,
        "numerical_features": {
            "descriptive_statistics": numerical_intelligence,
            "preparation_analysis": numerical_analysis,
        },
        "categorical_features": {
            "descriptive_statistics": categorical_intelligence,
            "preparation_analysis": categorical_analysis,
        },
        "feature_relationships": feature_relationships,
        "data_quality": {
            "warnings": data_quality_warnings,
            "excluded_features": excluded_features,
        },
    }
