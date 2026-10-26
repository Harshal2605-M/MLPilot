import pandas as pd
from sklearn.model_selection import train_test_split


def detect_task_type(df, target_column):
    if df.empty:
        raise ValueError("DataFrame is empty")

    if not target_column or not target_column.strip():
        raise ValueError("Target column is empty")

    if target_column not in df.columns:
        raise ValueError("Target column doesn't exist")

    target = df[target_column]

    if target.isna().any():
        raise ValueError("Target column contains missing values")

    # Rule 1: Non-numeric targets suggest classification
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"

    # Rule 2: Numeric targets with two unique values
    unique_values = target.nunique()

    if unique_values == 2:
        return "classification"

    # Rule 3: Check whether numeric values contain fractional parts
    has_fractional_values = (target % 1 != 0).any()

    if has_fractional_values:
        return "regression"

    # Rule 4: Many distinct integer values suggest regression
    if unique_values >= 30:
        return "regression"

    # Rule 5: Not enough evidence to decide
    return "uncertain"

def validate_task_choice(selected_task_type):
    allowed_types = {"classification", "regression"}

    if selected_task_type not in allowed_types:
        raise ValueError(
            "Task type must be classification or regression"
        )

    return selected_task_type


def split_features_target(df, target_column) :
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def split_train_test(X, y, test_size=0.2, random_state=42) :
    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=random_state)
    return X_train , X_test, y_train, y_test

def analyze_numeric_features(X_train):
    numeric_data = X_train.select_dtypes(include = "number")

    analysis = {}

    for column in numeric_data.columns:
        values = numeric_data[column]
        non_missing = values.dropna()

        missing_percentage = values.isna().mean() * 100 

        if non_missing.empty:
            analysis[column] = {
                "missing_percentage": float(missing_percentage),
                "skewness": None,
                "outlier_percentage": None,
                "imputation_strategy": "manual_review",
                "scaling_strategy": "manual_review",
            }
            continue

        skewness = non_missing.skew()

        q1 = non_missing.quantile(0.25)
        q3 = non_missing.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0 :
            outlier_percentage = 0.0

        else :
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = (
                (non_missing < lower_bound) | (non_missing > upper_bound)
                )
            outlier_percentage = outliers.mean() * 100

            has_many_outliers = outlier_percentage > 5 
            is_highly_skewed = skewness > 1

            if has_many_outliers or is_highly_skewed:
                imputation_strategy = "median"
                scaling_strategy = "robust"
            else:
                imputation_strategy = "mean"
                scaling_strategy = "standard"

            analysis[column] = {
            "missing_percentage": float(missing_percentage),
            "skewness": float(skewness),
            "outlier_percentage": float(outlier_percentage),
            "imputation_strategy": imputation_strategy,
            "scaling_strategy": scaling_strategy,
        }

    return analysis

def analyze_numeric_features(X_train):
    numeric_data = X_train.select_dtypes(include="number")

    analysis = {}

    for column in numeric_data.columns:
        values = numeric_data[column]
        non_missing = values.dropna()

        missing_percentage = float(values.isna().mean() * 100)

        # All values are missing
        if non_missing.empty:
            analysis[column] = {
                "missing_percentage": missing_percentage,
                "skewness": None,
                "outlier_percentage": None,
                "imputation_strategy": "manual_review",
                "scaling_strategy": "manual_review",
            }
            continue

        # High missingness requires user review
        if missing_percentage > 30:
            analysis[column] = {
                "missing_percentage": missing_percentage,
                "skewness": None,
                "outlier_percentage": None,
                "imputation_strategy": "manual_review",
                "scaling_strategy": "manual_review",
            }
            continue

        skewness = non_missing.skew()

        # Handle undefined skewness, such as a constant column
        if pd.isna(skewness):
            skewness = 0.0

        q1 = non_missing.quantile(0.25)
        q3 = non_missing.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outlier_percentage = 0.0
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = (
                (non_missing < lower_bound)
                | (non_missing > upper_bound)
            )

            outlier_percentage = float(outliers.mean() * 100)

        has_many_outliers = outlier_percentage > 5
        is_highly_skewed = abs(skewness) > 1

        if has_many_outliers or is_highly_skewed:
            imputation_strategy = (
                "median" if missing_percentage > 0 else "none"
            )
            scaling_strategy = "robust"
        else:
            imputation_strategy = (
                "mean" if missing_percentage > 0 else "none"
            )
            scaling_strategy = "standard"

        analysis[column] = {
            "missing_percentage": missing_percentage,
            "skewness": float(skewness),
            "outlier_percentage": float(outlier_percentage),
            "imputation_strategy": imputation_strategy,
            "scaling_strategy": scaling_strategy,
        }

    return analysis


def validate_ordinal_categories(X_train, ordinal_categories):
    """Validate user-approved category orders against training data."""

    categorical_columns = set(
        X_train.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns
    )

    for column, ordered_categories in ordinal_categories.items():
        if column not in categorical_columns:
            raise ValueError(
                f"'{column}' is not a recognized categorical feature."
            )

        if not ordered_categories:
            raise ValueError(
                f"Category order for '{column}' cannot be empty."
            )

        if len(ordered_categories) != len(set(ordered_categories)):
            raise ValueError(
                f"Category order for '{column}' contains duplicates."
            )

        
        observed_categories = set(
            X_train[column].dropna().unique()
        )

        approved_categories = set(ordered_categories)

        missing_categories = observed_categories - approved_categories
        unexpected_categories = approved_categories - observed_categories

        if missing_categories:
            raise ValueError(
                f"'{column}' has categories missing from the approved order: "
                f"{missing_categories}"
            )

        if unexpected_categories:
            raise ValueError(
                f"Approved order for '{column}' contains categories "
                f"not found in training data: {unexpected_categories}"
            )


    return True


def analyze_categorical_features(X_train, ordinal_categories=None):
    ordinal_categories = ordinal_categories or {}

    validate_ordinal_categories(
        X_train,
        ordinal_categories,
    )
    
    categorical_data = X_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    )

    analysis = {}

    for column in categorical_data.columns :

        values = categorical_data[column]
        non_missing = values.dropna()

        missing_percentage = float(values.isna().mean() * 100)
        unique_categories = non_missing.nunique()

        if non_missing.empty :
            analysis[column] = {
                "missing_percentage": missing_percentage,
                "unique_categories": 0,
                "cardinality_ratio": 0.0,
                "most_frequent_percentage": None,
                "imputation_strategy": "manual_review",
                "encoding_strategy": "manual_review",
            }
            continue 

        cardinality_ratio = float(
            unique_categories / len(non_missing)
        )

        most_frequent_percentage = float(
            non_missing.value_counts(normalize=True).iloc[0] * 100
        )

            
        if column in ordinal_categories:
            encoding_strategy = "ordinal"
        elif unique_categories > 50:
            encoding_strategy = "high_cardinality_review"
        else:
            encoding_strategy = "onehot"


        if missing_percentage > 30:
            imputation_strategy = "manual_review"
        elif missing_percentage > 0:
            imputation_strategy = "most_frequent"
        else:
            imputation_strategy = "none"

        analysis[column] = {
            "missing_percentage": missing_percentage,
            "unique_categories": unique_categories,
            "cardinality_ratio": cardinality_ratio,
            "most_frequent_percentage": most_frequent_percentage,
            "imputation_strategy": imputation_strategy,
            "encoding_strategy": encoding_strategy,
        }

    return analysis