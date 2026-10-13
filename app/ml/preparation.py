import pandas as pd


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