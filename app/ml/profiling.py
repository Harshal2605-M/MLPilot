import pandas as pd

def profile_dataset(df : pd.DataFrame, target_column : str) -> dict :

    if df.empty :
        raise ValueError("DataFrame is empty")
    
    if not target_column.strip():
        raise ValueError("Target column is empty")

    if target_column not in df.columns :
        raise ValueError("Target column doesn't exists")

    rows, columns = df.shape

    missing_values = {
    column: int(count)
    for column, count in df.isnull().sum().items()
    }
    missing_target_values = int(df[target_column].isnull().sum())

    duplicate_rows = int(df.duplicated().sum())

    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["string", "bool", "object", "category"]).columns.tolist()

    target_distribution = df[target_column].value_counts().to_dict()

    profile = {
        "rows" : rows,
        "columns" : columns,
        "numerical_columns" : numerical_columns,
        "categorical_columns" : categorical_columns,
        "duplicate_rows" : duplicate_rows,
        "missing_values" : missing_values,
        "target_distribution" : target_distribution,
        "missing_target_values" : missing_target_values
    }

    return profile