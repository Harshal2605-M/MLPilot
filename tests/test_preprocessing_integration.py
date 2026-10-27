
import pandas as pd

from app.ml.preparation import (
    split_features_target,
    split_train_test,
    analyze_numeric_features,
    analyze_categorical_features,
)

from app.ml.preprocessing.pipeline import (
    build_preprocessing_pipeline,
)


def test_preparation_and_preprocessing_work_together():
    df = pd.DataFrame({
        "Age": [20.0, 30.0, 40.0, 50.0, 60.0],
        "Salary": [20000.0, 30000.0, 40000.0, 50000.0, 60000.0],
        "City": ["Pune", "Mumbai", "Pune", "Delhi", "Mumbai"],
        "Purchased": ["Yes", "No", "Yes", "No", "Yes"],
    })

    # Step 1: Separate features and target.
    X, y = split_features_target(df, "Purchased")

    # Step 2: Split into training and test data.
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # Step 3: Analyze training features only.
    numerical_analysis = analyze_numeric_features(X_train)
    categorical_analysis = analyze_categorical_features(X_train)

    # Step 4: Get feature lists from analysis results.
    numerical_columns = list(numerical_analysis.keys())
    categorical_columns = list(categorical_analysis.keys())

    # Step 5: Build the combined preprocessing pipeline.
    pipeline = build_preprocessing_pipeline(
        numerical_columns=numerical_columns,
        nominal_columns=categorical_columns,
        numerical_analysis=numerical_analysis,
        categorical_analysis=categorical_analysis,
    )

    # Step 6: Fit on training data and transform it.
    X_train_processed = pipeline.fit_transform(X_train)

    # Step 7: Transform test data using the fitted pipeline.
    X_test_processed = pipeline.transform(X_test)

    # Step 8: Verify the results.
    assert X_train_processed.shape[0] == len(X_train)
    assert X_test_processed.shape[0] == len(X_test)

    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    assert not pd.isna(X_train_processed).any()
    assert not pd.isna(X_test_processed).any()
