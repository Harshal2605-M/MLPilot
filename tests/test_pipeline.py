
import pandas as pd

from app.ml.preprocessing.pipeline import (
    build_preprocessing_pipeline,
)


def test_combined_pipeline_processes_numerical_and_categorical_features():
    X_train = pd.DataFrame({
        "Age": [20.0, 30.0, 40.0, 50.0],
        "City": ["Pune", "Mumbai", "Pune", "Delhi"],
    })

    numerical_analysis = {
        "Age": {
            "imputation_strategy": "none",
            "scaling_strategy": "standard",
        }
    }

    categorical_analysis = {
        "City": {
            "imputation_strategy": "none",
            "encoding_strategy": "onehot",
        }
    }

    pipeline = build_preprocessing_pipeline(
        numerical_columns=["Age"],
        nominal_columns=["City"],
        numerical_analysis=numerical_analysis,
        categorical_analysis=categorical_analysis,
    )

    transformed = pipeline.fit_transform(X_train)

    # One numerical feature + three one-hot city columns.
    assert transformed.shape == (4, 4)
    assert not pd.isna(transformed).any()
