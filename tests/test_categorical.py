
import pandas as pd
import pytest

from app.ml.preprocessing.categorical import (
    build_categorical_pipeline,
)


def test_nominal_feature_uses_onehot_encoding():
    X_train = pd.DataFrame({
        "City": ["Pune", "Mumbai", "Pune", "Delhi"]
    })

    analysis = {
        "City": {
            "encoding_strategy": "onehot",
            "imputation_strategy": "most_frequent",
        }
    }

    pipeline = build_categorical_pipeline(
        nominal_columns=["City"],
        analysis=analysis,
    )

    transformed = pipeline.fit_transform(X_train)

    assert transformed.shape == (4, 3)


def test_ordinal_feature_uses_approved_order():
    X_train = pd.DataFrame({
        "Education_Level": [
            "Graduate",
            "Undergraduate",
            "Postgraduate",
        ]
    })

    ordinal_categories = {
        "Education_Level": [
            "Undergraduate",
            "Graduate",
            "Postgraduate",
        ]
    }

    analysis = {
        "Education_Level": {
            "encoding_strategy": "ordinal",
            "imputation_strategy": "most_frequent",
        }
    }

    pipeline = build_categorical_pipeline(
        nominal_columns=[],
        analysis=analysis,
        ordinal_categories=ordinal_categories,
    )

    transformed = pipeline.fit_transform(X_train)

    assert transformed.shape == (3, 1)
    assert transformed.ravel().tolist() == [1.0, 0.0, 2.0]


def test_missing_category_is_preserved_as_its_own_category():
    X_train = pd.DataFrame({
        "City": ["Pune", None, "Mumbai", "Pune"]
    })

    analysis = {
        "City": {
            "encoding_strategy": "onehot",
            "imputation_strategy": "missing_category",
        }
    }

    pipeline = build_categorical_pipeline(
        nominal_columns=["City"],
        analysis=analysis,
    )

    transformed = pipeline.fit_transform(X_train)

    # Pune, Mumbai, and the missing category
    assert transformed.shape == (4, 3)

    encoder = pipeline.named_transformers_["nominal_City"].named_steps[
        "encoder"
    ]

    assert "__MLPILOT_MISSING__" in encoder.categories_[0]

def test_unapproved_high_cardinality_feature_is_rejected():
    X_train = pd.DataFrame({
        "Customer_ID": [f"ID_{i}" for i in range(60)]
    })

    analysis = {
        "Customer_ID": {
            "encoding_strategy": "high_cardinality_review",
            "imputation_strategy": "most_frequent",
        }
    }

    with pytest.raises(ValueError, match="requires high-cardinality approval"):
        pipeline = build_categorical_pipeline(
            nominal_columns=["Customer_ID"],
            analysis=analysis,
        )
        pipeline.fit_transform(X_train)


def test_manual_review_feature_is_rejected():
    X_train = pd.DataFrame({
        "City": ["Pune", "Mumbai", "Delhi"]
    })

    analysis = {
        "City": {
            "encoding_strategy": "manual_review",
            "imputation_strategy": "manual_review",
        }
    }

    with pytest.raises(ValueError, match="not approved for one-hot encoding"):
        build_categorical_pipeline(
            nominal_columns=["City"],
            analysis=analysis,
        )

def test_unknown_nominal_category_does_not_raise_error():
    X_train = pd.DataFrame({
        "City": ["Pune", "Mumbai", "Delhi"]
    })

    X_new = pd.DataFrame({
        "City": ["Nagpur"]
    })

    analysis = {
        "City": {
            "encoding_strategy": "onehot",
            "imputation_strategy": "most_frequent",
        }
    }

    pipeline = build_categorical_pipeline(
        nominal_columns=["City"],
        analysis=analysis,
    )

    pipeline.fit(X_train)
    transformed = pipeline.transform(X_new)

    assert transformed.shape == (1, 3)

def test_unknown_ordinal_category_is_encoded_as_minus_one():
    X_train = pd.DataFrame({
        "Education_Level": [
            "Undergraduate",
            "Graduate",
            "Postgraduate",
        ]
    })

    X_new = pd.DataFrame({
        "Education_Level": ["Doctorate"]
    })

    ordinal_categories = {
        "Education_Level": [
            "Undergraduate",
            "Graduate",
            "Postgraduate",
        ]
    }

    analysis = {
        "Education_Level": {
            "encoding_strategy": "ordinal",
            "imputation_strategy": "most_frequent",
        }
    }

    pipeline = build_categorical_pipeline(
        nominal_columns=[],
        analysis=analysis,
        ordinal_categories=ordinal_categories,
    )

    pipeline.fit(X_train)
    transformed = pipeline.transform(X_new)

    assert transformed[0, 0] == -1


def test_missing_analysis_metadata_is_rejected():
    with pytest.raises(ValueError, match="No analysis result found"):
        build_categorical_pipeline(
            nominal_columns=["City"],
            analysis={},
        )


def test_none_imputation_strategy_is_supported():
    X_train = pd.DataFrame({
        "City": ["Pune", "Mumbai", "Pune"]
    })

    analysis = {
        "City": {
            "imputation_strategy": "none",
            "encoding_strategy": "onehot",
        }
    }

    pipeline = build_categorical_pipeline(
        nominal_columns=["City"],
        analysis=analysis,
    )

    transformed = pipeline.fit_transform(X_train)

    assert transformed.shape == (3, 2)


def test_empty_ordinal_categories_are_rejected():
    analysis = {
        "Level": {
            "imputation_strategy": "none",
            "encoding_strategy": "ordinal",
        }
    }

    with pytest.raises(ValueError, match="cannot be empty"):
        build_categorical_pipeline(
            nominal_columns=[],
            analysis=analysis,
            ordinal_categories={"Level": []},
        )


def test_duplicate_ordinal_categories_are_rejected():
    analysis = {
        "Level": {
            "imputation_strategy": "none",
            "encoding_strategy": "ordinal",
        }
    }

    with pytest.raises(ValueError, match="contains duplicates"):
        build_categorical_pipeline(
            nominal_columns=[],
            analysis=analysis,
            ordinal_categories={
                "Level": ["Low", "Medium", "Low"]
            },
        )


def test_high_cardinality_encoding_respects_category_limit():
    X_train = pd.DataFrame({
        "Product": [f"Product_{i}" for i in range(120)]
    })

    analysis = {
        "Product": {
            "imputation_strategy": "none",
            "encoding_strategy": "high_cardinality_review",
        }
    }

    pipeline = build_categorical_pipeline(
        nominal_columns=["Product"],
        analysis=analysis,
        approved_high_cardinality=["Product"],
        max_onehot_categories=10,
    )

    transformed = pipeline.fit_transform(X_train)

    assert transformed.shape[0] == 120
    assert transformed.shape[1] <= 10
