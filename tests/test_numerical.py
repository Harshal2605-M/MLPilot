
import numpy as np
import pandas as pd
import pytest

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler

from app.ml.preprocessing.numerical import (
    build_numerical_pipeline,
)


def make_numerical_transformer(numerical_columns, analysis):
    """Wrap numerical transformer specifications for unit testing."""
    transformers = build_numerical_pipeline(
        numerical_columns=numerical_columns,
        analysis=analysis,
    )

    if not transformers:
        return "passthrough"

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def test_numerical_pipeline_scales_feature():
    """Standard scaling should produce a zero-mean feature."""
    X_train = pd.DataFrame({
        "Age": [20.0, 30.0, 40.0]
    })

    analysis = {
        "Age": {
            "imputation_strategy": "none",
            "scaling_strategy": "standard",
        }
    }

    pipeline = make_numerical_transformer(["Age"], analysis)
    transformed = pipeline.fit_transform(X_train)

    assert transformed.shape == (3, 1)
    assert abs(transformed.mean()) < 1e-9


def test_numerical_pipeline_uses_mean_imputation():
    """Missing values should be handled using mean imputation."""
    X_train = pd.DataFrame({
        "Age": [20.0, None, 40.0]
    })

    analysis = {
        "Age": {
            "imputation_strategy": "mean",
            "scaling_strategy": "standard",
        }
    }

    pipeline = make_numerical_transformer(["Age"], analysis)
    transformed = pipeline.fit_transform(X_train)

    assert transformed.shape == (3, 1)
    assert not np.isnan(transformed).any()


def test_numerical_pipeline_uses_median_imputation():
    """Missing values should be handled using median imputation."""
    X_train = pd.DataFrame({
        "Income": [10.0, None, 30.0, 100.0]
    })

    analysis = {
        "Income": {
            "imputation_strategy": "median",
            "scaling_strategy": "standard",
        }
    }

    pipeline = make_numerical_transformer(["Income"], analysis)
    transformed = pipeline.fit_transform(X_train)

    assert transformed.shape == (4, 1)
    assert not np.isnan(transformed).any()


def test_numerical_pipeline_uses_robust_scaling():
    """Robust scaling should use the median and interquartile range."""
    X_train = pd.DataFrame({
        "Income": [10.0, 20.0, 30.0, 1000.0]
    })

    analysis = {
        "Income": {
            "imputation_strategy": "none",
            "scaling_strategy": "robust",
        }
    }

    pipeline = make_numerical_transformer(["Income"], analysis)
    transformed = pipeline.fit_transform(X_train)

    assert transformed.shape == (4, 1)
    assert isinstance(
        pipeline.named_transformers_["numeric_Income"]
        .named_steps["scaler"],
        RobustScaler,
    )


def test_numerical_pipeline_uses_standard_scaler():
    """Standard scaling should use StandardScaler."""
    X_train = pd.DataFrame({
        "Age": [20.0, 30.0, 40.0]
    })

    analysis = {
        "Age": {
            "imputation_strategy": "none",
            "scaling_strategy": "standard",
        }
    }

    pipeline = make_numerical_transformer(["Age"], analysis)
    pipeline.fit_transform(X_train)

    assert isinstance(
        pipeline.named_transformers_["numeric_Age"]
        .named_steps["scaler"],
        StandardScaler,
    )


def test_manual_review_feature_is_rejected():
    """Features needing manual review must not be processed."""
    analysis = {
        "Income": {
            "imputation_strategy": "manual_review",
            "scaling_strategy": "standard",
        }
    }

    with pytest.raises(ValueError, match="requires manual review"):
        build_numerical_pipeline(["Income"], analysis)


def test_manual_review_scaling_is_rejected():
    """Manual-review scaling strategies must be rejected."""
    analysis = {
        "Age": {
            "imputation_strategy": "none",
            "scaling_strategy": "manual_review",
        }
    }

    with pytest.raises(ValueError, match="requires manual review"):
        build_numerical_pipeline(["Age"], analysis)


def test_unsupported_imputation_strategy_is_rejected():
    """Unknown imputation strategies must not be silently accepted."""
    analysis = {
        "Age": {
            "imputation_strategy": "unknown",
            "scaling_strategy": "standard",
        }
    }

    with pytest.raises(
        ValueError,
        match="Unsupported imputation strategy",
    ):
        build_numerical_pipeline(["Age"], analysis)


def test_unsupported_scaling_strategy_is_rejected():
    """Unknown scaling strategies must not be silently accepted."""
    analysis = {
        "Age": {
            "imputation_strategy": "none",
            "scaling_strategy": "unknown",
        }
    }

    with pytest.raises(
        ValueError,
        match="Unsupported scaling strategy",
    ):
        build_numerical_pipeline(["Age"], analysis)


def test_missing_analysis_for_feature_is_rejected():
    """Every numerical feature must have analysis metadata."""
    with pytest.raises(ValueError, match="No analysis result found"):
        build_numerical_pipeline(
            numerical_columns=["Age"],
            analysis={},
        )


def test_empty_numerical_columns_returns_passthrough():
    """No numerical features should require no numerical transformation."""
    pipeline = make_numerical_transformer(
        numerical_columns=[],
        analysis={},
    )

    assert pipeline == "passthrough"


def test_multiple_numerical_features_are_processed():
    """Each feature should receive its own configured pipeline."""
    X_train = pd.DataFrame({
        "Age": [20.0, 30.0, 40.0],
        "Income": [100.0, 200.0, 300.0],
    })

    analysis = {
        "Age": {
            "imputation_strategy": "none",
            "scaling_strategy": "standard",
        },
        "Income": {
            "imputation_strategy": "none",
            "scaling_strategy": "robust",
        },
    }

    pipeline = make_numerical_transformer(
        ["Age", "Income"],
        analysis,
    )

    transformed = pipeline.fit_transform(X_train)

    assert transformed.shape == (3, 2)
    assert not np.isnan(transformed).any()
