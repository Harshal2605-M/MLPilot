
# MLPilot – ML Engine API Documentation

## 1. Overview

The ML Engine is responsible for preparing datasets, training machine learning models, evaluating their performance, generating dataset intelligence, and ranking trained models.

It uses Python and Scikit-learn for machine learning operations.

The engine does not make LLM-based decisions. It executes the experiment configuration supplied by the caller.

## 2. Main Entry Point

File: `app/ml/engine.py`

Function:

`run_baseline_experiment()`

### Function Signature

```python
run_baseline_experiment(
    df,
    target_column,
    selected_task_type=None,
    test_size=0.2,
    random_state=42,
    ordinal_categories=None,
    approved_high_cardinality=None,
    max_onehot_categories=100,
    use_case=None,
    primary_metric=None,
    secondary_metric=None,
)
```

## 3. Input Parameters

| Parameter | Description | Default |
|---|---|---|
| df | Input Pandas DataFrame | Required |
| target_column | Name of prediction target | Required |
| selected_task_type | Classification or regression selection | None |
| test_size | Proportion of data for testing | 0.2 |
| random_state | Random seed for splitting | 42 |
| ordinal_categories | User-approved ordinal categories | None |
| approved_high_cardinality | Approved high-cardinality features | None |
| max_onehot_categories | Maximum categories for one-hot encoding | 100 |
| use_case | Business or prediction use case | None |
| primary_metric | Primary evaluation metric | None |
| secondary_metric | Secondary evaluation metric | None |

## 4. Processing Workflow

The ML Engine follows these steps:

1. Detect the machine learning task.
2. Validate the selected task type.
3. Configure evaluation metrics.
4. Separate features and target.
5. Exclude entirely missing feature columns.
6. Split the dataset into training and testing sets.
7. Analyze numerical features.
8. Analyze categorical features.
9. Generate dataset intelligence.
10. Train registered classification or regression models.
11. Evaluate each successfully trained model.
12. Calculate experiment status.
13. Rank successfully evaluated models.
14. Return experiment results.

## 5. Model Training

The engine supports two task types.

### Classification

Uses:

`train_classification_models()`

Registered models include:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Classifier

### Regression

Uses:

`train_regression_models()`

Registered models include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regressor

Individual model failures are handled independently.

A failed model does not stop the remaining models from training.

## 6. Model Evaluation

File: `app/ml/evaluation.py`

### Classification Metrics

The current evaluation implementation calculates:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Confusion Matrix

### Regression Metrics

The current evaluation implementation calculates:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

## 7. Dataset Intelligence

File: `app/ml/dataset_intelligence.py`

The dataset intelligence module combines dataset profiling, preparation analysis, and additional feature insights.

It provides:

- Dataset overview
- Numerical feature intelligence
- Categorical feature intelligence
- Target analysis
- Feature relationships
- Data quality warnings
- Excluded feature information

Feature statistics and preparation analysis use training data to avoid preprocessing leakage.

## 8. Model Ranking

File: `app/ml/ranking.py`

Function:

`rank_models(model_results, metric_configuration, top_n=3)`

### Ranking Responsibilities

1. Validate metric configuration.
2. Ignore unsuccessful models.
3. Exclude models with missing or invalid primary metric values.
4. Sort using the primary metric.
5. Use the secondary metric for tie-breaking.
6. Use alphabetical model-name ordering if both metrics tie.
7. Return the top N models.

### Metric Direction

Higher is better:

- Accuracy
- Precision
- Recall
- F1-score
- R²

Lower is better:

- MAE
- MSE
- RMSE

Ranking is deterministic and does not depend on LLM-generated decisions.

## 9. Return Structure

The engine returns a dictionary containing:

| Key | Description |
|---|---|
| status | Overall experiment status |
| task_type | Classification or regression |
| target_column | Selected target |
| use_case | Experiment use case |
| metric_configuration | Primary and secondary metrics |
| model | First successful model, retained for backward compatibility |
| model_results | Individual model training and evaluation results |
| ranking | Deterministic ranking output |
| successful_models | Number of successful models |
| failed_models | Number of failed models |
| total_models | Total registered models evaluated |
| X_test | Test features |
| y_test | Actual test targets |
| numerical_analysis | Numerical preparation analysis |
| categorical_analysis | Categorical preparation analysis |
| excluded_features | Excluded feature information |
| dataset_intelligence | Combined dataset insights |

### Individual Model Result

Each model result contains:

- status
- model
- y_pred
- evaluation_metrics
- error

### Ranking Result

The ranking result contains:

- ranking: Ranked model details
- top_models: Names of selected top models
- excluded_models: Models excluded from ranking and their reasons

## 10. Experiment Status

The engine returns one of three statuses:

| Status | Meaning |
|---|---|
| completed | All models trained and evaluated successfully |
| partially_completed | At least one model succeeded and at least one failed |
| failed | No models succeeded |

## 11. Error Handling

The engine handles errors at different levels.

- Invalid dataset or target: preparation and validation errors.
- Uncertain task type: explicit task selection required.
- Individual model training failure: recorded in model results.
- Individual model evaluation failure: recorded in model results.
- Invalid ranking configuration: raises ValueError.

## 12. Example Usage

```python
result = run_baseline_experiment(
    df=df,
    target_column="Price",
    selected_task_type="regression",
    use_case="House price prediction",
    primary_metric="mae",
    secondary_metric="rmse",
)

print(result["status"])
print(result["model_results"])
print(result["ranking"])
print(result["dataset_intelligence"])
```

## 13. Architecture Responsibility

| Component | Responsibility |
|---|---|
| Preparation | Dataset validation and preparation |
| Preprocessing | Numerical and categorical transformations |
| Model Registry | Available ML models |
| Classification | Classification model training |
| Regression | Regression model training |
| Evaluation | Model performance metrics |
| Dataset Intelligence | Dataset insights |
| Ranking | Deterministic model comparison |
| Engine | Coordinates the complete ML workflow |

The ML Engine is responsible for actual machine learning computation.

The LLM Planner and Advisor will interact with the engine through structured inputs and outputs in later MLPilot modules.
