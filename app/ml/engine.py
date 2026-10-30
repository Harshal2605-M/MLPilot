from app.ml.ranking import rank_models
from app.ml.preparation import (
    detect_task_type,
    validate_task_choice,
    prepare_dataset,
    split_train_test,
    analyze_numeric_features,
    analyze_categorical_features,
)

from app.ml.dataset_intelligence import build_dataset_intelligence
from app.ml.classification import train_classification_models
from app.ml.regression import train_regression_models

from app.ml.evaluation import (
    evaluate_classification,
    evaluate_regression,
)


def run_baseline_experiment(
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
):
    """
    Prepare a dataset, train multiple models, and evaluate them.

    The LLM Experiment Planner supplies the use case and
    recommended metrics. The ML engine performs training
    and evaluation without making metric-selection decisions.

    Individual model failures do not stop the experiment.
    """

    # 1. Detect the ML task.
    detected_task_type = detect_task_type(df, target_column)

    if detected_task_type == "uncertain":
        if selected_task_type is None:
            raise ValueError(
                "Task type is uncertain. Please select classification "
                "or regression explicitly."
            )

        task_type = validate_task_choice(selected_task_type)

    else:
        task_type = detected_task_type

        if selected_task_type is not None:
            selected_task_type = validate_task_choice(
                selected_task_type
            )

            if selected_task_type != detected_task_type:
                raise ValueError(
                    f"Detected task type is '{detected_task_type}', "
                    f"but '{selected_task_type}' was selected. "
                    "Review the task type before training."
                )

    # 2. Validate metric configuration.
    # 2. Store metric recommendations if provided.
    # Temporary defaults until the LLM Planner supplies metrics.
    if primary_metric is None:
        if task_type == "classification":
            primary_metric = "f1_weighted"
            secondary_metric = "accuracy"
        else:
            primary_metric = "mae"
            secondary_metric = "rmse"

    metric_configuration = {
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
    }
    # 3. Prepare features and target.
    X, y, excluded_features = prepare_dataset(
        df,
        target_column,
    )

    # 4. Split the dataset.
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # 5. Analyze training features only.
    numerical_analysis = analyze_numeric_features(X_train)

    categorical_analysis = analyze_categorical_features(
        X_train,
        ordinal_categories=ordinal_categories,
    )

    # 6. Build dataset intelligence.
    dataset_intelligence = build_dataset_intelligence(
        df=df,
        target_column=target_column,
        task_type=task_type,
        X_train=X_train,
        y_train=y_train,
        numerical_analysis=numerical_analysis,
        categorical_analysis=categorical_analysis,
        excluded_features=excluded_features,
    )
    # 6. Extract feature names.
    numerical_columns = list(numerical_analysis.keys())
    nominal_columns = list(categorical_analysis.keys())

    # 7. Prepare common training arguments.
    training_arguments = {
        "X_train": X_train,
        "y_train": y_train,
        "numerical_columns": numerical_columns,
        "nominal_columns": nominal_columns,
        "numerical_analysis": numerical_analysis,
        "categorical_analysis": categorical_analysis,
        "ordinal_categories": ordinal_categories,
        "approved_high_cardinality": approved_high_cardinality,
        "max_onehot_categories": max_onehot_categories,
    }

    # 8. Train all registered models.
    if task_type == "classification":
        training_results = train_classification_models(
            **training_arguments
        )
    else:
        training_results = train_regression_models(
            **training_arguments
        )

    # 9. Evaluate each successfully trained model.
    model_results = {}

    for model_name, result in training_results.items():

        if result["status"] == "failed":
            model_results[model_name] = {
                "status": "failed",
                "model": None,
                "y_pred": None,
                "evaluation_metrics": None,
                "error": result["error"],
            }
            continue

        fitted_model = result["model"]

        try:
            y_pred = fitted_model.predict(X_test)

            if task_type == "classification":
                evaluation_metrics = evaluate_classification(
                    y_test,
                    y_pred,
                )
            else:
                evaluation_metrics = evaluate_regression(
                    y_test,
                    y_pred,
                )

            model_results[model_name] = {
                "status": "completed",
                "model": fitted_model,
                "y_pred": y_pred,
                "evaluation_metrics": evaluation_metrics,
                "error": None,
            }

        except Exception as e:
            model_results[model_name] = {
                "status": "failed",
                "model": fitted_model,
                "y_pred": None,
                "evaluation_metrics": None,
                "error": str(e),
            }

    # 10. Calculate experiment summary.
    successful_models = sum(
        result["status"] == "completed"
        for result in model_results.values()
    )

    failed_models = sum(
        result["status"] == "failed"
        for result in model_results.values()
    )

    if successful_models == len(model_results):
        experiment_status = "completed"
    elif successful_models > 0:
        experiment_status = "partially_completed"
    else:
        experiment_status = "failed"

    # 11. Rank successfully evaluated models.
    ranking_input = {
        model_name: {
            "status": result["status"],
            "metrics": result["evaluation_metrics"],
        }
        for model_name, result in model_results.items()
    }

    ranking_result = rank_models(
        model_results=ranking_input,
        metric_configuration=metric_configuration,
        top_n=3,
    )

    # 12. Preserve backward compatibility.
    baseline_model = next(
        (
            result["model"]
            for result in model_results.values()
            if result["status"] == "completed"
        ),
        None,
    )
    # 13. Return experiment results.
    return {
        "status": experiment_status,
        "task_type": task_type,
        "target_column": target_column,
        "use_case": use_case,
        "metric_configuration": metric_configuration,
        "model": baseline_model,
        "model_results": model_results,
        "ranking": ranking_result,
        "successful_models": successful_models,
        "failed_models": failed_models,
        "total_models": len(model_results),
        "X_test": X_test,
        "y_test": y_test,
        "numerical_analysis": numerical_analysis,
        "categorical_analysis": categorical_analysis,
        "excluded_features": excluded_features,
        "dataset_intelligence": dataset_intelligence,
    }
