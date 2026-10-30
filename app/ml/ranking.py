
import math


# Metrics where a higher value indicates better performance
HIGHER_IS_BETTER = {
    "accuracy",
    "precision",
    "precision_weighted",
    "precision_macro",
    "recall",
    "recall_weighted",
    "recall_macro",
    "f1",
    "f1_weighted",
    "f1_macro",
    "r2",
}

# Metrics where a lower value indicates better performance
LOWER_IS_BETTER = {
    "mae",
    "mse",
    "rmse",
}


def validate_metric_configuration(metric_configuration):
    """
    Validate primary and secondary metric configuration.
    """

    if not isinstance(metric_configuration, dict):
        raise ValueError("Metric configuration must be a dictionary")

    primary_metric = metric_configuration.get("primary_metric")
    secondary_metric = metric_configuration.get("secondary_metric")

    if not primary_metric:
        raise ValueError("Primary metric is required")

    if primary_metric not in HIGHER_IS_BETTER | LOWER_IS_BETTER:
        raise ValueError(f"Unsupported primary metric: {primary_metric}")

    if secondary_metric is not None:
        if secondary_metric not in HIGHER_IS_BETTER | LOWER_IS_BETTER:
            raise ValueError(
                f"Unsupported secondary metric: {secondary_metric}"
            )

        if secondary_metric == primary_metric:
            raise ValueError(
                "Primary and secondary metrics must be different"
            )

    return primary_metric, secondary_metric


def rank_models(model_results, metric_configuration, top_n=3):
    """
    Rank successfully trained models using primary and secondary metrics.
    """

    if not isinstance(model_results, dict):
        raise ValueError("Model results must be a dictionary")

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise ValueError("top_n must be a positive integer")

    primary_metric, secondary_metric = validate_metric_configuration(
        metric_configuration
    )

    ranking_candidates = []
    excluded_models = {}

    for model_name, result in model_results.items():

        # Ignore failed models
        if result.get("status") != "completed":
            excluded_models[model_name] = "Model training was not completed"
            continue

        metrics = result.get("metrics", {})

        # Primary metric must exist and be a valid finite number
        primary_value = metrics.get(primary_metric)

        if (
            isinstance(primary_value, bool)
            or not isinstance(primary_value, (int, float))
            or not math.isfinite(primary_value)
        ):
            excluded_models[model_name] = (
                f"Missing or invalid primary metric: {primary_metric}"
            )
            continue

        secondary_value = None

        if secondary_metric is not None:
            value = metrics.get(secondary_metric)

            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                secondary_value = value

        ranking_candidates.append({
            "model_name": model_name,
            "primary_metric": primary_value,
            "secondary_metric": secondary_value,
        })

    def sort_key(candidate):

        primary = candidate["primary_metric"]

        if primary_metric in HIGHER_IS_BETTER:
            primary = -primary

        secondary = candidate["secondary_metric"]

        if secondary_metric is None or secondary is None:
            secondary = float("inf")
        elif secondary_metric in HIGHER_IS_BETTER:
            secondary = -secondary

        return (
            primary,
            secondary,
            candidate["model_name"].lower()
        )

    ranking_candidates.sort(key=sort_key)

    ranked_models = []

    for index, candidate in enumerate(ranking_candidates[:top_n], start=1):

        ranked_models.append({
            "rank": index,
            "model_name": candidate["model_name"],
            "primary_metric": candidate["primary_metric"],
            "secondary_metric": candidate["secondary_metric"],
        })

    return {
        "ranking": ranked_models,
        "top_models": [
            model["model_name"] for model in ranked_models
        ],
        "excluded_models": excluded_models,
    }
