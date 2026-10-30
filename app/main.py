import math
import pandas as pd
from io import BytesIO
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.encoders import jsonable_encoder


from app.config import settings
from app.ml.profiling import profile_dataset
from app.services.dataset_service import read_csv_upload
from app.ml.engine import run_baseline_experiment

app = FastAPI(
    title = settings.app_name,
    version = settings.app_version
)

def make_json_safe(value):
    """Convert NumPy values and nested structures into JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value

@app.get("/health") 
def health_check():
    return{
        "status" : "healthy",
        "app" : settings.app_name,
        "version" : settings.app_version,
    }

@app.post("/profile")
async def profile_csv(file: UploadFile = File(...), target_column: str = Form(...)):

    try :
        df = await read_csv_upload(file)
        profile = profile_dataset(df, target_column)
        return profile

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc




@app.post("/experiment")
async def run_experiment(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    selected_task_type: str | None = Form(None),
    use_case: str | None = Form(None),
):
    try:
        df = await read_csv_upload(file)

        result = run_baseline_experiment(
            df=df,
            target_column=target_column,
            selected_task_type=selected_task_type,
            use_case=use_case,
        )

        # Extract serializable results for each model.
        model_results = {
            model_name: {
                "status": model_result["status"],
                "evaluation_metrics": model_result["evaluation_metrics"],
                "error": model_result["error"],
            }
            for model_name, model_result in result["model_results"].items()
        }

        response = {
            "status": result["status"],
            "task_type": result["task_type"],
            "target_column": result["target_column"],
            "use_case": result["use_case"],

            "dataset_summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "features": len(result["X_test"].columns),
                "training_rows": len(df) - len(result["X_test"]),
                "test_rows": len(result["X_test"]),
            },

            "metric_configuration": result["metric_configuration"],

            "model_results": model_results,

            "ranking": result["ranking"],

            "dataset_intelligence": result["dataset_intelligence"],

            "numerical_analysis": result["numerical_analysis"],
            "categorical_analysis": result["categorical_analysis"],
            "excluded_features": result["excluded_features"],

            "successful_models": result["successful_models"],
            "failed_models": result["failed_models"],
            "total_models": result["total_models"],

            "message": "Experiment execution finished.",
        }

        return jsonable_encoder(make_json_safe(response))

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
