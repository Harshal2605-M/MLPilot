from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
from io import BytesIO

from app.config import settings
from app.ml.profiling import profile_dataset
from app.services.dataset_service import read_csv_upload

app = FastAPI(
    title = settings.app_name,
    version = settings.app_version
)

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