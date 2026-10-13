
import pandas as pd
from io import BytesIO
from fastapi import UploadFile


async def read_csv_upload(file: UploadFile) -> pd.DataFrame:

    if not file.filename:
        raise ValueError("No file was uploaded")

    if not file.filename.lower().endswith(".csv"):
        raise ValueError("Only CSV files are allowed")

    content = await file.read()

    # 4. Check whether the uploaded file is empty
    if not content:
        raise ValueError("Uploaded file is empty")

    try:
        df = pd.read_csv(BytesIO(content))
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        raise ValueError(
            "Invalid or malformed CSV file"
        ) from exc

    return df
