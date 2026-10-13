import pandas as pd
from app.ml.profiling import profile_dataset

df = pd.DataFrame({
    "Age": [25, 30, 35, 40],
    "Churn": ["Yes", "No", None, "Yes"]
})
profile = profile_dataset(df, "Churn")
print(profile)