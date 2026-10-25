
import pandas as pd

from app.ml.preparation import (
    analyze_categorical_features,
)

X_train = pd.DataFrame({
    "Education_Level": [
        "Graduate",
        "Undergraduate",
        "Postgraduate",
        "Graduate",
    ],
    "Satisfaction": [
        "Good",
        "Excellent",
        "Poor",
        "Fair",
    ],
    "City": [
        "Pune",
        "Mumbai",
        "Delhi",
        "Pune",
    ],
})

approved_ordinal_categories = {
    "Education_Level": [
        "Undergraduate",
        "Graduate",
        "Postgraduate",
    ],
    "Satisfaction": [
        "Poor",
        "Fair",
        "Good",
        "Excellent",
    ],
}

result = analyze_categorical_features(
    X_train,
    ordinal_categories=approved_ordinal_categories,
)

for column, details in result.items():
    print(column, details["encoding_strategy"])
