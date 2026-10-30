
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR


def get_classification_models():
    """
    Return predefined classification models.
    Each call creates fresh estimator instances.
    """

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            random_state=42,
        ),
    }


def get_regression_models():
    """
    Return predefined regression models.
    Each call creates fresh estimator instances.
    """

    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsRegressor(
            n_neighbors=5,
        ),
        "Support Vector Regression": SVR(
            kernel="rbf",
        ),
    }


def get_models(task_type):
    """
    Return models based on the detected ML task.
    """

    if task_type == "classification":
        return get_classification_models()

    if task_type == "regression":
        return get_regression_models()

    raise ValueError(
        "Unsupported task type. Choose classification or regression."
    )
