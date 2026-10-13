
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_profile_csv_success():
    csv_content = (
        "Age,Salary,Churn\n"
        "25,30000,Yes\n"
        "30,45000,No\n"
        "25,30000,Yes\n"
        "40,,No\n"
    )

    response = client.post(
        "/profile",
        files={
            "file": (
                "data.csv",
                csv_content.encode("utf-8"),
                "text/csv",
            )
        },
        data={"target_column": "Churn"},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["rows"] == 4
    assert result["columns"] == 3
    assert result["duplicate_rows"] == 1
    assert result["missing_target_values"] == 0


def test_profile_rejects_non_csv_file():
    response = client.post(
        "/profile",
        files={
            "file": ("data.txt", b"Age,Churn\n25,Yes\n", "text/plain")
        },
        data={"target_column": "Churn"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only CSV files are allowed"


def test_profile_rejects_empty_upload():
    response = client.post(
        "/profile",
        files={"file": ("data.csv", b"", "text/csv")},
        data={"target_column": "Churn"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"


def test_profile_rejects_missing_target_column():
    response = client.post(
        "/profile",
        files={
            "file": (
                "data.csv",
                b"Age,Salary\n25,30000\n30,45000\n",
                "text/csv",
            )
        },
        data={"target_column": "Churn"},
    )

    assert response.status_code == 400


def test_profile_rejects_empty_csv():
    response = client.post(
        "/profile",
        files={"file": ("data.csv", b"", "text/csv")},
        data={"target_column": "Churn"},
    )

    assert response.status_code == 400
