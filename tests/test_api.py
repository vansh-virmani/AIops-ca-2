
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "location": "Whitefield",
            "total_sqft": 1200,
            "bhk": 2,
            "bath": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "predicted_price_lakhs" in body
    assert body["unit"] == "lakhs"


def test_invalid_area():
    response = client.post(
        "/predict",
        json={
            "location": "Whitefield",
            "total_sqft": -10,
            "bhk": 2,
            "bath": 2,
        },
    )
    assert response.status_code == 422
