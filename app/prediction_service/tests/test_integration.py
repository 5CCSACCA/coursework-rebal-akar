# prediction_service/tests/test_integration.py

import pytest
from httpx import AsyncClient
from main import app
from database.mongodb import mongodb
from core.config import settings
from jose import jwt

@pytest.mark.anyio
async def test_prediction_flow(app_client, test_user_data, auth_headers):
    """
    Test the full prediction flow including authentication, prediction, and database storage.
    """
    # 1. Test prediction endpoint
    prediction_request = {
        "input_texts": ["Integration test sentence."]
    }
    response = await app_client.post("/predict", json=prediction_request, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1

    # 2. Test retrieving predictions
    response = await app_client.get("/predict/predictions", headers=auth_headers)
    assert response.status_code == 200
    predictions = response.json()
    assert len(predictions) >= 1
    assert predictions[0]["input_text"] == "Integration test sentence."

    # 3. Verify data in the database
    user = await mongodb.db.users.find_one({"username": test_user_data["username"]})
    user_id = str(user["_id"])
    stored_predictions = await mongodb.db.predictions.find({"user_id": user_id}).to_list(length=100)
    assert len(stored_predictions) >= 1
    assert stored_predictions[0]["input_text"] == "Integration test sentence."

    # 4. Test with invalid token
    invalid_headers = {"Authorization": "Bearer invalidtoken"}
    response = await app_client.post("/predict", json=prediction_request, headers=invalid_headers)
    assert response.status_code == 401

@pytest.mark.anyio
async def test_cross_service_integration(app_client, test_user_data):
    """
    Test integration with the authentication service.
    """
    # 1. Attempt to predict without authentication
    prediction_request = {
        "input_texts": ["Test without authentication."]
    }
    response = await app_client.post("/predict", json=prediction_request)
    assert response.status_code == 401

    # 2. Authenticate user
    auth_service_url = "http://localhost:8000"  # Adjust if necessary

    async with AsyncClient(base_url=auth_service_url) as auth_client:
        # Login user
        response = await auth_client.post("/users/login", data={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        })
        assert response.status_code == 200
        token = response.json()["access_token"]

    auth_headers = {"Authorization": f"Bearer {token}"}

    # 3. Attempt prediction with valid authentication
    response = await app_client.post("/predict", json=prediction_request, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
