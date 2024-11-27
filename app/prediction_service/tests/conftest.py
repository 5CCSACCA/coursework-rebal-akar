# prediction_service/tests/conftest.py

import pytest
from httpx import AsyncClient
from motor.motor_asyncio import AsyncIOMotorClient
from main import app
from database.mongodb import mongodb
from core.config import settings
from fastapi.testclient import TestClient
from unittest.mock import patch
from datetime import datetime, timedelta
from jose import jwt
from typing import Dict


@pytest.fixture(scope="session")
def anyio_backend():
    return 'asyncio'

@pytest.fixture(scope="session")
async def test_app():
    # Setup the database connection
    mongodb.client = AsyncIOMotorClient(settings.MONGODB_URL)
    mongodb.db = mongodb.client[settings.DATABASE_NAME + "_test"]  # Use a separate test database

    # Start the app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    # Teardown: Drop the test database
    mongodb.client.drop_database(settings.DATABASE_NAME + "_test")
    mongodb.client.close()

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def test_user_data():
    return {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "Test@1234"
    }

@pytest.fixture
def auth_header(test_user_data):
    # Generate a valid token for the test user
    token = jwt.encode(
        {"sub": test_user_data["username"]},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def mock_predict_text():
    # Mock the predict_text function
    with patch("ml_model.model.predict_text") as mock_predict:
        def side_effect(texts):
            return [
                {
                    "input_text": text,
                    "prediction": 0,
                    "prediction_label": "Neutral",
                    "probabilities": [0.9, 0.05, 0.05],
                }
                for text in texts
            ]
        mock_predict.side_effect = side_effect
        yield mock_predict

@pytest.fixture
async def auth_headers(test_user_data):
    # Simulate registration and login via the auth service
    auth_service_url = "http://localhost:8000"  # Adjust if necessary

    async with AsyncClient(base_url=auth_service_url) as client:
        # Register user
        response = await client.post("/users/register", json=test_user_data)
        assert response.status_code == 200

        # Login user
        response = await client.post("/users/login", data={
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        })
        assert response.status_code == 200
        token = response.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}
