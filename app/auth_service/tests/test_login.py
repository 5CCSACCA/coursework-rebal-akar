# tests/test_login.py

import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_login_success(async_client, create_user):
    """Test successful user login."""
    login_data = {
        "username": "testuser",
        "password": "Password123!"
    }
    response = await async_client.post("/auth/users/login", data=login_data)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}, response: {response.text}"
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_login_invalid_username(async_client):
    """Test login with invalid username."""
    login_data = {
        "username": "nonexistentuser",
        "password": "Password123!"
    }
    response = await async_client.post("/auth/users/login", data=login_data)
    assert response.status_code == 400, f"Expected status code 400, got {response.status_code}, response: {response.text}"
    assert response.json()["detail"] == "Incorrect username or password"

@pytest.mark.asyncio
async def test_login_invalid_password(async_client, create_user):
    """Test login with invalid password."""
    login_data = {
        "username": "testuser",
        "password": "WrongPassword123!"
    }
    response = await async_client.post("/auth/users/login", data=login_data)
    assert response.status_code == 400, f"Expected status code 400, got {response.status_code}, response: {response.text}"
    assert response.json()["detail"] == "Incorrect username or password"
