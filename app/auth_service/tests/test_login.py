import pytest

@pytest.mark.asyncio
async def test_register_success(async_client):
    """Test successful user registration."""
    user_data = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "Password123!"
    }
    response = await async_client.post("/users/register", json=user_data)
    assert response.status_code == 201  # Changed from 200 to 201 as per router definition
    assert response.json()["username"] == "newuser"
    assert response.json()["email"] == "newuser@example.com"

@pytest.mark.asyncio
async def test_register_existing_username(async_client, create_user):
    """Test registration with an existing username."""
    user_data = {
        "username": "testuser",  # Already created in create_user fixture
        "email": "another@example.com",
        "password": "Password123!"
    }
    response = await async_client.post("/users/register", json=user_data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Username or email already registered"

@pytest.mark.asyncio
async def test_register_existing_email(async_client, create_user):
    """Test registration with an existing email."""
    user_data = {
        "username": "anotheruser",
        "email": "testuser@example.com",  # Already created in create_user fixture
        "password": "Password123!"
    }
    response = await async_client.post("/users/register", json=user_data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Username or email already registered"

@pytest.mark.asyncio
async def test_register_invalid_password(async_client):
    """Test registration with an invalid password."""
    user_data = {
        "username": "weakpassworduser",
        "email": "weak@example.com",
        "password": "weak"  # Does not meet complexity requirements
    }
    response = await async_client.post("/users/register", json=user_data)
    assert response.status_code == 422  # Unprocessable Entity
