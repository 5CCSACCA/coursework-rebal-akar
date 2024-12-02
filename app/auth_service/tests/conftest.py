import pytest
from httpx import AsyncClient, ASGITransport
from main import app  # Ensure this imports your FastAPI app correctly
from database.mongodb import connect_to_mongo, close_mongo_connection, mongodb
from core.config import settings
import motor.motor_asyncio
import asyncio

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def async_client():
    """Create an instance of the FastAPI test client."""
    test_db_url = settings.MONGODB_URL
    test_db_name = "hatespeech_test_db"

    # Create a MongoDB client
    test_client = motor.motor_asyncio.AsyncIOMotorClient(test_db_url)

    # Modify the settings for testing
    original_db_name = settings.DATABASE_NAME
    settings.DATABASE_NAME = test_db_name

    # Connect to the test database
    await connect_to_mongo()

    # Drop the test database before tests
    await test_client.drop_database(test_db_name)

    # Use FastAPI's app with ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    # Drop the test database after tests
    await test_client.drop_database(test_db_name)

    # Restore the original database name
    settings.DATABASE_NAME = original_db_name

    # Close the MongoDB connection
    await close_mongo_connection()

@pytest.fixture
async def create_user(async_client):
    """Fixture to create a user for testing."""
    user_data = {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "Password123!"
    }

    # Cleanup conflicting users
    await mongodb.db.users.delete_many({
        "$or": [
            {"username": user_data["username"]},
            {"email": user_data["email"]}
        ]
    })

    response = await async_client.post("/users/register", json=user_data)
    assert response.status_code == 201, f"Failed to create user in setup. Response: {response.text}"
    return response.json()

@pytest.fixture(autouse=True)
async def cleanup():
    """Clean up the database after each test."""
    yield
    await mongodb.db.users.delete_many({})
    await mongodb.db.predictions.delete_many({})
