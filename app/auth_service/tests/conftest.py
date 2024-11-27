# tests/conftest.py

import pytest
from httpx import AsyncClient
from motor.motor_asyncio import AsyncIOMotorClient
from main import app
from database.mongodb import mongodb
from core.config import settings

@pytest.fixture(scope="session")
def anyio_backend():
    return 'asyncio'

@pytest.fixture(scope="session")
async def test_app():
    # Setup the database connection
    mongodb.client = AsyncIOMotorClient(settings.MONGODB_URL)
    mongodb.db = mongodb.client[settings.DATABASE_NAME + "_test"]  # Use a separate test database

    yield app

    # Teardown: Drop the test database
    mongodb.client.drop_database(settings.DATABASE_NAME + "_test")
    mongodb.client.close()
