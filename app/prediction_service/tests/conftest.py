import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import MagicMock, AsyncMock, patch
from jose import jwt
from main import app  
from auth.auth import get_current_user 
import logging
import pytest
from fastapi import HTTPException, status


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

SECRET_KEY = "ABCDEFG"
ALGORITHM = "HS256"

def override_get_current_user_valid():
    return {"_id": "testuserid", "username": "testuser"}

def override_get_current_user_invalid():
    raise HTTPException(
        status_code=401, detail="Could not validate credentials"
    )

@pytest.fixture
def mock_invalid_auth():
    app.dependency_overrides[get_current_user] = override_get_current_user_invalid

@pytest.fixture
def mock_valid_auth():
    app.dependency_overrides[get_current_user] = override_get_current_user_valid


@pytest_asyncio.fixture
def valid_token():
    """Generate a valid JWT token for testing."""
    payload = {"sub": "testuser"}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

@pytest.fixture
def auth_headers(valid_token):
    """Provide authorization headers using the valid token."""
    return {"Authorization": f"Bearer {valid_token}"}

@pytest_asyncio.fixture
async def async_client_fixture():
    """Create an AsyncClient for testing with ASGITransport."""
    logger.debug("Setting up AsyncClient...")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as ac:
        yield ac
    logger.debug("AsyncClient teardown completed.")

@pytest.fixture(autouse=True)
def override_get_current_user():
    """Override the get_current_user dependency to return a mock user."""
    async def mock_get_current_user(token: str = None):
        return {"username": "testuser", "_id": "testuserid"}

    app.dependency_overrides[get_current_user] = mock_get_current_user
    yield
    app.dependency_overrides.clear()

@pytest.fixture(autouse=True)
def mock_database():
    """Mock the database operations to prevent real DB interactions."""
    logger.debug("Patching 'database.mongodb.mongodb' with MagicMock...")
    with patch("database.mongodb.mongodb") as mock_mongodb:
        # Initialize db attribute
        mock_mongodb.db = MagicMock()
        logger.debug("Mock database initialized.")

        # Mock the predictions collection
        mock_mongodb.db.predictions = MagicMock()
        logger.debug("Mock predictions collection initialized.")

        # Mock methods of the predictions collection
        mock_mongodb.db.predictions.insert_many = AsyncMock(return_value=None)
        mock_mongodb.db.predictions.find = AsyncMock(return_value=[])
        mock_mongodb.db.predictions.count_documents = AsyncMock(return_value=0)
        mock_mongodb.db.predictions.delete_many = AsyncMock(return_value=None)
        
        yield mock_mongodb
        logger.debug("Tearing down mock database.")

@pytest.fixture(autouse=True)
def override_mongodb(mock_database):
    """Override the mongodb instance in predict router."""
    from routers import predict
    predict.mongodb = mock_database 
    yield


