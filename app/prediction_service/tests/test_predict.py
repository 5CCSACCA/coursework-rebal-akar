import pytest

@pytest.mark.asyncio
async def test_predict_success(async_client_fixture, auth_headers, mock_database):
    """Test successful prediction request."""
    request_data = {"input_texts": ["This is great", "I hate this"]}
    response = await async_client_fixture.post("/predict/predict", headers=auth_headers, json=request_data)

    assert response.status_code == 201
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2

@pytest.mark.asyncio
async def test_predict_invalid_token(async_client_fixture, mock_invalid_auth, mock_database):
    """Test prediction with an invalid token."""
    headers = {"Authorization": "Bearer invalidtoken"}
    request_data = {"input_texts": ["This is a test"]}
    response = await async_client_fixture.post("/predict/predict", headers=headers, json=request_data)

    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials"

@pytest.mark.asyncio
async def test_predict_missing_auth(async_client_fixture, mock_invalid_auth, mock_database):
    """Test prediction without an authorization token."""
    request_data = {"input_texts": ["This is a test"]}
    response = await async_client_fixture.post("/predict/predict", json=request_data)

    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials"

@pytest.mark.asyncio
async def test_predict_empty_input(async_client_fixture, auth_headers, mock_database):
    """Test prediction with empty input."""
    request_data = {"input_texts": []}
    response = await async_client_fixture.post("/predict/predict", headers=auth_headers, json=request_data)

    assert response.status_code == 400
    assert response.json()["detail"] == "No input texts provided"

@pytest.mark.asyncio
async def test_predict_exceeds_batch_size(async_client_fixture, auth_headers, mock_database):
    """Test prediction request exceeding batch size."""
    request_data = {"input_texts": ["Test"] * 101}
    response = await async_client_fixture.post("/predict/predict", headers=auth_headers, json=request_data)

    assert response.status_code == 400
    assert "Batch size exceeds maximum limit" in response.json()["detail"]
