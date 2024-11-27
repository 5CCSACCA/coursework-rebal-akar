# prediction_service/tests/test_predict.py

import pytest
from httpx import AsyncClient
from jose import jwt
from datetime import datetime, timedelta
from core.config import settings
from database.mongodb import mongodb
from bson.objectid import ObjectId
from unittest.mock import patch

@pytest.mark.anyio
class TestPrediction:

    async def test_predict_success(self, test_app, test_user_data, auth_header, mock_predict_text):
        """
        Test successful prediction with valid authentication and input.
        """
        # Prepare request data
        request_data = {
            "input_texts": ["This is a test sentence.", "Another test sentence."]
        }

        response = await test_app.post("/predict", json=request_data, headers=auth_header)
        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert "hate_offensive_tweets" in data
        assert len(data["predictions"]) == 2

        for prediction in data["predictions"]:
            assert prediction["prediction"] == 0
            assert prediction["prediction_label"] == "Neutral"
            assert prediction["probabilities"] == [0.9, 0.05, 0.05]

        # Check that predictions are stored in the database
        user = await mongodb.db.users.find_one({"username": test_user_data["username"]})
        user_id = str(user["_id"])
        predictions = await mongodb.db.predictions.find({"user_id": user_id}).to_list(length=100)
        assert len(predictions) == 2
        for prediction in predictions:
            assert prediction["user_id"] == user_id
            assert prediction["input_text"] in request_data["input_texts"]

    async def test_predict_unauthenticated(self, test_app):
        """
        Test prediction without authentication.
        """
        request_data = {
            "input_texts": ["This is a test sentence."]
        }
        response = await test_app.post("/predict", json=request_data)
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    async def test_predict_no_input_texts(self, test_app, auth_header):
        """
        Test prediction with empty input_texts.
        """
        request_data = {
            "input_texts": []
        }
        response = await test_app.post("/predict", json=request_data, headers=auth_header)
        assert response.status_code == 400
        assert response.json()["detail"] == "No input texts provided"

    async def test_predict_exceeds_batch_size(self, test_app, auth_header):
        """
        Test prediction with batch size exceeding the limit.
        """
        request_data = {
            "input_texts": ["Test"] * 101  # Exceeding the limit of 100
        }
        response = await test_app.post("/predict", json=request_data, headers=auth_header)
        assert response.status_code == 400
        assert response.json()["detail"] == "Batch size exceeds maximum limit of 100"

    async def test_predict_max_predictions_per_user(self, test_app, test_user_data, auth_header, mock_predict_text):
        """
        Test that the oldest predictions are deleted when max predictions per user is exceeded.
        """
        # Set MAX_PREDICTIONS_PER_USER to a smaller number for testing
        from routers.predict import MAX_PREDICTIONS_PER_USER
        original_max = MAX_PREDICTIONS_PER_USER
        MAX_PREDICTIONS_PER_USER = 5  # For testing

        try:
            # Insert initial predictions to reach the limit
            for _ in range(5):
                await test_app.post("/predict", json={"input_texts": ["Test"]}, headers=auth_header)

            # Verify that 5 predictions exist
            user = await mongodb.db.users.find_one({"username": test_user_data["username"]})
            user_id = str(user["_id"])
            predictions = await mongodb.db.predictions.find({"user_id": user_id}).to_list(length=100)
            assert len(predictions) == 5

            # Insert one more prediction
            await test_app.post("/predict", json={"input_texts": ["New Test"]}, headers=auth_header)

            # Verify that only 5 predictions exist and the oldest is deleted
            predictions = await mongodb.db.predictions.find({"user_id": user_id}).sort("created_at", 1).to_list(length=100)
            assert len(predictions) == 5
            input_texts = [prediction["input_text"] for prediction in predictions]
            assert "New Test" in input_texts
            assert "Test" in input_texts  # Since "Test" was used multiple times

        finally:
            # Reset MAX_PREDICTIONS_PER_USER to its original value
            MAX_PREDICTIONS_PER_USER = original_max

    async def test_get_user_predictions(self, test_app, test_user_data, auth_header, mock_predict_text):
        """
        Test retrieving user predictions with pagination.
        """
        # Insert multiple predictions
        for i in range(10):
            await test_app.post("/predict", json={"input_texts": [f"Test {i}"]}, headers=auth_header)

        # Retrieve predictions with skip and limit
        response = await test_app.get("/predict/predictions?skip=0&limit=5", headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

        # Verify that the predictions are ordered by created_at descending
        created_ats = [prediction["created_at"] for prediction in data]
        assert created_ats == sorted(created_ats, reverse=True)

        # Retrieve next page
        response = await test_app.get("/predict/predictions?skip=5&limit=5", headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    async def test_get_user_predictions_unauthenticated(self, test_app):
        """
        Test retrieving predictions without authentication.
        """
        response = await test_app.get("/predict/predictions")
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    async def test_predict_with_hate_speech(self, test_app, auth_header):
        """
        Test prediction where the input text is classified as hate speech.
        """
        # Mock the predict_text function to return a hate speech prediction
        with patch("ml_model.model.predict_text") as mock_predict:
            def side_effect(texts):
                return [
                    {
                        "input_text": text,
                        "prediction": 2,
                        "prediction_label": "Hate",
                        "probabilities": [0.1, 0.2, 0.7],
                    }
                    for text in texts
                ]
            mock_predict.side_effect = side_effect

            request_data = {
                "input_texts": ["This is a hateful sentence."]
            }
            response = await test_app.post("/predict", json=request_data, headers=auth_header)
            assert response.status_code == 200
            data = response.json()

            assert len(data["predictions"]) == 1
            assert data["predictions"][0]["prediction_label"] == "Hate"
            assert len(data["hate_offensive_tweets"]) == 1
            assert data["hate_offensive_tweets"][0]["prediction_label"] == "Hate"

    async def test_predict_model_error(self, test_app, auth_header):
        """
        Test prediction when the model raises an exception.
        """
        # Mock the predict_text function to raise an exception
        with patch("ml_model.model.predict_text") as mock_predict:
            mock_predict.side_effect = Exception("Model error")

            request_data = {
                "input_texts": ["This will cause an error."]
            }
            response = await test_app.post("/predict", json=request_data, headers=auth_header)
            assert response.status_code == 500
            assert response.json()["detail"] == "Internal Server Error"

    async def test_predict_invalid_input(self, test_app, auth_header):
        """
        Test prediction with invalid input data.
        """
        # Missing input_texts field
        request_data = {}
        response = await test_app.post("/predict", json=request_data, headers=auth_header)
        assert response.status_code == 422

        # input_texts is not a list
        request_data = {"input_texts": "This is not a list."}
        response = await test_app.post("/predict", json=request_data, headers=auth_header)
        assert response.status_code == 422

    async def test_get_predictions_no_predictions(self, test_app, auth_header):
        """
        Test retrieving predictions when the user has no predictions.
        """
        response = await test_app.get("/predict/predictions", headers=auth_header)
        assert response.status_code == 200
        data = response.json()
        assert data == []

