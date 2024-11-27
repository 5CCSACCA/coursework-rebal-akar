# routers/predict.py

from fastapi import APIRouter, Depends, HTTPException, status
from schemas.prediction import (
    PredictionRequest,
    PredictionResult,
    BatchPredictionResponse,
    PredictionOut,
)
from auth.auth import get_current_user
from ml_model.model import predict_text
from database.mongodb import mongodb
from datetime import datetime, timedelta
from typing import List
from bson.objectid import ObjectId
from pymongo import DESCENDING, ASCENDING

router = APIRouter()

MAX_PREDICTIONS_PER_USER = 100  # Maximum predictions allowed per user

@router.post("/predict", response_model=BatchPredictionResponse)
async def predict(
    request: PredictionRequest, current_user: dict = Depends(get_current_user)
):
    texts = request.input_texts
    if not texts:
        raise HTTPException(status_code=400, detail="No input texts provided")
    if len(texts) > 100:
        raise HTTPException(
            status_code=400, detail="Batch size exceeds maximum limit of 100"
        )

    prediction_results = await predict_text(texts)

    # Prepare predictions to insert
    predictions_to_insert = []
    for result in prediction_results:
        prediction_data = {
            "user_id": str(current_user["_id"]),
            "input_text": result["input_text"],
            "prediction_result": {
                "prediction": result["prediction"],
                "prediction_label": result["prediction_label"],
                "probabilities": result["probabilities"],
            },
            "created_at": datetime.utcnow(),
        }
        predictions_to_insert.append(prediction_data)

    # Insert new predictions
    if predictions_to_insert:
        await mongodb.db.predictions.insert_many(predictions_to_insert)

        # Count total predictions after insertion
        total_predictions = await mongodb.db.predictions.count_documents(
            {"user_id": str(current_user["_id"])}
        )

        # If total exceeds the limit, delete the oldest predictions
        if total_predictions > MAX_PREDICTIONS_PER_USER:
            # Calculate number of predictions to delete
            excess = total_predictions - MAX_PREDICTIONS_PER_USER

            # Find the oldest predictions to delete
            oldest_predictions = await mongodb.db.predictions.find(
                {"user_id": str(current_user["_id"])}
            ).sort("created_at", ASCENDING).limit(excess).to_list(length=excess)

            # Extract their IDs
            oldest_ids = [prediction["_id"] for prediction in oldest_predictions]

            # Delete the oldest predictions
            await mongodb.db.predictions.delete_many({"_id": {"$in": oldest_ids}})

    # Filter hate and offensive tweets
    hate_offensive_tweets = [
        PredictionResult(**result)
        for result in prediction_results
        if result["prediction"] in [1, 2]
    ]

    response = BatchPredictionResponse(
        predictions=[PredictionResult(**result) for result in prediction_results],
        hate_offensive_tweets=hate_offensive_tweets,
    )
    return response


@router.get("/predictions", response_model=List[PredictionOut])
async def get_user_predictions(
    skip: int = 0,
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
):
    user_id = str(current_user["_id"])
    predictions_cursor = (
        mongodb.db.predictions.find({"user_id": user_id})
        .sort("created_at", DESCENDING)
        .skip(skip)
        .limit(limit)
    )
    predictions = []
    async for prediction in predictions_cursor:
        prediction_out = PredictionOut(
            id=str(prediction["_id"]),
            user_id=prediction["user_id"],
            input_text=prediction["input_text"],
            prediction_result=PredictionResult(**prediction["prediction_result"]),
            created_at=prediction["created_at"],
        )
        predictions.append(prediction_out)
    return predictions
