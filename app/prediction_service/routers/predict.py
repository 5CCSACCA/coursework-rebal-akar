"""
Prediction Router for Prediction Service

This module handles prediction requests and retrieval of past predictions.
"""
import logging
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
logger = logging.getLogger("prediction_service.routers.predict")

MAX_PREDICTIONS_PER_USER = 100  # Maximum predictions allowed per user

@router.post(
    "/predict",
    response_model=BatchPredictionResponse,
    status_code=201,
    summary="Make predictions on input texts",
    description="""
    Analyze a list of input texts to detect hate speech, offensive language, or neutral content.

    - Maximum of 100 input texts per request.

    - Also returns the predictions of hate speech texts below the predictions 

    **Example Request:**

    ```json
    {
      "input_texts": [
        "I hate this!",
        "This is great."
      ]
    }
    ```

    **Example Response:**

    ```json
    {
      "predictions": [
        {
          "prediction": 2,
          "prediction_label": "Hate",
          "probabilities": [0.05, 0.10, 0.85]
        },
        {
          "prediction": 0,
          "prediction_label": "Neutral",
          "probabilities": [0.90, 0.05, 0.05]
        }
      ],
      "hate_offensive_tweets": [
        {
          "prediction": 2,
          "prediction_label": "Hate",
          "probabilities": [0.05, 0.10, 0.85]
        }
      ]
    }
    ```
    """
)
async def predict(
    request: PredictionRequest, current_user: dict = Depends(get_current_user)
):
    """
    Make predictions on a batch of input texts.

    Args:
        request (PredictionRequest): The prediction request containing input texts.
        current_user (dict, optional): The authenticated user.

    Returns:
        BatchPredictionResponse: The prediction results and filtered hate/offensive tweets.

    Raises:
        HTTPException: If the input is invalid or if prediction processing fails.
    """
    username = current_user["username"]
    logger.info(f"User '{username}' initiated a prediction request with {len(request.input_texts)} texts.")
    
    texts = request.input_texts
    if not texts:
        logger.warning(f"User '{username}' submitted an empty prediction request.")
        raise HTTPException(status_code=400, detail="No input texts provided")
    if len(texts) > 100:
        logger.warning(f"User '{username}' submitted a prediction request exceeding the limit: {len(texts)} texts.")
        raise HTTPException(
            status_code=400, detail="Batch size exceeds maximum limit of 100"
        )

    try:
        prediction_results = await predict_text(texts)
        logger.info(f"User '{username}' prediction processing completed successfully.")
    except Exception as e:
        logger.exception(f"Error during prediction processing for user '{username}': {e}")
        raise HTTPException(status_code=500, detail="Prediction processing failed")

    # Prepare predictions to insert
    predictions_to_insert = []
    for result in prediction_results:
        prediction_data = {
            "user_id": current_user["user_id"],
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
        try:
            await mongodb.db.predictions.insert_many(predictions_to_insert)
            logger.info(f"Inserted {len(predictions_to_insert)} predictions for user '{username}'.")
        except Exception as e:
            logger.exception(f"Failed to insert predictions for user '{username}': {e}")
            raise HTTPException(status_code=500, detail="Failed to store predictions")

        # Count total predictions after insertion
        try:
            total_predictions = await mongodb.db.predictions.count_documents(
                {"user_id": current_user["user_id"]}
            )
            logger.debug(f"User '{username}' has a total of {total_predictions} predictions.")
        except Exception as e:
            logger.exception(f"Failed to count predictions for user '{username}': {e}")

        # If total exceeds the limit, delete the oldest predictions
        if total_predictions > MAX_PREDICTIONS_PER_USER:
            excess = total_predictions - MAX_PREDICTIONS_PER_USER
            logger.info(f"User '{username}' exceeded prediction limit by {excess}. Deleting oldest predictions.")

            try:
                oldest_predictions = await mongodb.db.predictions.find(
                    {"user_id": current_user["user_id"]}
                ).sort("created_at", ASCENDING).limit(excess).to_list(length=excess)

                oldest_ids = [prediction["_id"] for prediction in oldest_predictions]
                await mongodb.db.predictions.delete_many({"_id": {"$in": oldest_ids}})
                logger.info(f"Deleted {len(oldest_ids)} oldest predictions for user '{username}'.")
            except Exception as e:
                logger.exception(f"Failed to delete old predictions for user '{username}': {e}")
                raise HTTPException(status_code=500, detail="Failed to delete old predictions")

    # Filter hate and offensive tweets
    hate_offensive_tweets = [
        PredictionResult(**result)
        for result in prediction_results
        if result["prediction"] in [1, 2]
    ]
    logger.debug(f"Hate/Offensive tweets for user '{username}': {hate_offensive_tweets}")

    response = BatchPredictionResponse(
        predictions=[PredictionResult(**result) for result in prediction_results],
        hate_offensive_tweets=hate_offensive_tweets,
    )
    logger.info(f"Prediction response sent to user '{username}'.")
    return response

@router.get(
    "/predictions",
    response_model=List[PredictionOut],
    summary="Retrieve user predictions",
    description="""
    Fetch a list of past predictions made by the authenticated user.

    **Parameters:**
        Specify min and max range from your predictions
    - `skip` : min range, e.g 2 would fetch the 3rd prediction and above
    - `limit` : Specify the number of predictions to fetch, starting from after the skip
    """
)
async def get_user_predictions(
    skip: int = 0,
    limit: int = 10,
    current_user: dict = Depends(get_current_user),
):
    username = current_user["username"]
    user_id = current_user["user_id"]


    logger.info(f"User '{username}' requested to retrieve predictions with skip={skip}, limit={limit}.")
    
    
    try:
        predictions_cursor = (
            mongodb.db.predictions.find({"user_id": user_id})
            .sort("created_at", DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        predictions = []
        async for prediction in predictions_cursor:
            prediction_result_data = prediction["prediction_result"]

            prediction_result_data["input_text"] = prediction.get("input_text", "No input text available")
            prediction_out = PredictionOut(
                id=str(prediction["_id"]),
                user_id=prediction["user_id"],
                input_text=prediction.get("input_text", "No input text available"),
                prediction_result=PredictionResult(**prediction_result_data),                
                created_at=prediction["created_at"],
            )
            predictions.append(prediction_out)
        logger.info(f"User '{username}' retrieved {len(predictions)} predictions successfully.")
        return predictions
    except Exception as e:
        logger.exception(f"Failed to retrieve predictions for user '{username}': {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch predictions")
