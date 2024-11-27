# prediction_service/schemas/prediction.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    input_texts: List[str]

class PredictionResult(BaseModel):
    prediction: int
    prediction_label: str
    probabilities: List[float]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    hate_offensive_tweets: List[PredictionResult]


class PredictionOut(BaseModel):
    id: Optional[str]
    user_id: str
    input_text: str
    prediction_result: PredictionResult
    created_at: datetime