from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    input_texts: List[str] = Field(..., 
                                   example=["I hate this!", "This is great."],
                                   description="List of input texts to analyze for hate speech.")

class PredictionResult(BaseModel):
    input_text: str
    prediction: int = Field(..., example=1, description="Numeric representation of the prediction.")
    prediction_label: str = Field(..., example="Offensive", description="Label corresponding to the prediction.")
    probabilities: List[float]  = Field(..., example=[0.1, 0.7, 0.2], description="Probability scores on wether its neutral, offensive or hatespeech.")

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    hate_offensive_tweets: List[PredictionResult]


class PredictionOut(BaseModel):
    id: Optional[str]
    user_id: str
    input_text: str
    prediction_result: PredictionResult
    created_at: datetime