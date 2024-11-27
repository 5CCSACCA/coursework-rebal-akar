# prediction_service/main.py

from fastapi import FastAPI
from routers import predict
from database.mongodb import connect_to_mongo, close_mongo_connection
from routers import users, health
from routers import users, predict, health

app = FastAPI(
    title="Prediction Service",
    description="Handles predictions using the machine learning model.",
    version="1.0.0",
)

app.add_event_handler("startup", connect_to_mongo)
app.add_event_handler("shutdown", close_mongo_connection)

app.include_router(users.router, tags=["Users"], prefix="/users")
app.include_router(predict.router, tags=["Predictions"], prefix="/predict")
app.include_router(health.router, tags=["Health"])

