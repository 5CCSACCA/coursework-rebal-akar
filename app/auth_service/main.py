# auth_service/main.py
from fastapi import FastAPI
from routers import users
from database.mongodb import connect_to_mongo, close_mongo_connection
from fastapi.middleware.cors import CORSMiddleware  
from routers import users, health

app = FastAPI(
    title="Authentication Service",
    description="Handles user registration and authentication.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001","http://127.0.0.1:8001"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#Come back to this


app.add_event_handler("startup", connect_to_mongo)
app.add_event_handler("shutdown", close_mongo_connection)

app.include_router(users.router, tags=["Users"], prefix="/users")
app.include_router(health.router, tags=["Health"])