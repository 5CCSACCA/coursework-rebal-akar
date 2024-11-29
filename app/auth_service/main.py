# auth_service/main.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from routers import users
from database.mongodb import connect_to_mongo, close_mongo_connection
from fastapi.middleware.cors import CORSMiddleware  
#from routers import users, health

app = FastAPI(
    title="Authentication Service",
    description="Handles user registration and authentication.",
    version="1.0.0",
    openapi_url="/auth/openapi.json",
    docs_url="/auth/docs",
    redoc_url=None,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_event_handler("startup", connect_to_mongo)
app.add_event_handler("shutdown", close_mongo_connection)

app.include_router(users.router, tags=["Users"], prefix="/auth/users")

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/auth/docs")