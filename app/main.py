from fastapi import FastAPI
from routers import users, predict

app = FastAPI()

app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(predict.router, tags=["predict"])
