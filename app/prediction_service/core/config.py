# prediction_service/core/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    MONGODB_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    DATABASE_NAME: str = "hatespeech_db"

    class Config:
        env_file = ".env"

settings = Settings()
