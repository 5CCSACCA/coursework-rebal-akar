"""
Configuration Setting

Defines configuration setting for Authentication Service

"""

from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    MONGODB_URL: str
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    DATABASE_NAME: str = "hatespeech_db"

    class Config:
        env_file = ".env"

settings = Settings()

