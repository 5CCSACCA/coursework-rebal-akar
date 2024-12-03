"""
Authentication utilities

Provides functionalities for creating JWT access token and handle authentication dependencies within Authentication Service.
"""



from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from core.config import settings
from schemas.user import TokenData
from database.mongodb import mongodb

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/users/login")

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create JWT access token
    Args:
        data(dict): Data to encod token
    Returns:
        str: Encoded JWT token as a string
    """
    
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta 
    else:
       datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

