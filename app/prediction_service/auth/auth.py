# prediction_service/auth/auth.py
import logging
from fastapi import Depends, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from core.config import settings
from schemas.user import TokenData
from database.mongodb import mongodb

logger = logging.getLogger("prediction_service.auth.auth")

# Update tokenUrl to point to the auth_service's login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="http://localhost/auth/users/login")


SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            print("Token payload does not contain 'sub'")
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as e:
        print(f"JWTError: {e}")
        raise credentials_exception
    user = await mongodb.db.users.find_one({"username": token_data.username})
    if user is None:
        logger.warning(f"User not found in database: {token_data.username}")
        raise credentials_exception
    logger.info(f"User retrieved from database: {token_data.username}")
    return user
