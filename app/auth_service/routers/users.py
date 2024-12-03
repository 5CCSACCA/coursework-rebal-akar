# auth_service/routers/users.py
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta, datetime
from core.security import verify_password, get_password_hash
from core.config import settings
from auth.auth import create_access_token
from schemas.user import UserCreate, UserOut, Token
from database.mongodb import mongodb
from datetime import datetime, timedelta

router = APIRouter()
logger = logging.getLogger("auth_service.routers.users")


MAX_FAILED_ATTEMPTS = 10
LOCKOUT_DURATION = timedelta(minutes=15)


@router.post(
    "/register",
    response_model=UserOut,
    status_code=201,
    summary="Register a new user",
    description="""
    Register a new user by providing a unique username, email and a password.

    **Note:** The password must be at least 8 characters long and include an uppercase letter, lowercase letter, a digit and a special character.
    """
)
async def register(user: UserCreate):
    logger.info(f"Attempting to register user: {user.username}")
    try:
        existing_user = await mongodb.db.users.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
        if existing_user:
            logger.warning(f"Registration failed: Username or email already registered ({user.username}, {user.email})")
            raise HTTPException(status_code=400, detail="Username or email already registered")
        user_dict = user.dict()
        user_dict["hashed_password"] = hashed_password
        user_dict.pop("password")
        user_dict["created_at"] = datetime.utcnow()
        user_dict["updated_at"] = datetime.utcnow()
        
        result = await mongodb.db.users.insert_one(user_dict)
        user_out = UserOut(**user_dict, id=str(result.inserted_id))
        
        logger.info(f"User registered successfully: {user.username}")
        return user_out
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Error during user registration for {user.username}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during registration")




@router.post(
    "/login",
    response_model=Token,
    summary="User login",
    description="""
    Authenticate your user by providing a valid username and password.

    Upon successful authentication, returns a JWT access token that must be included in the `Authorization` header for protected endpoints.

    The JWT access token will be given to you, simply go to http://localhost/prediction/docs to perform predictions.

    """
)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    logger.info(f"User attemption to log in: {username}")
    try:
        user = await mongodb.db.users.find_one({"username": username})
        print(f"Received username: {form_data.username!r} (type: {type(form_data.username)})")
        if not user:
            logger.warning(f"Login failed: User '{username}' not found.")  
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        # Check if account is locked
        if user.get("account_locked_until") and user["account_locked_until"] > datetime.utcnow():
            logger.warning(f"Account locked for user '{username}' until {user['account_locked_until']}.")
            raise HTTPException(
                status_code=403,
                detail=f"Account locked until {user['account_locked_until']}"
            )
        
        if not verify_password(form_data.password, user["hashed_password"]):
            logger.warning(f"Login failed: Incorrect password for user '{username}'.")            # Increment failed login attempts

            failed_attempts = user.get("failed_login_attempts", 0) + 1
            update_data = {"failed_login_attempts": failed_attempts}

            if failed_attempts >= MAX_FAILED_ATTEMPTS:
                update_data["account_locked_until"] = datetime.utcnow() + LOCKOUT_DURATION
                update_data["failed_login_attempts"] = 0  # Reset after lockout
                logger.warning(f"User '{username}' has been locked out due to too many failed login attempts.")

            await mongodb.db.users.update_one({"_id": user["_id"]}, {"$set": update_data})
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        else:
            logger.info(f"User '{username}' logged in successfully.")            
            await mongodb.db.users.update_one(
                {"_id": user["_id"]},
                {"$set": {"failed_login_attempts": 0, "account_locked_until": None}}
            )

            access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": user["username"]}, expires_delta=access_token_expires
            )
            return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Error during login for user '{username}': {e}")
        raise HTTPException(status_code=500, detail="Internal server error during login")
