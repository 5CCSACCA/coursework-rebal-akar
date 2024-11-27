# auth_service/routers/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta, datetime
from core.security import verify_password, get_password_hash
from core.config import settings
from auth.auth import create_access_token
from schemas.user import UserCreate, UserOut, Token
from database.mongodb import mongodb

router = APIRouter()

@router.post("/register", response_model=UserOut)
async def register(user: UserCreate):
    existing_user = await mongodb.db.users.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username or email already registered")
        print(f"Registering user '{user.username}' with hashed password: {hashed_password}")
    hashed_password = get_password_hash(user.password)
    user_dict = user.dict()
    user_dict["hashed_password"] = hashed_password
    user_dict.pop("password")
    user_dict["created_at"] = datetime.utcnow()
    user_dict["updated_at"] = datetime.utcnow()
    result = await mongodb.db.users.insert_one(user_dict)
    user_out = UserOut(**user_dict, id=str(result.inserted_id))
    return user_out

# auth_service/routers/users.py
from datetime import datetime, timedelta

MAX_FAILED_ATTEMPTS = 10
LOCKOUT_DURATION = timedelta(minutes=15)

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await mongodb.db.users.find_one({"username": form_data.username})
    print(f"Received username: {form_data.username!r} (type: {type(form_data.username)})")
    if not user:
        print(f"Login failed: User '{form_data.username}' not found.")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    # Check if account is locked
    if user.get("account_locked_until") and user["account_locked_until"] > datetime.utcnow():
        raise HTTPException(
            status_code=403,
            detail=f"Account locked until {user['account_locked_until']}"
        )
    
    print(f"Attempting login for user '{form_data.username}' with hashed password: {user['hashed_password']}")
    if not verify_password(form_data.password, user["hashed_password"]):
        print(f"Login failed: Incorrect password for user '{form_data.username}'.")
        # Increment failed login attempts

        failed_attempts = user.get("failed_login_attempts", 0) + 1
        update_data = {"failed_login_attempts": failed_attempts}

        if failed_attempts >= MAX_FAILED_ATTEMPTS:
            update_data["account_locked_until"] = datetime.utcnow() + LOCKOUT_DURATION
            update_data["failed_login_attempts"] = 0  # Reset after lockout

        await mongodb.db.users.update_one({"_id": user["_id"]}, {"$set": update_data})
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    else:
        # Reset failed login attempts on successful login
        print(f"Login successful for user '{form_data.username}'.")
        await mongodb.db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"failed_login_attempts": 0, "account_locked_until": None}}
        )

        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"]}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}