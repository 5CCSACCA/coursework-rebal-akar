"""
User Schemas

This module defines the Pydantic models for user data structures used in the Authentication Service.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, validator, Field
import re

class UserBase(BaseModel):
    username: str = Field(..., example="johndoe")
    email: EmailStr = Field(..., example="johndoe@example.com")

class UserCreate(UserBase):
    password: str =Field(..., 
                          example="Password123!",
                          description="Password must be at least 8 characters long and include uppercase letters, lowercase letters, digits, and special characters.")

    @validator('password')
    def password_complexity(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v


class UserInDB(UserBase):
    hashed_password: str
    failed_login_attempts: int = 0
    account_locked_until: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class UserOut(UserBase):
    id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class Token(BaseModel):
    access_token: str = Field(..., example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    token_type: str = Field(..., example="bearer", description="Type of the token, typically 'bearer'.")

class TokenData(BaseModel):
    username: Optional[str] = None 

