# prediction_service/schemas/user.py
from pydantic import BaseModel
from typing import Optional

class TokenData(BaseModel):
    username: Optional[str] = None
