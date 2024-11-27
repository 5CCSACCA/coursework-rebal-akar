# auth_service/routers/health.py

from fastapi import APIRouter, status
from database.mongodb import mongodb

router = APIRouter()

@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    try:
        # Check database connection
        await mongodb.db.command("ping")
        return {"status": "OK"}
    except Exception as e:
        return {"status": "DOWN", "details": str(e)}
