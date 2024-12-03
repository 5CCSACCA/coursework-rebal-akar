# auth_service/routers/health.py
from fastapi import APIRouter
from database.mongodb import mongodb
import logging

router = APIRouter()
logger = logging.getLogger("auth_service.routers.health")

@router.get("/health", summary="Health Check", description="Returns the health status of the service.")
async def health():
    try:
        # Simple health check: ping the database
        await mongodb.db.command("ping")
        logger.info("Health check passed: Database is reachable.")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "details": str(e)}
