# prediction_service/routers/health.py
from fastapi import APIRouter
from database.mongodb import mongodb
import logging

router = APIRouter()
logger = logging.getLogger("prediction_service.routers.health")

@router.get("/health", summary="Health check", description="Returns the health status of the service.")
async def health_check():
    try:
        # Check database connection
        await mongodb.db.command("ping")
        logger.info("Health check passed: Database is reachable.")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "details": str(e)}
