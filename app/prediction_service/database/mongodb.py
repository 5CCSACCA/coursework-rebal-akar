# prediction_service/database/mongodb.py
import motor.motor_asyncio
import logging
from core.config import settings

logger = logging.getLogger("prediction_service.database.mongodb")

class MongoDB:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db: motor.motor_asyncio.AsyncIOMotorDatabase = None

mongodb = MongoDB()

async def connect_to_mongo():
    try:
        mongodb.client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URL)
        mongodb.db = mongodb.client[settings.DATABASE_NAME]


        # Create index for predictions collection on user_id
        await mongodb.db.predictions.create_index("user_id")
        await mongodb.db.predictions.create_index([("user_id", 1), ("created_at", 1)])
        
        logger.info("Connected to MongoDB and indexes created successfully.")
    except Exception as e:
        logger.exception(f"Failed to connect to MongoDB: {e}")
        raise e

async def close_mongo_connection():
    mongodb.client.close()
    logger.info("Closed connection with MongoDB.")