"""
MongoDB Connection

This module manages the connection to MongoDB database uses Motor for asynchronous operations.

"""


import motor.motor_asyncio
from core.config import settings
import logging

logger = logging.getLogger("auth_service.database.mongodb")

class MongoDB:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db: motor.motor_asyncio.AsyncIOMotorDatabase = None

mongodb = MongoDB()

async def connect_to_mongo():
    try:
        mongodb.client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URL)
        mongodb.db = mongodb.client[settings.DATABASE_NAME]

        # Create indexes for user collection
        await mongodb.db.users.create_index("username", unique=True)
        await mongodb.db.users.create_index("email", unique=True)
        await mongodb.db.users.create_index("user_id", unique=True)  
        
        # Create index for predictions collection
        await mongodb.db.predictions.create_index("user_id")

        logger.info("Connected to MongoDB and indexes created successfully.")
    except Exception as e:
        logger.exception(f"Failed to connect to MongoDB: {e}")
        raise e

async def close_mongo_connection():
    try:
        mongodb.client.close()
        logger.info("Closed connection with MongoDB.")
    except Exception as e:
        logger.exception(f"Error closing MongoDB connection: {e}")
