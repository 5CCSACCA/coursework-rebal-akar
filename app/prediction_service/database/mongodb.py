# prediction_service/database/mongodb.py
import motor.motor_asyncio
from core.config import settings

class MongoDB:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db: motor.motor_asyncio.AsyncIOMotorDatabase = None

mongodb = MongoDB()

async def connect_to_mongo():
    mongodb.client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URL)
    mongodb.db = mongodb.client[settings.DATABASE_NAME]


    # Create index for predictions collection on user_id
    await mongodb.db.predictions.create_index("user_id")
    await mongodb.db.predictions.create_index([("user_id", 1), ("created_at", 1)])
    
    print("Connected to MongoDB")

async def close_mongo_connection():
    mongodb.client.close()
    print("Closed connection with MongoDB")
