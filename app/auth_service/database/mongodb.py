# db/mongodb.py
import motor.motor_asyncio
from core.config import settings

class MongoDB:
    client: motor.motor_asyncio.AsyncIOMotorClient = None
    db: motor.motor_asyncio.AsyncIOMotorDatabase = None

mongodb = MongoDB()

async def connect_to_mongo():
    mongodb.client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URL)
    mongodb.db = mongodb.client[settings.DATABASE_NAME]

    # Create indexes for user collection
    await mongodb.db.users.create_index("username", unique=True)
    await mongodb.db.users.create_index("email", unique=True)

    # Create index for predictions collection on user_id
    await mongodb.db.predictions.create_index("user_id")

    print("Connected to MongoDB")

async def close_mongo_connection():
    mongodb.client.close()
    print("Closed connection with MongoDB")


