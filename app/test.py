import motor.motor_asyncio
import asyncio

# Replace these with your MongoDB connection details
MONGODB_URL="mongodb+srv://k23060616:1496@cluster0.93asi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DATABASE_NAME = "hatespeech_db"

async def test_mongo_connection():
    try:
        # Create a MongoDB client
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGODB_URL)

        # Test the connection
        db = client[DATABASE_NAME]
        result = await db.command("ping")
        print("MongoDB connection successful:", result)
    except Exception as e:
        print("Failed to connect to MongoDB:", e)

# Run the connection test
asyncio.run(test_mongo_connection())
