# auth_service/main.py
import sys
import logging
import time
import json
from pathlib import Path
from fastapi import FastAPI, Request,Response
from fastapi.responses import RedirectResponse
from routers import users,health
from database.mongodb import connect_to_mongo, close_mongo_connection
from fastapi.middleware.cors import CORSMiddleware  
from routers import users, health
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
sys.path.append(str(Path(__file__).parent))

app = FastAPI(
    title="Authentication Service",
    description="Handles user registration and authentication.",
    version="1.0.0",
    openapi_url="/auth/openapi.json",
    docs_url="/auth/docs",
)

@app.get("/auth")
async def root():
    return {"message": "Welcome to the Authentication Service, proceed to http://localhost/auth/docs"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configure Logging
logger = logging.getLogger("auth_service")
logger.setLevel(logging.INFO)

# Create handler for stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# Define log format (JSON)
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

formatter = JsonFormatter()
handler.setFormatter(formatter)
logger.addHandler(handler)

REQUEST_COUNT = Counter('auth_service_request_count', 'Total HTTP requests to Auth Service')
REQUEST_LATENCY = Histogram('auth_service_request_latency_seconds', 'Latency of HTTP requests to Auth Service')
EXCEPTION_COUNT = Counter('auth_service_exception_count', 'Total exceptions in Auth Service')

@app.middleware("http")
async def log_requests(request: Request, call_next):
    REQUEST_COUNT.inc()
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    start_time = time.time()
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code} for {request.method} {request.url}")
        return response
    except Exception as e:
        EXCEPTION_COUNT.inc()
        logger.exception(f"Exception occurred during request processing: {e}")
        raise e




@app.get("/auth/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)



# Event handlers for startup and shutdown
@app.on_event("startup")
async def startup_event():
    await connect_to_mongo()
    logger.info("Connected to MongoDB")

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()
    logger.info("Closed connection with MongoDB")


app.include_router(users.router, tags=["Users"], prefix="/auth/users")
app.include_router(health.router, tags=["Health"]) 

