# prediction_service/main.py
import logging
import sys
import time
import json
from pathlib import Path
from fastapi import FastAPI, Request, Response
from routers import predict,health
from database.mongodb import connect_to_mongo, close_mongo_connection
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram
sys.path.append(str(Path(__file__).parent))


app = FastAPI(
    title="Prediction Service",
    description="Handles predictions using the machine learning model.",
    version="1.0.0",
    openapi_url="/predict/openapi.json",
    docs_url="/predict/docs",
    redoc_url=None
)

@app.get("/predict")
async def root():
    return {"message": "Welcome to the Authentication Service, please proceed to http://localhost/predict/docs"}

# Configure Logging
logger = logging.getLogger("prediction_service")
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

# Define Prometheus metrics (Optional, remove if not using Prometheus yet)
REQUEST_COUNT = Counter('prediction_service_request_count', 'Total HTTP requests to Prediction Service')
REQUEST_LATENCY = Histogram('prediction_service_request_latency_seconds', 'Latency of HTTP requests to Prediction Service')
EXCEPTION_COUNT = Counter('prediction_service_exception_count', 'Total exceptions in Prediction Service')

# Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    REQUEST_COUNT.inc()
    logger.info(f"Incoming request: {request.method} {request.url}")
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        REQUEST_LATENCY.observe(process_time)
        logger.info(f"Response status: {response.status_code} for {request.method} {request.url} in {process_time:.4f}s")
        return response
    except Exception as e:
        EXCEPTION_COUNT.inc()
        logger.exception(f"Exception occurred during request processing: {e}")
        raise e
    

@app.get("/predict/metrics")
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


app.include_router(predict.router, tags=["Predictions"], prefix="/predict")
app.include_router(health.router, tags=["Health"])

