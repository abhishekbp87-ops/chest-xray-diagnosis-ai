# backend/main.py
"""
FastAPI app for Chest X-ray inference (robust imports + safe fallback).
Start server from project root:
    uvicorn backend.main:app --reload
"""

import os
import sys
import time
import uuid
import io
import random
from contextlib import asynccontextmanager
from typing import Dict, Optional
from pydantic import BaseModel, Field

import torch
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image

import structlog

# ----------------------------
# Make sure project root is importable: add project root to sys.path
# backend/main.py -> project root is parent of backend/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Try to import your src modules. If they fail, provide safe mocks so server still starts.
USE_MOCK = False
try:
    from src.data_processor import preprocess_image        # your preprocessing function
    from src.medical_model import load_model               # your model loader
    from src.infer import predict_image                    # your inference wrapper
except Exception as e:
    # If import fails, use safe mock implementations and log a warning
    USE_MOCK = True
    print("WARNING: Could not import src.* modules. Running in MOCK mode. Error:", e)

    def preprocess_image(img: Image.Image):
        # basic placeholder: convert to RGB and return as-is
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def load_model(path: str, num_classes: int = 2, device=None):
        # placeholder: return None (indicates no real model loaded)
        return None

    def predict_image(model, processed_image) -> Dict:
        # placeholder prediction
        return {"label": "Normal", "confidence": round(random.uniform(0.6, 0.98), 3)}

# ----------------------------
# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# ----------------------------
# Pydantic models
class PredictionResponse(BaseModel):
    prediction_id: str
    label: str
    confidence: float
    processing_time: float
    model_version: str
    metadata: Dict = {}

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    model_loaded: bool
    gpu_available: bool

# ----------------------------
# globals
model_cache = {"model": None, "loaded_at": None}
security = HTTPBearer(auto_error=False)

# ----------------------------
# Lifespan: load model at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("app_starting")
    try:
        await _load_model_on_startup()
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
    yield
    await _cleanup()

async def _load_model_on_startup():
    if USE_MOCK:
        logger.warn("Using MOCK mode - skipping model load")
        model_cache["model"] = None
        model_cache["loaded_at"] = time.time()
        return

    logger.info("Loading ML model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # adjust path if your models are stored somewhere else
    model_path = os.path.join(ROOT_DIR, "models", "best_model.pth")
    model = load_model(model_path, num_classes=2, device=device)
    model_cache["model"] = model
    model_cache["loaded_at"] = time.time()
    logger.info("model_loaded", device=str(device), path=model_path)

async def _cleanup():
    logger.info("cleanup_resources")
    model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ----------------------------
# App creation
app = FastAPI(
    title="Chest X-ray AI Diagnostics",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

# Static + templates (use backend/ absolute paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # backend/
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ----------------------------
# Helpers
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if credentials:
        return {"user_id": "demo_user"}
    return None

def validate_image_file(file: UploadFile) -> None:
    allowed_types = ["image/jpeg", "image/jpg", "image/png"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type")
    # file.size may not be present; don't rely solely on it

# ----------------------------
# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    rid = str(uuid.uuid4())
    ts = time.time()
    logger.info("request_start", id=rid, method=request.method, path=request.url.path)
    try:
        resp = await call_next(request)
        dur = time.time() - ts
        resp.headers["X-Request-ID"] = rid
        resp.headers["X-Process-Time"] = f"{dur:.4f}"
        logger.info("request_end", id=rid, status=resp.status_code, duration=round(dur, 4))
        return resp
    except Exception as e:
        logger.error("request_error", id=rid, error=str(e))
        raise

# ----------------------------
# Routes
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        version="1.0.0",
        model_loaded=model_cache["model"] is not None,
        gpu_available=torch.cuda.is_available(),
    )

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_endpoint(background_tasks: BackgroundTasks,
                           file: UploadFile = File(...),
                           user = Depends(get_current_user)):
    start = time.time()
    prediction_id = str(uuid.uuid4())

    try:
        validate_image_file(file)
        contents = await file.read()

        pil_img = Image.open(io.BytesIO(contents))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        processed = preprocess_image(pil_img)

        # if no model loaded and mock mode enabled, predict_image will return mock label
        prediction = predict_image(model_cache["model"], processed)

        processing_time = time.time() - start

        # background logging (non-blocking)
        background_tasks.add_task(
            lambda pid, fn, pred, ptime: logger.info("analytics", pid=pid, filename=fn, pred=pred, ptime=ptime),
            prediction_id, getattr(file, "filename", "unknown"), prediction, processing_time
        )

        return PredictionResponse(
            prediction_id=prediction_id,
            label=prediction.get("label", "Unknown"),
            confidence=prediction.get("confidence", 0.0),
            processing_time=processing_time,
            model_version="mock" if USE_MOCK else "resnet-v1",
            metadata={
                "image_size": pil_img.size,
                "file_size": len(contents),
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("prediction_error", error=str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")

# ----------------------------
# small utility endpoints
@app.get("/api/v1/stats")
async def stats():
    gpu = torch.cuda.is_available()
    gpu_mem = {}
    if gpu:
        gpu_mem = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated(),
        }
    return {
        "model_loaded": model_cache["model"] is not None,
        "model_loaded_at": model_cache.get("loaded_at"),
        "gpu_available": gpu,
        "gpu_memory": gpu_mem,
    }

# ----------------------------
# Error handlers
@app.exception_handler(404)
async def handle_404(request: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Not found"})

@app.exception_handler(500)
async def handle_500(request: Request, exc):
    logger.error("internal_error", error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
