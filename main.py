import io
import os
import re
import logging
import threading
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import joblib
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    easyocr = None
    EASYOCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("receipt-api")

# ---------------- Settings ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "expense_classifier.pkl"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024      # 5 MB per image
MAX_IMAGE_PIXELS = 25_000_000           # blocks huge "image bombs"
MAX_SIDE = 2000                         # big photos are shrunk to this (faster OCR)
MAX_ITEMS = 100                         # for /predict-expense
MAX_ITEM_LENGTH = 200
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# Enforce Pillow pixel limit so it raises cleanly instead of unhandled bomb error
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

ALLOWED_ORIGINS = [
    o.strip() for o in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
    ).split(",") if o.strip()
]
ENABLE_DOCS = os.environ.get("ENABLE_DOCS") == "1"

# ---------------- App (must be created BEFORE any @app route) ----------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Smart Receipt & Invoice Analyzer API",
    description="Backend API powered by FastAPI, Scikit-Learn ML Classifier, and Python OCR.",
    version="1.0.0",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Reject very large requests early."""
    limit = MAX_UPLOAD_BYTES + 1_000_000 if request.url.path == "/upload-receipt-image" else 100_000
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > limit:
        return JSONResponse({"detail": "Request is too large."}, status_code=413)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------- Load model and OCR once at startup ----------------
try:
    model_pipeline = joblib.load(MODEL_PATH)
    logger.info("Loaded ML model from %s", MODEL_PATH)
except Exception:
    logger.exception("Could not load the ML model")
    model_pipeline = None

reader = None
if EASYOCR_AVAILABLE:
    try:
        logger.info("Initializing EasyOCR reader...")
        reader = easyocr.Reader(["en"], gpu=False)
        logger.info("EasyOCR reader ready")
    except Exception:
        logger.exception("Could not initialize EasyOCR reader; OCR endpoint will be unavailable")
        reader = None
else:
    logger.warning("easyocr is not installed; OCR endpoint will be unavailable")

ocr_lock = threading.Lock()


# ---------------- Schemas ----------------
class ExpensePayload(BaseModel):
    items: list[str] = Field(default_factory=list)


# ---------------- Helpers ----------------
def load_image(contents: bytes):
    """Check the file really is a normal-size JPEG/PNG/WEBP image, then open it."""
    try:
        with Image.open(io.BytesIO(contents)) as im:
            fmt, (w, h) = im.format, im.size
    except Image.DecompressionBombError:
        raise HTTPException(status_code=413, detail="Image dimensions are too large.")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail="Only JPEG, PNG or WEBP images are allowed.")
    if w * h > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=413, detail="Image dimensions are too large.")

    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def preprocess(img):
    """Resize, convert to gray and remove noise so the OCR reads better."""
    longest = max(img.shape[:2])
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif longest < 800:
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.bilateralFilter(gray, 11, 17, 17)


SKIP_LINE = re.compile(
    r"\b(total|subtotal|tax|vat|cash|change|visa|mastercard|thank you|balance|amount due)\b",
    re.IGNORECASE,
)


def clean_lines(ocr_results):
    """Keep only lines that look like item names."""
    lines = []
    for _, text, conf in ocr_results:
        text = text.strip()
        if conf < 0.3 or len(text) < 3:
            continue
        if not re.search(r"[A-Za-z]{2,}", text):
            continue
        if SKIP_LINE.search(text):
            continue
        lines.append(text[:MAX_ITEM_LENGTH])
    return lines


def extract_items_list(payload: Union[ExpensePayload, list[str]]) -> list[str]:
    if isinstance(payload, ExpensePayload):
        return payload.items
    return payload


# ---------------- Routes ----------------
@app.get("/")
def read_root():
    info = {
        "project": "Smart Receipt & Invoice Analyzer",
        "status": "Active & Running",
        "model_loaded": model_pipeline is not None,
        "ocr_available": reader is not None,
    }
    if ENABLE_DOCS:
        info["docs_url"] = "/docs"
    return info


@app.post("/upload-receipt-image")
@limiter.limit("5/minute")
def upload_receipt_image(request: Request, file: UploadFile = File(...)):
    """
    Accepts a receipt image, runs OpenCV preprocessing and EasyOCR,
    then classifies each item line using the .pkl model.
    """
    if reader is None:
        raise HTTPException(status_code=503, detail="OCR service is not available on this server.")
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="The prediction service is not ready.")

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large (maximum 5 MB).")

    gray = preprocess(load_image(contents))

    if not ocr_lock.acquire(timeout=30):
        raise HTTPException(status_code=503, detail="The server is busy. Please try again in a moment.")
    try:
        ocr_results = reader.readtext(gray)
    except Exception:
        logger.exception("OCR failed")
        raise HTTPException(status_code=500, detail="Could not read text from the image.")
    finally:
        ocr_lock.release()

    lines = clean_lines(ocr_results)
    filename = os.path.basename(file.filename or "")[:100]

    if not lines:
        return {
            "status": "success",
            "filename": filename,
            "total_items_detected": 0,
            "message": "No clear text detected. Try uploading a sharper image.",
            "breakdown": [],
        }

    try:
        predictions = model_pipeline.predict(lines)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Could not classify the items.")

    breakdown = [
        {"item_description": item, "predicted_category": str(cat)}
        for item, cat in zip(lines, predictions)
    ]
    return {
        "status": "success",
        "filename": filename,
        "total_items_detected": len(breakdown),
        "breakdown": breakdown,
    }


@app.post("/predict-expense")
@limiter.limit("30/minute")
def predict_expense(request: Request, payload: Union[ExpensePayload, list[str]]):
    """
    Accepts either {"items": [...]} or [...] and predicts spending categories
    using the trained scikit-learn pipeline.
    """
    items = extract_items_list(payload)

    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="The prediction service is not ready.")
    if not items:
        raise HTTPException(status_code=400, detail="Item list cannot be empty.")
    if len(items) > MAX_ITEMS:
        raise HTTPException(status_code=400, detail=f"Too many items (maximum {MAX_ITEMS}).")
    if any(len(item) > MAX_ITEM_LENGTH for item in items):
        raise HTTPException(status_code=400, detail=f"Each item must be at most {MAX_ITEM_LENGTH} characters.")

    try:
        predictions = model_pipeline.predict(items)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Could not classify the items.")

    results = [
        {"item_description": item, "predicted_category": str(cat)}
        for item, cat in zip(items, predictions)
    ]
    return {
        "status": "success",
        "total_items_analyzed": len(results),
        "breakdown": results,
    }