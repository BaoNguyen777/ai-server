import io
import os
import re
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
API_KEY = os.getenv("AI_API_KEY", "")
CONF = float(os.getenv("YOLO_CONF", "0.25"))
IMG_SIZE = int(os.getenv("YOLO_IMGSZ", "960"))
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "15"))

model: YOLO | None = None
ocr = None


def auth(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, ocr
    model = YOLO(MODEL_PATH)
    if PaddleOCR is not None:
        # PaddleOCR v3 API. Vietnamese is supported by PP-OCRv3.
        try:
            ocr = PaddleOCR(
                lang="vi",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as exc:
            print(f"[WARN] PaddleOCR could not start: {exc}")
            ocr = None
    else:
        print("[WARN] PaddleOCR is not installed. Detection will still work.")
    yield


app = FastAPI(title="Vietnam Plate AI Server", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_image(data: bytes) -> np.ndarray:
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_FILE_MB} MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return image


def normalize_text(text: str) -> str:
    text = text.upper()
    replacements = {
        "O": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
    }
    return re.sub(r"[^A-Z0-9]", "", text).translate(str.maketrans(replacements))


def plate_candidates(texts: list[str]) -> list[str]:
    joined = "".join(texts).upper()
    joined = re.sub(r"[^A-Z0-9]", "", joined)

    # Vietnamese plate shapes commonly begin with 2 digits + 1-2 letters,
    # followed by 4-5 digits. This is a post-processing heuristic, not a legal validator.
    patterns = [
        r"\d{2}[A-Z]{1,2}\d{4,5}",
        r"\d{2}[A-Z]{1,2}\d{3}[A-Z]\d{1,2}",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, joined))

    return list(dict.fromkeys(candidates))


def format_plate(raw: str) -> str:
    raw = normalize_text(raw)
    if len(raw) < 7:
        return raw

    # 43A12345 -> 43A-123.45
    if re.fullmatch(r"\d{2}[A-Z]{1,2}\d{5}", raw):
        return f"{raw[:-5]}-{raw[-5:-2]}.{raw[-2:]}"

    # 43A1234 -> 43A-1234
    if re.fullmatch(r"\d{2}[A-Z]{1,2}\d{4}", raw):
        return f"{raw[:-4]}-{raw[-4:]}"

    return raw


def run_ocr(image: np.ndarray) -> tuple[list[str], list[float]]:
    if ocr is None:
        return [], []

    result = ocr.predict(image)
    texts: list[str] = []
    scores: list[float] = []

    for item in result:
        data: Any = getattr(item, "json", None)
        if callable(data):
            data = data()
        if data is None:
            try:
                data = item["res"]
            except Exception:
                data = None

        if isinstance(data, dict) and "res" in data:
            data = data["res"]

        if isinstance(data, dict):
            for text, score in zip(
                data.get("rec_texts", []),
                data.get("rec_scores", []),
            ):
                if text:
                    texts.append(str(text))
                    scores.append(float(score))

    return texts, scores


def crop_plate(image: np.ndarray, box: list[float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    pad_x = int((x2 - x1) * 0.08)
    pad_y = int((y2 - y1) * 0.12)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image
    return crop


def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    scale = 3
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.detailEnhance(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), sigma_s=10, sigma_r=0.15)
    return gray


def extract_cccd(texts: list[str]) -> tuple[str, float]:
    candidates: list[tuple[str, float]] = []
    for text in texts:
        digits = re.sub(r"\D", "", text)
        if len(digits) == 12:
            candidates.append((digits, 0.9))
        elif len(digits) > 12:
            for i in range(len(digits) - 11):
                part = digits[i : i + 12]
                candidates.append((part, 0.75))

    if not candidates:
        return "", 0
    return max(candidates, key=lambda x: x[1])


def detect(image: np.ndarray) -> dict[str, Any]:
    assert model is not None
    results = model.predict(image, conf=CONF, imgsz=IMG_SIZE, verbose=False)
    result = results[0]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        # Still run whole-image OCR so a CCCD photo can return its 12-digit ID.
        texts, scores = run_ocr(image)
        cccd, cccd_conf = extract_cccd(texts)
        return {
            "licensePlate": "",
            "cccd": cccd,
            "confidence": 0,
            "plateConfidence": 0,
            "cccdConfidence": cccd_conf,
            "detections": [],
        }

    confs = boxes.conf.cpu().numpy().tolist()
    xyxy = boxes.xyxy.cpu().numpy().tolist()
    best_index = int(np.argmax(confs))
    plate_crop = crop_plate(image, xyxy[best_index])

    texts, scores = run_ocr(preprocess_plate(plate_crop))
    candidates = plate_candidates(texts)
    raw_plate = candidates[0] if candidates else normalize_text("".join(texts))
    license_plate = format_plate(raw_plate)

    # Whole-image OCR is used only for the 12-digit CCCD field.
    all_texts, all_scores = run_ocr(image)
    cccd, cccd_conf = extract_cccd(all_texts)

    return {
        "licensePlate": license_plate,
        "cccd": cccd,
        "confidence": float(confs[best_index]),
        "plateConfidence": float(np.mean(scores)) if scores else 0,
        "cccdConfidence": cccd_conf,
        "detections": [
            {
                "box": [round(v, 2) for v in box],
                "confidence": round(float(conf), 4),
                "class": str(result.names.get(int(cls), cls)) if hasattr(result, "names") else str(cls),
            }
            for box, conf, cls in zip(
                xyxy,
                confs,
                boxes.cls.cpu().numpy().tolist(),
            )
        ],
    }


@app.get("/")
def root():
    return {
        "service": "Vietnam Plate AI Server",
        "status": "ok",
        "model": MODEL_PATH,
        "ocr": ocr is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "ocr_loaded": ocr is not None,
    }


@app.post("/recognize", dependencies=[Depends(auth)])
async def recognize(file: UploadFile = File(...)):
    data = await file.read()
    image = read_image(data)
    result = detect(image)
    return result


@app.post("/recognize/batch", dependencies=[Depends(auth)])
async def recognize_batch(files: list[UploadFile] = File(...)):
    output = []
    for file in files:
        try:
            data = await file.read()
            image = read_image(data)
            result = detect(image)
            output.append({"filename": file.filename, **result})
        except HTTPException as exc:
            output.append({"filename": file.filename, "error": exc.detail})
        except Exception as exc:
            output.append({"filename": file.filename, "error": str(exc)})
    return {"success": True, "results": output}
