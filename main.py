import os
import re
import traceback
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("PADDLEX_HOME", "/tmp/paddlex")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

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
ocr_init_error: str | None = None


def auth(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(title="Vietnam Plate AI Server", version="1.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    global model
    model = YOLO(MODEL_PATH)
    print(f"[STARTUP] YOLO loaded: {MODEL_PATH}", flush=True)


def get_ocr():
    global ocr, ocr_init_error
    if ocr is not None:
        return ocr
    if PaddleOCR is None:
        ocr_init_error = "PaddleOCR is not installed"
        print("[OCR] PaddleOCR import is unavailable", flush=True)
        return None

    try:
        print("[OCR] Initializing lightweight PP-OCRv5 mobile models...", flush=True)
        print("[OCR] MKL-DNN disabled for CPU compatibility", flush=True)
        ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            device="cpu",
        )
        print("[OCR] PaddleOCR ready", flush=True)
    except Exception as exc:
        ocr_init_error = f"{type(exc).__name__}: {exc}"
        print(f"[OCR] initialization failed: {ocr_init_error}", flush=True)
        traceback.print_exc()
        ocr = None
    return ocr


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
    replacements = {"O": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8"}
    return re.sub(r"[^A-Z0-9]", "", text).translate(str.maketrans(replacements))


def plate_candidates(texts: list[str]) -> list[str]:
    joined = re.sub(r"[^A-Z0-9]", "", "".join(texts).upper())
    patterns = [r"\d{2}[A-Z]{1,2}\d{4,5}", r"\d{2}[A-Z]{1,2}\d{3}[A-Z]\d{1,2}"]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, joined))
    return list(dict.fromkeys(candidates))


def format_plate(raw: str) -> str:
    raw = normalize_text(raw)
    if len(raw) < 7:
        return raw
    if re.fullmatch(r"\d{2}[A-Z]{1,2}\d{5}", raw):
        return f"{raw[:-5]}-{raw[-5:-2]}.{raw[-2:]}"
    if re.fullmatch(r"\d{2}[A-Z]{1,2}\d{4}", raw):
        return f"{raw[:-4]}-{raw[-4:]}"
    return raw


def run_ocr(image: np.ndarray) -> tuple[list[str], list[float]]:
    pipeline = get_ocr()
    if pipeline is None:
        return [], []
    print(f"[OCR] predict start shape={image.shape}", flush=True)
    result = pipeline.predict(image)
    print("[OCR] predict complete", flush=True)
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
            for text, score in zip(data.get("rec_texts", []), data.get("rec_scores", [])):
                if text:
                    texts.append(str(text))
                    scores.append(float(score))
    print(f"[OCR] texts={texts}", flush=True)
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
    return crop if crop.size else image


def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.detailEnhance(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), sigma_s=10, sigma_r=0.15)


def extract_cccd(texts: list[str]) -> tuple[str, float]:
    candidates: list[tuple[str, float]] = []
    for text in texts:
        digits = re.sub(r"\D", "", text)
        if len(digits) == 12:
            candidates.append((digits, 0.9))
        elif len(digits) > 12:
            for i in range(len(digits) - 11):
                candidates.append((digits[i : i + 12], 0.75))
    return max(candidates, key=lambda x: x[1]) if candidates else ("", 0)


def detect(image: np.ndarray) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="YOLO model is not loaded")

    print("[DETECT] YOLO predict start", flush=True)
    results = model.predict(image, conf=CONF, imgsz=IMG_SIZE, verbose=False)
    print("[DETECT] YOLO predict complete", flush=True)
    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("[DETECT] no YOLO boxes; running full-image OCR", flush=True)
        texts, _ = run_ocr(image)
        cccd, cccd_conf = extract_cccd(texts)
        return {"licensePlate": "", "cccd": cccd, "confidence": 0, "plateConfidence": 0, "cccdConfidence": cccd_conf, "detections": []}

    confs = boxes.conf.cpu().numpy().tolist()
    xyxy = boxes.xyxy.cpu().numpy().tolist()
    best_index = int(np.argmax(confs))
    print(f"[DETECT] boxes={len(xyxy)} best_conf={confs[best_index]:.4f}", flush=True)

    texts, scores = run_ocr(preprocess_plate(crop_plate(image, xyxy[best_index])))
    candidates = plate_candidates(texts)
    raw_plate = candidates[0] if candidates else normalize_text("".join(texts))
    license_plate = format_plate(raw_plate)
    print(f"[DETECT] plate={license_plate} candidates={candidates}", flush=True)

    all_texts, _ = run_ocr(image)
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
            for box, conf, cls in zip(xyxy, confs, boxes.cls.cpu().numpy().tolist())
        ],
    }


@app.get("/")
def root():
    return {"service": "Vietnam Plate AI Server", "status": "ok", "model": MODEL_PATH, "model_loaded": model is not None, "ocr_loaded": ocr is not None}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "ocr_loaded": ocr is not None, "ocr_init_error": ocr_init_error}


@app.post("/recognize", dependencies=[Depends(auth)])
async def recognize(file: UploadFile = File(...)):
    print(f"[RECOGNIZE] request received filename={file.filename} content_type={file.content_type}", flush=True)
    try:
        data = await file.read()
        print(f"[RECOGNIZE] file read bytes={len(data)}", flush=True)
        image = read_image(data)
        print(f"[RECOGNIZE] image decoded shape={image.shape}", flush=True)
        result = detect(image)
        print("[RECOGNIZE] success", flush=True)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[RECOGNIZE] UNHANDLED {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Recognition failed: {type(exc).__name__}: {exc}") from exc


@app.post("/recognize/batch", dependencies=[Depends(auth)])
async def recognize_batch(files: list[UploadFile] = File(...)):
    output = []
    for file in files:
        try:
            data = await file.read()
            image = read_image(data)
            output.append({"filename": file.filename, **detect(image)})
        except HTTPException as exc:
            output.append({"filename": file.filename, "error": exc.detail})
        except Exception as exc:
            output.append({"filename": file.filename, "error": str(exc)})
    return {"success": True, "results": output}
