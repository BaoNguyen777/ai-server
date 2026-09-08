import os
import re
import traceback
import gc
import ctypes
from typing import Any

# Railway/container memory safety: keep native ML runtimes conservative.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("PADDLEX_HOME", "/tmp/paddlex")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PADDLE_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

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
PLATE_OCR_MAX_SIDE = int(os.getenv("PLATE_OCR_MAX_SIDE", "1280"))
PLATE_SCALE = float(os.getenv("PLATE_OCR_SCALE", "2.0"))

model: YOLO | None = None
ocr = None
ocr_init_error: str | None = None


def auth(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


app = FastAPI(title="Vietnam License Plate AI Server", version="1.6.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def trim_native_memory():
    """Ask glibc to return unused native heap pages after ML inference."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


@app.on_event("startup")
def startup():
    global model
    model = YOLO(MODEL_PATH)
    print(f"[STARTUP] YOLO loaded: {MODEL_PATH}", flush=True)
    print(f"[STARTUP] YOLO conf={CONF} imgsz={IMG_SIZE}", flush=True)
    print(f"[STARTUP] plate OCR max side={PLATE_OCR_MAX_SIDE} scale={PLATE_SCALE}", flush=True)


def get_ocr():
    """Lazy-load only the plate OCR pipeline."""
    global ocr, ocr_init_error

    if ocr is not None:
        return ocr

    if PaddleOCR is None:
        ocr_init_error = "PaddleOCR is not installed"
        print("[OCR] PaddleOCR import is unavailable", flush=True)
        return None

    try:
        print("[OCR] Initializing PP-OCRv5 mobile English/alphanumeric pipeline...", flush=True)
        print("[OCR] CPU memory-safe plate-only mode", flush=True)
        ocr = PaddleOCR(
            lang="en",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            device="cpu",
            text_det_limit_side_len=640,
            text_det_limit_type="max",
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


def resize_max_side(image: np.ndarray, max_side: int, label: str) -> np.ndarray:
    """Downscale an image before OCR to prevent CPU/RAM spikes."""
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    current_max = max(h, w)
    if current_max <= max_side:
        return image

    scale = max_side / current_max
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    print(f"[OCR] resizing {label}: {w}x{h} -> {new_w}x{new_h}", flush=True)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def clean_alnum(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def correct_confusion_characters(raw: str) -> str:
    chars = list(clean_alnum(raw))
    n = len(chars)
    if n < 5:
        return "".join(chars)

    char_to_digit = {
        "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1",
        "S": "5", "B": "8", "G": "6", "Z": "2",
    }

    for i in range(min(2, n)):
        if chars[i].isalpha() and chars[i] in char_to_digit:
            chars[i] = char_to_digit[chars[i]]

    for i in range(n - 1, 2, -1):
        if chars[i].isalpha():
            if chars[i] in char_to_digit:
                chars[i] = char_to_digit[chars[i]]
            else:
                break

    return "".join(chars)


def plate_candidates(text: str) -> list[str]:
    """Generate standard one-line plate candidates from OCR text."""
    raw = correct_confusion_characters(text)
    patterns = [
        r"\d{2}[A-Z]{1,2}\d{4,5}",
        r"\d{2}[A-Z]{1,2}\d[A-Z]\d{1,2}",
    ]

    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, raw))

    original = clean_alnum(text)
    for pattern in patterns:
        candidates.extend(re.findall(pattern, original))

    return list(dict.fromkeys(candidates))


def motorcycle_candidates(rows: list[str]) -> list[str]:
    """Parse Vietnamese motorcycle plates that are printed on two rows."""
    if len(rows) < 2:
        return []

    normalized = [clean_alnum(row) for row in rows if clean_alnum(row)]
    if len(normalized) < 2:
        return []

    candidates: list[str] = []

    for i in range(len(normalized) - 1):
        top = correct_confusion_characters(normalized[i])
        bottom = correct_confusion_characters(normalized[i + 1])

        if re.fullmatch(r"\d{2}[A-Z]\d", top) and re.fullmatch(r"\d{4,5}", bottom):
            candidates.append(top + bottom)
            continue

        if re.fullmatch(r"\d{2}", top) and re.fullmatch(r"[A-Z]\d", bottom):
            if i + 2 < len(normalized):
                number = correct_confusion_characters(normalized[i + 2])
                if re.fullmatch(r"\d{4,5}", number):
                    candidates.append(top + bottom + number)

    return list(dict.fromkeys(candidates))


def format_plate(raw: str, motorcycle: bool = False) -> str:
    raw = correct_confusion_characters(raw)

    if motorcycle:
        match = re.fullmatch(r"(\d{2})([A-Z])(\d)(\d{4,5})", raw)
        if not match:
            return ""
        city_code, letter, series_number, numbers = match.groups()
        if len(numbers) == 5:
            numbers = f"{numbers[:3]}.{numbers[3:]}"
        return f"{city_code}-{letter}{series_number} {numbers}"

    match = re.fullmatch(r"(\d{2})([A-Z]{1,2})(\d{4,5})", raw)
    if not match:
        return ""

    city_code, series, numbers = match.groups()
    if len(numbers) == 5:
        numbers = f"{numbers[:3]}.{numbers[3:]}"
    return f"{city_code}{series}-{numbers}"


def _json_from_result(item: Any) -> Any:
    data = getattr(item, "json", None)
    if callable(data):
        data = data()

    if data is None:
        try:
            data = item["res"]
        except Exception:
            data = None

    if isinstance(data, dict) and "res" in data:
        data = data["res"]

    return data


def run_ocr(image: np.ndarray, label: str = "plate") -> list[dict[str, Any]]:
    pipeline = get_ocr()
    if pipeline is None or image is None or image.size == 0:
        return []

    image_for_ocr = resize_max_side(image, PLATE_OCR_MAX_SIDE, label)
    created_resized = image_for_ocr is not image
    result = None

    try:
        print(f"[OCR] predict start shape={image_for_ocr.shape} label={label}", flush=True)
        result = pipeline.predict(image_for_ocr)
        print("[OCR] predict complete", flush=True)

        items: list[dict[str, Any]] = []
        for item in result:
            data = _json_from_result(item)
            if not isinstance(data, dict):
                continue

            rec_texts = data.get("rec_texts", []) or []
            rec_scores = data.get("rec_scores", []) or []
            rec_polys = data.get("rec_polys", []) or data.get("dt_polys", []) or []

            for i, text in enumerate(rec_texts):
                if not text:
                    continue

                try:
                    score = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                except Exception:
                    score = 0.0

                box = rec_polys[i] if i < len(rec_polys) else None
                if box is None:
                    items.append({"cx": 0.0, "cy": float(i), "height": 1.0, "text": str(text), "confidence": score})
                    continue

                try:
                    points = np.asarray(box, dtype=float).reshape(-1, 2)
                    xs = points[:, 0]
                    ys = points[:, 1]
                    cx = float(xs.mean())
                    cy = float(ys.mean())
                    height = max(float(ys.max() - ys.min()), 1.0)
                except Exception:
                    cx = 0.0
                    cy = float(i)
                    height = 1.0

                items.append({"cx": cx, "cy": cy, "height": height, "text": str(text), "confidence": score})

        print("[OCR] items=" + str([{"text": x["text"], "confidence": round(x["confidence"], 3), "cx": round(x["cx"], 1), "cy": round(x["cy"], 1)} for x in items]), flush=True)
        return items
    finally:
        result = None
        if created_resized:
            del image_for_ocr
        gc.collect()
        trim_native_memory()


def group_ocr_rows(items: list[dict[str, Any]]) -> tuple[list[str], float]:
    if not items:
        return [], 0.0

    items = sorted(items, key=lambda x: x["cy"])
    rows: list[list[dict[str, Any]]] = []
    current_row = [items[0]]
    avg_height = max(float(items[0]["height"]), 1.0)

    for item in items[1:]:
        threshold = max(avg_height * 0.75, 4.0)
        if abs(float(item["cy"]) - float(current_row[-1]["cy"])) < threshold:
            current_row.append(item)
            avg_height = sum(float(x["height"]) for x in current_row) / len(current_row)
        else:
            rows.append(current_row)
            current_row = [item]
            avg_height = max(float(item["height"]), 1.0)

    rows.append(current_row)

    row_texts: list[str] = []
    confidences: list[float] = []
    for row in rows:
        row.sort(key=lambda x: x["cx"])
        text = "".join(str(x["text"]) for x in row)
        cleaned = clean_alnum(text)
        if cleaned:
            row_texts.append(cleaned)
        confidences.extend(float(x["confidence"]) for x in row)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    print(f"[OCR] rows={row_texts} count={len(row_texts)}", flush=True)
    return row_texts, avg_confidence


def crop_plate(image: np.ndarray, box: list[float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    crop_w = x2 - x1
    crop_h = y2 - y1
    aspect = crop_w / max(crop_h, 1)

    if aspect < 1.8:
        margin_x = int(crop_w * 0.08)
        margin_y = int(crop_h * 0.14)
    else:
        margin_x = int(crop_w * 0.06)
        margin_y = int(crop_h * 0.08)

    crop_x1 = max(0, x1 - margin_x)
    crop_y1 = max(0, y1 - margin_y)
    crop_x2 = min(w, x2 + margin_x)
    crop_y2 = min(h, y2 + margin_y)

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    return crop if crop.size else image


def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    aspect = w / max(h, 1)
    scale = max(1.0, PLATE_SCALE)
    interpolation = cv2.INTER_CUBIC if aspect < 1.8 else cv2.INTER_LINEAR
    processed = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interpolation)
    processed = resize_max_side(processed, PLATE_OCR_MAX_SIDE, "plate crop")
    return processed


def recognize_plate_crop(crop: np.ndarray) -> tuple[str, float, list[str]]:
    processed = preprocess_plate(crop)
    try:
        items = run_ocr(processed, "plate")
        rows, ocr_conf = group_ocr_rows(items)
        if not rows:
            print("[PLATE] rejected: OCR returned no rows", flush=True)
            return "", 0.0, []

        motorcycle = motorcycle_candidates(rows)
        if motorcycle:
            raw = motorcycle[0]
            formatted = format_plate(raw, motorcycle=True)
            print(f"[PLATE] motorcycle raw={raw} formatted={formatted} ocr_conf={ocr_conf:.4f} rows={rows}", flush=True)
            return formatted, ocr_conf, motorcycle

        combined = "".join(rows)
        candidates = plate_candidates(combined)
        if not candidates:
            print(f"[PLATE] rejected OCR rows={rows} combined='{combined}'", flush=True)
            return "", 0.0, []

        raw = candidates[0]
        formatted = format_plate(raw)
        print(f"[PLATE] raw={raw} formatted={formatted} ocr_conf={ocr_conf:.4f} rows={rows} candidates={candidates}", flush=True)
        return formatted, ocr_conf, candidates
    finally:
        del processed
        gc.collect()
        trim_native_memory()


def detect(image: np.ndarray) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="YOLO model is not loaded")

    print("[DETECT] YOLO predict start", flush=True)
    results = model.predict(image, conf=CONF, imgsz=IMG_SIZE, verbose=False)
    print("[DETECT] YOLO predict complete", flush=True)

    result = results[0]
    boxes = result.boxes
    detections: list[dict[str, Any]] = []
    xyxy: list[list[float]] = []

    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy().tolist()
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        classes = boxes.cls.cpu().numpy().tolist()

        for box, conf, cls in zip(xyxy, confs, classes):
            class_name = (
                str(result.names.get(int(cls), cls))
                if hasattr(result, "names") and isinstance(result.names, dict)
                else str(cls)
            )
            detections.append({
                "box": [round(float(v), 2) for v in box],
                "confidence": round(float(conf), 4),
                "class": class_name,
            })

        xyxy = [list(map(float, box)) for box in xyxy]

    print(f"[DETECT] boxes={len(detections)}", flush=True)

    try:
        del boxes
        del result
        del results
    except Exception:
        pass
    gc.collect()
    trim_native_memory()

    # IMPORTANT: never invent/fallback to a number. No valid OCR => null.
    license_plate: str | None = None
    plate_confidence = 0.0
    best_yolo_confidence = 0.0

    if detections:
        ranked = sorted(
            zip(detections, range(len(detections))),
            key=lambda x: float(x[0]["confidence"]),
            reverse=True,
        )

        max_plate_attempts = max(1, int(os.getenv("MAX_PLATE_OCR_ATTEMPTS", "2")))
        for attempt, (detection_info, original_index) in enumerate(ranked[:max_plate_attempts]):
            print(f"[DETECT] OCR attempt={attempt + 1}/{max_plate_attempts} yolo_conf={detection_info['confidence']}", flush=True)
            candidate_crop = crop_plate(image, xyxy[original_index])
            try:
                candidate_plate, candidate_ocr_conf, _ = recognize_plate_crop(candidate_crop)
                if candidate_plate and candidate_plate.strip():
                    license_plate = candidate_plate.strip()
                    plate_confidence = candidate_ocr_conf
                    best_yolo_confidence = float(detection_info["confidence"])
                    break
            finally:
                del candidate_crop
                gc.collect()
                trim_native_memory()

        if license_plate:
            print(f"[DETECT] valid plate={license_plate} yolo_conf={best_yolo_confidence:.4f}", flush=True)
        else:
            print("[DETECT] no valid Vietnamese plate from YOLO crops", flush=True)

    return {
        "success": license_plate is not None,
        "licensePlate": license_plate,
        "confidence": best_yolo_confidence,
        "plateConfidence": plate_confidence,
        "detections": detections,
    }


@app.get("/")
def root():
    return {
        "service": "Vietnam License Plate AI Server",
        "status": "ok",
        "version": "1.6.0",
        "model": MODEL_PATH,
        "model_loaded": model is not None,
        "ocr_loaded": ocr is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "ocr_loaded": ocr is not None,
        "ocr_error": ocr_init_error,
        "version": "1.6.0",
    }


@app.post("/recognize", dependencies=[Depends(auth)])
async def recognize(file: UploadFile = File(...)):
    print(f"[RECOGNIZE] request received filename={file.filename} content_type={file.content_type}", flush=True)
    data = await file.read()
    print(f"[RECOGNIZE] file read bytes={len(data)}", flush=True)

    image = read_image(data)
    del data
    print(f"[RECOGNIZE] image decoded shape={image.shape}", flush=True)

    try:
        response = detect(image)
        print("[RECOGNIZE] success", flush=True)
        return response
    finally:
        del image
        gc.collect()
        trim_native_memory()


@app.post("/recognize/batch", dependencies=[Depends(auth)])
async def recognize_batch(files: list[UploadFile] = File(...)):
    results: list[dict[str, Any]] = []
    for file in files:
        data = await file.read()
        image = read_image(data)
        del data
        try:
            results.append({
                "filename": file.filename,
                "result": detect(image),
            })
        except Exception as exc:
            results.append({
                "filename": file.filename,
                "error": str(exc),
            })
        finally:
            del image
            gc.collect()
            trim_native_memory()

    return {"results": results}
