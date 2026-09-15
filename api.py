import os
import gc
import re
import asyncio
import threading
from typing import Any

import numpy as np
from fastapi import Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

import main as main_module
from main import app, auth, read_image, trim_native_memory

# Railway memory-safe plate-only inference.
# CCCD/document OCR has intentionally been removed from this service.
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_original_detect = main_module.detect
_original_plate_candidates = main_module.plate_candidates

try:
    if main_module.startup in app.router.on_startup:
        app.router.on_startup.remove(main_module.startup)
except Exception as exc:
    print(f"[MEMORY] could not remove eager YOLO startup: {exc}", flush=True)


@app.middleware("http")
async def ai_concurrency_guard(request, call_next):
    """Serialize memory-heavy AI inference without dropping normal batch requests."""
    if request.url.path != "/api/ai":
        return await call_next(request)

    while not _inference_lock.acquire(blocking=False):
        await asyncio.sleep(0.05)

    print("[CONCURRENCY] /api/ai inference slot acquired", flush=True)
    try:
        return await call_next(request)
    finally:
        _inference_lock.release()
        print("[CONCURRENCY] /api/ai slot released", flush=True)


def _ensure_yolo():
    if main_module.model is not None:
        return main_module.model
    with _model_lock:
        if main_module.model is None:
            from ultralytics import YOLO
            print(f"[YOLO] lazy loading model: {main_module.MODEL_PATH}", flush=True)
            main_module.model = YOLO(main_module.MODEL_PATH)
            print("[YOLO] lazy model ready", flush=True)
    return main_module.model


def _release_yolo():
    try:
        with _model_lock:
            main_module.model = None
        gc.collect()
        trim_native_memory()
    except Exception as exc:
        print(f"[MEMORY] YOLO release failed: {exc}", flush=True)


def _lazy_detect(image: np.ndarray) -> dict[str, Any]:
    _ensure_yolo()
    return _original_detect(image)


main_module.detect = _lazy_detect


def _plate_candidates_with_ocr_ambiguity(text: str) -> list[str]:
    """Handle common OCR confusion where the series letter is read as a digit."""
    candidates = list(_original_plate_candidates(text))
    raw = main_module.clean_alnum(text)

    if len(raw) == 8 and re.fullmatch(r"\d{3}\d{5}", raw):
        prefix = raw[:2]
        series_digit = raw[2]
        numbers = raw[3:]
        series_map = {
            "8": ["B"],
            "0": ["D", "O", "Q"],
            "1": ["I", "L"],
            "5": ["S"],
            "6": ["G"],
            "2": ["Z"],
        }
        for letter in series_map.get(series_digit, []):
            candidates.append(f"{prefix}{letter}{numbers}")

    return list(dict.fromkeys(candidates))


main_module.plate_candidates = _plate_candidates_with_ocr_ambiguity


def _resize_input_for_detection(image: np.ndarray) -> np.ndarray:
    """Resize the detection copy so its longest side is at most 640px."""
    if image is None or image.size == 0:
        return image

    max_side = max(1, int(os.getenv("AI_INPUT_MAX_SIDE", "640")))
    h, w = image.shape[:2]
    current = max(h, w)
    if current <= max_side:
        return image

    import cv2
    scale = max_side / current
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    print(f"[MEMORY] AI input resize: {w}x{h} -> {nw}x{nh}", flush=True)
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def _resize_plate_crop_memory_safe(crop: np.ndarray) -> np.ndarray:
    """Upscale tiny OCR crops and hard-cap large crops."""
    if crop is None or crop.size == 0:
        return crop

    max_side = max(1, int(os.getenv("PLATE_OCR_SAFE_MAX_SIDE", "640")))
    min_side = max(1, int(os.getenv("PLATE_OCR_MIN_SIDE", "360")))
    h, w = crop.shape[:2]
    current = max(h, w)

    import cv2

    if current < min_side:
        scale = min_side / current
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        print(f"[OCR] small plate crop upscale: {w}x{h} -> {nw}x{nh}", flush=True)
        return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)

    if current > max_side:
        scale = max_side / current
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        print(f"[MEMORY] plate crop cap: {w}x{h} -> {nw}x{nh}", flush=True)
        return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    return crop


def _safe_recognize_plate_crop(crop: np.ndarray):
    safe_crop = _resize_plate_crop_memory_safe(crop)
    try:
        return main_module.recognize_plate_crop(safe_crop)
    finally:
        if safe_crop is not crop:
            del safe_crop
        gc.collect()
        trim_native_memory()


def _improved_plate_detect(image: np.ndarray) -> dict[str, Any]:
    """Single-pass, memory-safe YOLO + PaddleOCR plate inference."""
    detect_image = _resize_input_for_detection(image)
    model = _ensure_yolo()
    print("[DETECT+] YOLO predict start", flush=True)

    results = model.predict(
        detect_image,
        conf=max(0.18, float(os.getenv("YOLO_CONF", "0.25"))),
        imgsz=int(os.getenv("YOLO_IMGSZ", "640")),
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes
    detections: list[dict[str, Any]] = []

    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.cpu().numpy().tolist()
        xyxy = boxes.xyxy.cpu().numpy().tolist()
        classes = boxes.cls.cpu().numpy().tolist()
        for box, conf, cls in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = [float(v) for v in box]
            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)
            aspect = width / height
            class_name = str(result.names.get(int(cls), cls)) if hasattr(result, "names") and isinstance(result.names, dict) else str(cls)
            detections.append({
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "confidence": round(float(conf), 4),
                "class": class_name,
                "aspect": round(aspect, 3),
                "motorcycleShape": aspect < 1.9,
            })

    # YOLO coordinates belong to the resized detection image. Keep them for
    # diagnostics, but map the selected box back to the original image before OCR.
    original_h, original_w = image.shape[:2]
    detect_h, detect_w = detect_image.shape[:2]
    scale_x = original_w / max(detect_w, 1)
    scale_y = original_h / max(detect_h, 1)

    del boxes, result, results, model
    if detect_image is not image:
        del detect_image
    gc.collect()
    trim_native_memory()

    if not detections:
        print("[DETECT+] no YOLO detections", flush=True)
        return {
            "success": False,
            "licensePlate": None,
            "confidence": 0.0,
            "plateConfidence": 0.0,
            "detections": [],
        }

    ranked = sorted(
        detections,
        key=lambda d: (1 if d["motorcycleShape"] else 0, float(d["confidence"])),
        reverse=True,
    )
    detection = ranked[0]

    print(
        f"[DETECT+] OCR attempt=1/1 conf={detection['confidence']} aspect={detection['aspect']}",
        flush=True,
    )

    detect_box = detection["box"]
    original_box = [
        detect_box[0] * scale_x,
        detect_box[1] * scale_y,
        detect_box[2] * scale_x,
        detect_box[3] * scale_y,
    ]
    detection["detectBox"] = detection["box"]
    detection["box"] = [round(v, 2) for v in original_box]

    detect_width = max(detect_box[2] - detect_box[0], 1.0)
    detect_height = max(detect_box[3] - detect_box[1], 1.0)
    original_width = max(original_box[2] - original_box[0], 1.0)
    original_height = max(original_box[3] - original_box[1], 1.0)
    print(
        f"[DETECT+] bbox mapped {detect_width:.0f}x{detect_height:.0f} -> {original_width:.0f}x{original_height:.0f}",
        flush=True,
    )

    crop = main_module.crop_plate(image, detection["box"])
    try:
        plate, ocr_conf, parsed = _safe_recognize_plate_crop(crop)
    finally:
        del crop
        gc.collect()
        trim_native_memory()

    if not plate:
        print("[DETECT+] no valid Vietnamese plate from YOLO crop", flush=True)
        return {
            "success": False,
            "licensePlate": None,
            "confidence": 0.0,
            "plateConfidence": 0.0,
            "detections": detections,
        }

    score = float(ocr_conf) * 0.70 + float(detection["confidence"]) * 0.30
    if detection["motorcycleShape"] and "-" in plate:
        score += 0.08

    print(f"[DETECT+] selected plate={plate} score={score:.4f}", flush=True)

    return {
        "success": True,
        "licensePlate": plate.strip(),
        "confidence": detection["confidence"],
        "plateConfidence": float(ocr_conf),
        "detections": detections,
        "candidateScores": [{
            "plate": plate.strip(),
            "score": round(score, 4),
            "ocrConfidence": round(float(ocr_conf), 4),
            "yoloConfidence": round(float(detection["confidence"]), 4),
        }],
    }


def classify_and_recognize(image: np.ndarray, requested_type: str = "plate") -> dict[str, Any]:
    plate_result = _improved_plate_detect(image)
    return {
        "success": bool(plate_result.get("licensePlate")),
        "type": "plate",
        "result": plate_result,
    }


@app.get("/memory")
def memory_status():
    return {
        "model_loaded": main_module.model is not None,
        "ocr_loaded": main_module.ocr is not None,
        "model_path": main_module.MODEL_PATH,
        "mode": "plate-only",
        "max_concurrent_inference": 1,
        "queue_mode": "wait",
        "ai_input_max_side": int(os.getenv("AI_INPUT_MAX_SIDE", "640")),
    }


@app.post("/api/ai", dependencies=[Depends(auth)])
async def unified_ai(
    file: UploadFile = File(...),
    type: str = "plate",
):
    print(f"[AI] request filename={file.filename} type={type} mode=plate-only", flush=True)
    data = await file.read()
    image = read_image(data)
    del data

    try:
        response = classify_and_recognize(image, type)
        print(
            f"[AI] result type={response.get('type')} success={response.get('success')}",
            flush=True,
        )
        return response
    finally:
        del image
        gc.collect()
        trim_native_memory()
        _release_yolo()
