import os
import re
import gc
import threading
from typing import Any

import cv2
import numpy as np
from fastapi import Depends, File, HTTPException, UploadFile

import main as main_module
from main import (
    app,
    auth,
    get_ocr,
    read_image,
    resize_max_side,
    _json_from_result,
    trim_native_memory,
)

CCCD_MAX_SIDE = int(os.getenv('CCCD_OCR_MAX_SIDE', '1600'))
_model_lock = threading.Lock()
_original_detect = main_module.detect

try:
    if main_module.startup in app.router.on_startup:
        app.router.on_startup.remove(main_module.startup)
except Exception as exc:
    print(f'[MEMORY] could not remove eager YOLO startup: {exc}', flush=True)


def _ensure_yolo():
    if main_module.model is not None:
        return main_module.model
    with _model_lock:
        if main_module.model is None:
            from ultralytics import YOLO
            print(f'[YOLO] lazy loading model: {main_module.MODEL_PATH}', flush=True)
            main_module.model = YOLO(main_module.MODEL_PATH)
            print('[YOLO] lazy model ready', flush=True)
    return main_module.model


def _lazy_detect(image: np.ndarray) -> dict[str, Any]:
    _ensure_yolo()
    return _original_detect(image)

main_module.detect = _lazy_detect


def _clean_text(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def _ocr_document(image: np.ndarray) -> dict[str, Any]:
    pipeline = get_ocr()
    if pipeline is None:
        raise HTTPException(status_code=503, detail='PaddleOCR is unavailable')

    image_for_ocr = resize_max_side(image, CCCD_MAX_SIDE, 'cccd')
    result = None
    try:
        print(f'[CCCD] OCR start shape={image_for_ocr.shape}', flush=True)
        result = pipeline.predict(image_for_ocr)
        items: list[dict[str, Any]] = []
        for item in result:
            data = _json_from_result(item)
            if not isinstance(data, dict):
                continue
            texts = data.get('rec_texts', []) or []
            scores = data.get('rec_scores', []) or []
            polys = data.get('rec_polys', []) or data.get('dt_polys', []) or []
            for i, text in enumerate(texts):
                text = _clean_text(text)
                if not text:
                    continue
                try:
                    score = float(scores[i]) if i < len(scores) else 0.0
                except Exception:
                    score = 0.0
                cx = cy = 0.0
                if i < len(polys):
                    try:
                        points = np.asarray(polys[i], dtype=float).reshape(-1, 2)
                        cx = float(points[:, 0].mean())
                        cy = float(points[:, 1].mean())
                    except Exception:
                        pass
                items.append({'text': text, 'confidence': score, 'cx': cx, 'cy': cy})

        items.sort(key=lambda x: (x['cy'], x['cx']))
        lines = [x['text'] for x in items]
        raw_text = '\n'.join(lines)
        compact = re.sub(r'[^0-9]', '', raw_text)

        number_matches = re.findall(r'(?<!\d)(?:\d[ .-]?){12}(?!\d)', raw_text)
        cccd_number = None
        if number_matches:
            candidate = re.sub(r'\D', '', number_matches[0])
            if len(candidate) == 12:
                cccd_number = candidate
        if cccd_number is None:
            fallback = re.findall(r'(?<!\d)\d{12}(?!\d)', compact)
            cccd_number = fallback[0] if fallback else None

        date_matches = re.findall(r'(?<!\d)(\d{1,2})[/. -](\d{1,2})[/. -](\d{4})(?!\d)', raw_text)
        dates = [f'{int(d):02d}/{int(m):02d}/{y}' for d, m, y in date_matches]

        full_name = None
        for i, line in enumerate(lines):
            upper = line.upper()
            if any(label in upper for label in ('HỌ VÀ TÊN', 'FULL NAME', 'HO VA TEN')):
                candidate = re.sub(r'^(HỌ\s*VÀ\s*TÊN|HO\s*VA\s*TEN|FULL\s*NAME)\s*[:.-]?\s*', '', line, flags=re.I).strip()
                if not candidate and i + 1 < len(lines):
                    candidate = lines[i + 1].strip()
                if candidate:
                    full_name = candidate
                break

        fields = {
            'cccdNumber': cccd_number,
            'fullName': full_name,
            'dateOfBirth': dates[0] if dates else None,
            'otherDates': dates[1:],
        }
        confidence_values = [float(x['confidence']) for x in items if x['confidence'] > 0]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

        print(f'[CCCD] number={cccd_number} dates={dates} items={len(items)}', flush=True)
        return {
            'success': bool(cccd_number or items),
            'documentType': 'cccd' if cccd_number else 'unknown',
            'confidence': round(confidence, 4),
            'fields': fields,
            'rawText': raw_text,
            'ocrItems': items,
        }
    finally:
        result = None
        if image_for_ocr is not image:
            del image_for_ocr
        gc.collect()
        trim_native_memory()


def _enhance_motorcycle_crop(crop: np.ndarray) -> np.ndarray:
    if crop is None or crop.size == 0:
        return crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _improved_plate_detect(image: np.ndarray) -> dict[str, Any]:
    """Plate inference optimized for Vietnamese motorcycle two-line plates.

    The existing detector often chooses the highest-confidence crop. For a
    motorcycle image that can be the wrong rectangle. Here we rank portrait
    plate boxes first, try several candidates, and only run a lightweight
    enhancement retry when the first OCR pass fails.
    """
    model = _ensure_yolo()
    print('[DETECT+] YOLO predict start', flush=True)
    results = model.predict(
        image,
        conf=max(0.18, float(os.getenv('YOLO_CONF', '0.25'))),
        imgsz=int(os.getenv('YOLO_IMGSZ', '960')),
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
            class_name = str(result.names.get(int(cls), cls)) if hasattr(result, 'names') and isinstance(result.names, dict) else str(cls)
            detections.append({
                'box': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                'confidence': round(float(conf), 4),
                'class': class_name,
                'aspect': round(aspect, 3),
                'motorcycleShape': aspect < 1.9,
            })
    try:
        del boxes
        del result
        del results
    except Exception:
        pass
    gc.collect()
    trim_native_memory()

    # Portrait/near-square plates are much more likely to be motorcycle plates.
    ranked = sorted(
        enumerate(detections),
        key=lambda pair: (
            1 if pair[1]['motorcycleShape'] else 0,
            float(pair[1]['confidence']),
        ),
        reverse=True,
    )

    max_attempts = max(1, int(os.getenv('MAX_PLATE_OCR_ATTEMPTS', '3')))
    candidates: list[dict[str, Any]] = []
    for attempt, (index, detection) in enumerate(ranked[:max_attempts]):
        print(f"[DETECT+] OCR attempt={attempt + 1}/{max_attempts} conf={detection['confidence']} aspect={detection['aspect']}", flush=True)
        crop = main_module.crop_plate(image, detection['box'])
        try:
            plate, ocr_conf, parsed = main_module.recognize_plate_crop(crop)
            if not plate:
                # Motorcycle plates are small and often have uneven lighting.
                if detection['motorcycleShape']:
                    enhanced = _enhance_motorcycle_crop(crop)
                    try:
                        alt_plate, alt_conf, alt_parsed = main_module.recognize_plate_crop(enhanced)
                    finally:
                        del enhanced
                    if alt_plate and alt_conf >= ocr_conf:
                        plate, ocr_conf, parsed = alt_plate, alt_conf, alt_parsed

            if plate:
                score = float(ocr_conf) * 0.70 + float(detection['confidence']) * 0.30
                if detection['motorcycleShape'] and '-' in plate:
                    score += 0.08
                candidates.append({
                    'plate': plate.strip(),
                    'ocrConfidence': float(ocr_conf),
                    'yoloConfidence': float(detection['confidence']),
                    'score': score,
                    'detection': detection,
                    'parsed': parsed,
                })
        finally:
            del crop
            gc.collect()
            trim_native_memory()

    if not candidates:
        print('[DETECT+] no valid Vietnamese plate from YOLO crops', flush=True)
        return {
            'success': False,
            'licensePlate': None,
            'confidence': 0.0,
            'plateConfidence': 0.0,
            'detections': detections,
        }

    best = max(candidates, key=lambda item: item['score'])
    print(f"[DETECT+] selected plate={best['plate']} score={best['score']:.4f} candidates={[(x['plate'], round(x['score'], 4)) for x in candidates]}", flush=True)
    return {
        'success': True,
        'licensePlate': best['plate'],
        'confidence': best['yoloConfidence'],
        'plateConfidence': best['ocrConfidence'],
        'detections': detections,
        'candidateScores': [
            {'plate': item['plate'], 'score': round(item['score'], 4), 'ocrConfidence': round(item['ocrConfidence'], 4), 'yoloConfidence': round(item['yoloConfidence'], 4)}
            for item in candidates
        ],
    }


def classify_and_recognize(image: np.ndarray, requested_type: str) -> dict[str, Any]:
    mode = requested_type.strip().lower() if requested_type else 'auto'
    if mode not in {'auto', 'plate', 'cccd'}:
        raise HTTPException(status_code=400, detail='type must be auto, plate, or cccd')

    if mode in {'auto', 'cccd'}:
        cccd_result = _ocr_document(image)
        if cccd_result.get('documentType') == 'cccd':
            return {'success': True, 'type': 'cccd', 'result': cccd_result}
        if mode == 'cccd':
            return {'success': False, 'type': 'cccd', 'result': cccd_result}

    plate_result = _improved_plate_detect(image)
    return {
        'success': bool(plate_result.get('licensePlate')),
        'type': 'plate',
        'result': plate_result,
    }


@app.get('/memory')
def memory_status():
    return {
        'model_loaded': main_module.model is not None,
        'ocr_loaded': main_module.ocr is not None,
        'model_path': main_module.MODEL_PATH,
    }


@app.post('/api/ai', dependencies=[Depends(auth)])
async def unified_ai(
    file: UploadFile = File(...),
    type: str = 'auto',
):
    print(f'[AI] request filename={file.filename} type={type}', flush=True)
    data = await file.read()
    image = read_image(data)
    del data
    try:
        response = classify_and_recognize(image, type)
        print(f"[AI] result type={response.get('type')} success={response.get('success')}", flush=True)
        return response
    finally:
        del image
        gc.collect()
        trim_native_memory()
