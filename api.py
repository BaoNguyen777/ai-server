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

# main.py eagerly loaded YOLO during startup. Remove that handler so Railway
# can boot with OCR/CPU dependencies without allocating the YOLO weights.
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

# Existing /recognize routes in main.py resolve detect through main's globals.
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


def classify_and_recognize(image: np.ndarray, requested_type: str) -> dict[str, Any]:
    mode = requested_type.strip().lower() if requested_type else 'auto'
    if mode not in {'auto', 'plate', 'cccd'}:
        raise HTTPException(status_code=400, detail='type must be auto, plate, or cccd')

    # OCR first. A CCCD/auto request does not allocate YOLO unless OCR fails
    # to find a 12-digit CCCD number.
    if mode in {'auto', 'cccd'}:
        cccd_result = _ocr_document(image)
        if cccd_result.get('documentType') == 'cccd':
            return {'success': True, 'type': 'cccd', 'result': cccd_result}
        if mode == 'cccd':
            return {'success': False, 'type': 'cccd', 'result': cccd_result}

    plate_result = _lazy_detect(image)
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
