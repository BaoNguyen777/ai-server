import re
import gc
from typing import Any

import cv2
import numpy as np
from fastapi import Depends, File, HTTPException, UploadFile

from main import (
    app,
    auth,
    detect,
    get_ocr,
    read_image,
    resize_max_side,
    _json_from_result,
    trim_native_memory,
)

CCCD_MAX_SIDE = int(__import__('os').getenv('CCCD_OCR_MAX_SIDE', '1600'))


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

        # Vietnamese CCCD number: exactly 12 digits. Prefer a standalone 12-digit match.
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

        # Keep parsing conservative: only assign a name when a likely label exists.
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

        # The order on Vietnamese CCCD is commonly: DOB, sex, nationality, birthplace/residence.
        # We expose OCR lines too, so uncertain fields are never fabricated.
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

    if mode in {'auto', 'plate'}:
        plate_result = detect(image)
        if plate_result.get('licensePlate'):
            return {
                'success': True,
                'type': 'plate',
                'result': plate_result,
            }
        if mode == 'plate':
            return {
                'success': False,
                'type': 'plate',
                'result': plate_result,
            }

    cccd_result = _ocr_document(image)
    if cccd_result.get('documentType') == 'cccd':
        return {
            'success': True,
            'type': 'cccd',
            'result': cccd_result,
        }

    return {
        'success': False,
        'type': 'unknown',
        'result': {
            'plate': plate_result if mode == 'auto' else None,
            'cccd': cccd_result,
        },
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
