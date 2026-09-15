FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PADDLEX_HOME=/tmp/paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    MODEL_PATH=models/license-plate-finetune-v1s.pt \
    YOLO_IMGSZ=640 \
    PLATE_OCR_MAX_SIDE=1280 \
    PLATE_OCR_SAFE_MAX_SIDE=960 \
    PLATE_OCR_MIN_SIDE=960 \
    PLATE_OCR_SCALE=1.5 \
    OCR_DET_SIDE=960 \
    OCR_DET_THRESH=0.18 \
    OCR_BOX_THRESH=0.35 \
    OCR_UNCLIP_RATIO=2.2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY main.py .
COPY api.py .

# The source file keeps the tuning change visible in GitHub, while this build
# guard removes the accidental unsupported constructor keyword before startup.
RUN sed -i '/device_id=0 if False else None,/d' main.py

RUN mkdir -p models && \
    python -c "from urllib.request import urlretrieve; urlretrieve('https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.pt?download=true', 'models/license-plate-finetune-v1s.pt')" && \
    ln -s license-plate-finetune-v1s.pt models/best.pt

EXPOSE 8000

# Exactly one worker: one YOLO/PaddleOCR runtime only.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
