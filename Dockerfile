FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PADDLEX_HOME=/tmp/paddlex \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    MODEL_PATH=models/license-plate-finetune-v1s.pt

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

# MorseTechLab YOLOv11 license-plate detector (v1s, 37.8 MB).
# v1s is selected as the Railway-friendly variant; the model repository also
# provides larger n/m/l/x variants when more RAM is available.
RUN mkdir -p models && \
    python -c "from urllib.request import urlretrieve; urlretrieve('https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.pt?download=true', 'models/license-plate-finetune-v1s.pt')" && \
    ln -s license-plate-finetune-v1s.pt models/best.pt

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
