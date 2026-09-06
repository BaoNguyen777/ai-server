# AI Server - Vietnam License Plate + CCCD OCR

Server này chạy `best.pt` bạn đã upload, dùng Ultralytics YOLO để detect vùng biển số và PaddleOCR để đọc ký tự. Ultralytics hỗ trợ load custom checkpoint bằng `YOLO("path/to/best.pt")` và predict trên ảnh; PaddleOCR hiện hỗ trợ pipeline OCR và ngôn ngữ Vietnamese. Xem tài liệu chính thức ở README cuối file.

## 1. Cấu trúc

```text
ai-server/
├── main.py
├── requirements.txt
├── .env.example
└── models/
    └── best.pt
```

## 2. Khuyến nghị môi trường

Dùng Python 3.10 hoặc 3.11. Tạo virtual environment:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Cài dependencies

```bash
pip install -r requirements.txt
```

PaddleOCR 3.x yêu cầu PaddlePaddle 3.x trở lên. Nếu máy có GPU NVIDIA/CUDA, hãy cài đúng build PaddlePaddle GPU theo CUDA của máy trước khi cài PaddleOCR.

## 4. Chạy

Copy `.env.example` thành `.env` và đổi `AI_API_KEY`.

Windows:

```bash
copy .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Sau đó:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Mở:

```text
http://127.0.0.1:8000/health
```

## 5. Test 1 ảnh

```bash
curl -X POST "http://127.0.0.1:8000/recognize" ^
  -H "x-api-key: change-this-secret" ^
  -F "file=@test.jpg"
```

PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/recognize" `
  -H "x-api-key: change-this-secret" `
  -F "file=@test.jpg"
```

Response ví dụ:

```json
{
  "licensePlate": "43A-123.45",
  "cccd": "",
  "confidence": 0.94,
  "plateConfidence": 0.91,
  "cccdConfidence": 0,
  "detections": [
    {
      "box": [120, 240, 520, 340],
      "confidence": 0.94,
      "class": "license_plate"
    }
  ]
}
```

## 6. Web 1 Vercel

Trong Vercel Environment Variables:

```env
AI_API_URL=https://YOUR-AI-SERVER/ 
AI_API_KEY=change-this-secret
```

Web 1 gọi `/api/recognize`, còn Next.js forward ảnh sang AI server.

## 7. Lưu ý về CCCD

`best.pt` là model detect biển số. Nó KHÔNG phải model detect CCCD. Phần CCCD trong server hiện dùng OCR toàn ảnh + regex 12 chữ số. Nếu ảnh CCCD có nhiều nền/nhiễu hoặc cần độ chính xác cao, nên thêm một model detector riêng cho vùng CCCD trước OCR.

Không log CCCD đầy đủ trong production và cần authentication/authorization nếu dữ liệu được dùng thật.

## 8. Batch

AI server có thêm:

```http
POST /recognize/batch
```

với nhiều `files`. Web 1 hiện gọi từng ảnh với concurrency 3 để dễ kiểm soát timeout; có thể đổi sang endpoint batch khi chuyển sang job queue.

## Official docs

Ultralytics Python/YOLO predict: https://docs.ultralytics.com/usage/python
PaddleOCR installation: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/installation.html
PaddleOCR OCR usage: https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/pipeline_usage/OCR.html
