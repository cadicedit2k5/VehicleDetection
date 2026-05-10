# 🚗 Vehicle Detection — Web App

Ứng dụng web nhận diện phương tiện giao thông sử dụng mô hình **YOLOv3** được huấn luyện trên **BDD100K + COCO 2017**.  
Backend: **Flask (Python)** — Frontend: **HTML + CSS + JavaScript** (thuần, không framework).

---

## 📁 Cấu Trúc Thư Mục

```
VehicleDetection/          ← Thư mục gốc dự án
├── models/
│   └── yolo_bd100k.pth    ← File weights đã train (cần có trước khi chạy)
├── notebooks/
│   └── myyolovehicledetection.ipynb
└── webapp/                ← Thư mục ứng dụng web
    ├── app.py             ← Flask server (entry point)
    ├── model.py           ← Định nghĩa kiến trúc YOLOv3
    ├── utils.py           ← Tiền xử lý, inference, vẽ bounding box
    ├── requirements.txt   ← Danh sách thư viện Python
    └── static/
        ├── index.html     ← Giao diện người dùng
        ├── css/
        │   └── style.css
        └── js/
            └── main.js
```

> **Lưu ý quan trọng:** Các lệnh `python` và `flask` phải được chạy từ thư mục **`VehicleDetection/`** (thư mục chứa cả `models/` và `webapp/`), không phải từ bên trong `webapp/`.

---

## ⚙️ Yêu Cầu Hệ Thống

| Thành phần | Phiên bản tối thiểu |
|-----------|---------------------|
| Python | 3.9+ |
| PyTorch | 1.13+ |
| CUDA (tùy chọn) | 11.x+ (tăng tốc GPU) |
| RAM | ≥ 4 GB |
| Dung lượng | ~750 MB (cho file weights) |

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### Bước 1 — Di chuyển vào thư mục gốc dự án

```bash
cd /path/to/VehicleDetection
```

### Bước 2 — Tạo và kích hoạt môi trường ảo

```bash
# Tạo virtualenv
python -m venv .venv

# Kích hoạt (Linux / macOS)
source .venv/bin/activate

# Kích hoạt (Windows)
.venv\Scripts\activate
```

### Bước 3 — Cài đặt các thư viện

```bash
pip install -r webapp/requirements.txt
```

Danh sách thư viện (`requirements.txt`):

```
flask>=2.3
torch>=1.13
torchvision
opencv-python-headless
Pillow>=9.0
numpy
```

> **Với GPU NVIDIA:** Cài PyTorch có hỗ trợ CUDA từ [pytorch.org](https://pytorch.org/get-started/locally/) trước khi chạy lệnh trên.

### Bước 4 — Kiểm tra file weights

Đảm bảo file checkpoint đã có tại đúng vị trí:

```bash
ls models/
# Kết quả mong đợi:
# yolo_bd100k.pth
```

> Nếu chưa có, tải file `.pth` từ Kaggle Model: `editcadic/bd100k-yolo-model` và đặt vào thư mục `models/`.

### Bước 5 — Khởi động server

```bash
python -m flask --app webapp.app run --host=0.0.0.0 --port=5001
```

Hoặc chạy trực tiếp:

```bash
python webapp/app.py
```

### Bước 6 — Mở trình duyệt

```
http://localhost:5001
```

---

## 🔌 API Endpoints

| Endpoint | Method | Mô Tả |
|----------|--------|-------|
| `GET /` | GET | Trang giao diện chính |
| `GET /model-info` | GET | Thông tin model đang chạy |
| `POST /detect` | POST | Gửi ảnh để nhận diện phương tiện |

### `POST /detect` — Chi tiết

**Request** (multipart/form-data):

| Field | Type | Bắt buộc | Mô Tả |
|-------|------|----------|-------|
| `image` | File | ✅ | File ảnh (JPEG, PNG, ...) |
| `conf_thresh` | float | ❌ | Ngưỡng độ tin cậy (mặc định: `0.80`) |
| `iou_thresh` | float | ❌ | Ngưỡng IoU cho NMS (mặc định: `0.50`) |

**Response** (JSON):

```json
{
  "boxes": [
    {
      "class_id": 0,
      "class_name": "car",
      "score": 0.9231,
      "x": 0.523,
      "y": 0.418,
      "w": 0.142,
      "h": 0.089
    }
  ],
  "count_per_class": {
    "car": 3,
    "bus": 1,
    "truck": 0,
    "motorcycle": 0,
    "bicycle": 0
  },
  "total": 4,
  "inference_ms": 42.5,
  "original_size": { "w": 1280, "h": 720 },
  "result_image": "<base64 JPEG string>"
}
```

> Tọa độ `x`, `y`, `w`, `h` được chuẩn hóa trong khoảng **[0, 1]** theo định dạng YOLO (x_center, y_center, width, height).

---

## 🧪 Kiểm Tra Nhanh bằng `curl`

```bash
# Kiểm tra server đang chạy
curl http://localhost:5001/model-info

# Gửi ảnh để detect (conf=0.8, iou=0.5)
curl -X POST http://localhost:5001/detect \
  -F "image=@/path/to/your/image.jpg" \
  -F "conf_thresh=0.8" \
  -F "iou_thresh=0.5"
```

---

## 🎨 Các Lớp & Màu Sắc

| Class | ID | Màu |
|-------|----|-----|
| `car` | 0 | 🔵 Cyan |
| `bus` | 1 | 🟠 Orange |
| `truck` | 2 | 🟢 Green |
| `motorcycle` | 3 | 🟣 Pink-Purple |
| `bicycle` | 4 | 🟡 Yellow |

---

## ⚠️ Xử Lý Sự Cố

### Lỗi: `No module named 'webapp'`

Đảm bảo bạn đang chạy từ thư mục **`VehicleDetection/`** (thư mục cha của `webapp/`):

```bash
cd /path/to/VehicleDetection
python -m flask --app webapp.app run --port=5001
```

### Lỗi: `Không tìm thấy checkpoint`

Server vẫn khởi động nhưng chạy ở **chế độ Demo** (không có inference thật).  
Kiểm tra đường dẫn weights được cấu hình trong `webapp/utils.py`:

```python
# Dòng 45 trong utils.py
CHECK_POINT = 'models/yolo_bd100k.pth'
```

Đảm bảo file `yolo_bd100k.pth` nằm trong `models/` tại thư mục gốc.

### Lỗi: `Không thể đọc ảnh`

- Kiểm tra ảnh có đúng định dạng không (JPEG, PNG, BMP, WEBP).
- Dung lượng ảnh quá lớn: thử resize xuống dưới 10 MB trước khi upload.

### Chạy chậm / không có GPU

Mô hình tự động dùng CPU nếu không có GPU.  
Để kiểm tra device đang dùng:

```bash
curl http://localhost:5001/model-info | python -m json.tool | grep device
```

---

## 🔧 Cấu Hình Nâng Cao

### Đổi port

```bash
python -m flask --app webapp.app run --port=8080
```

### Đổi file weights

Sửa `CHECK_POINT` trong `webapp/utils.py`:

```python
CHECK_POINT = 'models/yolov3_bdd100k (6).pth'
```

### Bật chế độ Debug (reload tự động)

```bash
python -m flask --app webapp.app run --debug --port=5001
```

> **Không dùng `--debug` trên môi trường production.**

---

## 📦 Chạy Bằng Môi Trường Có Sẵn (`.venv`)

Nếu thư mục `.venv` đã tồn tại trong project:

```bash
# Kích hoạt
source .venv/bin/activate

# Chạy server
python -m flask --app webapp.app run --host=0.0.0.0 --port=5001
```

---

*Flask server — YOLOv3 Vehicle Detection | BDD100K + COCO 2017*
