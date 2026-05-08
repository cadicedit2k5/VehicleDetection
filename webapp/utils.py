"""
Utility functions — trích và mở rộng từ notebook myyolovehicledetection.ipynb
"""

import io
import time
import base64
import logging

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from webapp.model import YOLOv3

# ─── Constants ───────────────────────────────────────────────────────────────

CLASSES   = ["car", "bus", "truck", "motorcycle", "bicycle"]
IMAGE_SIZE = 416
NUM_CLASSES = 5

# Anchors chuẩn hóa từ notebook
ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9,  0.78)],   # Scale 1 – 13×13
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],   # Scale 2 – 26×26
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],   # Scale 3 – 52×52
]
S = [13, 26, 52]

# Màu sắc của từng class (BGR cho OpenCV, RGBA cho canvas)
CLASS_COLORS_RGB = {
    "car":        (0,  210, 255),   # Cyan
    "bus":        (255, 140,  0),   # Orange
    "truck":      (50,  220, 100),  # Green
    "motorcycle": (220,  50, 180),  # Pink-purple
    "bicycle":    (255, 220,  30),  # Yellow
}

# Normalize ImageNet
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# CHECK_POINT = 'models/yolov3_bdd100k (6).pth'
CHECK_POINT = 'models/yolo_bd100k.pth'


_model = None
_device = None

logger = logging.getLogger(__name__)


# ─── Model Loading ────────────────────────────────────────────────────────────

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: str):
    """Tải YOLOv3 và nạp weights từ checkpoint."""
    global _model, _device
    _device = get_device()
    logger.info(f"Đang dùng device: {_device}")

    _model = YOLOv3(num_classes=NUM_CLASSES).to(_device)
    _model.eval()

    checkpoint = torch.load(checkpoint_path, map_location=_device, weights_only=False)

    # Hỗ trợ nhiều định dạng checkpoint
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint   # Raw state_dict

    _model.load_state_dict(state)
    logger.info("Đã load model thành công!")
    return _model


def get_model():
    return _model


# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    """Resize → RGB → Normalize → Tensor (1, 3, 416, 416)."""
    img = cv2.resize(image_bgr, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(_device)


# ─── Post-processing ─────────────────────────────────────────────────────────

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    else:
        box1_x1, box1_y1 = boxes_preds[..., 0:1], boxes_preds[..., 1:2]
        box1_x2, box1_y2 = boxes_preds[..., 2:3], boxes_preds[..., 3:4]
        box2_x1, box2_y1 = boxes_labels[..., 0:1], boxes_labels[..., 1:2]
        box2_x2, box2_y2 = boxes_labels[..., 2:3], boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection  = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, prob_threshold, box_format="midpoint"):
    bboxes = [b for b in bboxes if b[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)[:100]
    result = []
    while bboxes:
        chosen = bboxes.pop(0)
        bboxes = [
            b for b in bboxes
            if b[0] != chosen[0]
            or intersection_over_union(
                torch.tensor(chosen[2:]), torch.tensor(b[2:]),
                box_format=box_format
            ) < iou_threshold
        ]
        result.append(chosen)
    return result


def cells_to_bboxes(predictions, anchors, S_val, is_preds=True):
    BATCH_SIZE = predictions.shape[0]
    box_predictions = predictions[..., 1:5].clone()
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:]  = torch.exp(box_predictions[..., 2:]) * anchors
        scores     = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores     = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S_val)
        .repeat(BATCH_SIZE, 3, S_val, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x   = 1 / S_val * (box_predictions[..., 0:1] + cell_indices)
    y   = 1 / S_val * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S_val * box_predictions[..., 2:4]

    converted = torch.cat((best_class, scores, x, y, w_h), dim=-1)
    return converted.reshape(BATCH_SIZE, 3 * S_val * S_val, 6).tolist()


# ─── Inference Pipeline ───────────────────────────────────────────────────────

def run_inference(image_bytes: bytes, conf_thresh: float = 0.5, iou_thresh: float = 0.5):
    """
    Nhận raw image bytes, trả về dict kết quả inference.
    Boxes trả về: [class_id, score, x_center, y_center, w, h]  (tọa độ 0-1)
    """
    model = get_model()
    if model is None:
        raise RuntimeError("Model chưa được load!")

    # Decode ảnh
    arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Không thể đọc ảnh. Hãy thử ảnh khác.")

    orig_h, orig_w = img_bgr.shape[:2]

    # Preprocess
    tensor = preprocess(img_bgr)

    # Inference
    t0 = time.perf_counter()
    with torch.no_grad():
        raw_outputs = model(tensor)
    inference_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Decode boxes từ 3 scale
    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(_device)

    all_boxes = []
    for i in range(3):
        anchor_t = scaled_anchors[i]
        batch_boxes = cells_to_bboxes(raw_outputs[i], anchor_t, S_val=S[i], is_preds=True)
        all_boxes += batch_boxes[0]

    # NMS
    nms_boxes = non_max_suppression(all_boxes, iou_thresh, conf_thresh, box_format="midpoint")

    # Đếm theo class
    count_per_class = {c: 0 for c in CLASSES}
    for b in nms_boxes:
        cls_id = int(b[0])
        if 0 <= cls_id < len(CLASSES):
            count_per_class[CLASSES[cls_id]] += 1

    # Chuyển về dạng dict tuần tự
    boxes_out = []
    for b in nms_boxes:
        cls_id = int(b[0])
        if cls_id < 0 or cls_id >= len(CLASSES):
            continue
        boxes_out.append({
            "class_id":   cls_id,
            "class_name": CLASSES[cls_id],
            "score":      round(float(b[1]), 4),
            "x":          round(float(b[2]), 6),   # x_center (0-1)
            "y":          round(float(b[3]), 6),   # y_center (0-1)
            "w":          round(float(b[4]), 6),   # width    (0-1)
            "h":          round(float(b[5]), 6),   # height   (0-1)
        })

    # Vẽ ảnh kết quả để preview
    result_image_b64 = draw_boxes_on_image(img_bgr, boxes_out)

    return {
        "boxes":           boxes_out,
        "count_per_class": count_per_class,
        "total":           len(boxes_out),
        "inference_ms":    inference_ms,
        "original_size":   {"w": orig_w, "h": orig_h},
        "result_image":    result_image_b64,
    }


# ─── Draw Boxes ───────────────────────────────────────────────────────────────

def draw_boxes_on_image(img_bgr: np.ndarray, boxes: list) -> str:
    """Vẽ bounding boxes lên ảnh và trả về base64 JPEG."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    W, H = pil_img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for b in boxes:
        cls_name = b["class_name"]
        color = CLASS_COLORS_RGB.get(cls_name, (255, 255, 255))

        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        x1 = int((x - w / 2) * W)
        y1 = int((y - h / 2) * H)
        x2 = int((x + w / 2) * W)
        y2 = int((y + h / 2) * H)

        # Vẽ rectangle
        for t in range(3):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)

        # Label background
        label = f"{cls_name} {b['score']:.2f}"
        bbox_text = font.getbbox(label)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - th - 3), label, fill=(0, 0, 0), font=font)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
