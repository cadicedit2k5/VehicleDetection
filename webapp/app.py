"""
Flask server — YOLOv3 Vehicle Detection Web App
"""

import os
import glob
import logging

from flask import Flask, request, jsonify, send_from_directory

from webapp import utils

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", static_url_path="")

# ─── Model init ──────────────────────────────────────────────────────────────

def find_checkpoint():
    """Tìm file .pth đầu tiên trong thư mục models/ (đường dẫn tương đối)."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    patterns = [
        os.path.join(base, "models", "*.pth"),
        os.path.join(base, "models", "*.pt"),
    ]
    for p in patterns:
        files = glob.glob(p)
        if files:
            return files[0]
    return None


MODEL_PATH = utils.CHECK_POINT or find_checkpoint()

if MODEL_PATH and os.path.exists(MODEL_PATH):
    logger.info(f"Đang load checkpoint: {MODEL_PATH}")
    utils.load_model(MODEL_PATH)
    MODEL_LOADED = True
else:
    print(MODEL_PATH)
    logger.warning("Không tìm thấy checkpoint. Chạy ở chế độ Demo (không có inference thật).")
    MODEL_LOADED = False


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/model-info")
def model_info():
    return jsonify({
        "loaded":       MODEL_LOADED,
        "checkpoint":   os.path.basename(MODEL_PATH) if MODEL_PATH else None,
        "classes":      utils.CLASSES,
        "input_size":   utils.IMAGE_SIZE,
        "num_classes":  utils.NUM_CLASSES,
        "device":       str(utils.get_device()),
        "anchors":      utils.ANCHORS,
    })


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "Thiếu file ảnh (field: image)"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "File trống"}), 400

    conf_thresh = float(request.form.get("conf_thresh", 0.80))
    iou_thresh  = float(request.form.get("iou_thresh",  0.5))

    try:
        image_bytes = file.read()
        result = utils.run_inference(image_bytes, conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Lỗi khi inference")
        return jsonify({"error": f"Lỗi server: {str(e)}"}), 500


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
