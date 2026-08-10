from pathlib import Path
from uuid import uuid4
import json
import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from detector import CampusVehicleDetector


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(__name__)
detector = CampusVehicleDetector(BASE_DIR / "models")

def read_json(name, fallback):
    path = DATA_DIR / name
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return fallback

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")



@app.route("/api/fleet")
def fleet():
    return jsonify(read_json("fleet.json", []))


@app.route("/api/pass-records")
def pass_records():
    return jsonify(read_json("pass_records.json", []))


@app.route("/api/access-lists")
def access_lists():
    return jsonify(read_json("access_lists.json", {"white": [], "black": []}))


@app.route("/api/settings")
def settings():
    return jsonify(read_json("settings.json", {}))

@app.route("/detect", methods=["POST"])
def detect():
    image_file = request.files.get("image")
    if image_file is None or image_file.filename == "":
        return jsonify({"error": "请先选择一张车辆图片"}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"error": "仅支持 JPG、PNG、BMP、WEBP 图片"}), 400

    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    original_name = secure_filename(image_file.filename)
    suffix = Path(original_name).suffix.lower()
    job_id = uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    output_path = OUTPUT_DIR / f"{job_id}_result.jpg"
    image_file.save(input_path)

    image = cv2.imdecode(np.fromfile(str(input_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "图片读取失败，请换一张图片重试"}), 400

    result = detector.detect(image, original_name)
    annotated = detector.draw_result(image.copy(), result)
    cv2.imencode(".jpg", annotated)[1].tofile(str(output_path))

    plate_images = []
    for index, plate in enumerate(result["plates"], start=1):
        crop = detector.crop(image, plate["box"])
        if crop.size == 0:
            continue
        plate_name = f"{job_id}_plate_{index}.jpg"
        plate_path = OUTPUT_DIR / plate_name
        cv2.imencode(".jpg", crop)[1].tofile(str(plate_path))
        plate_images.append(f"/outputs/{plate_name}")

    vehicle_images = []
    vehicle_source = result["vehicles"] or [{"box": [0, 0, image.shape[1], image.shape[0]], "label": "原始车辆图"}]
    for index, vehicle in enumerate(vehicle_source[:4], start=1):
        crop = detector.crop(image, vehicle["box"])
        if crop.size == 0:
            continue
        vehicle_name = f"{job_id}_vehicle_{index}.jpg"
        vehicle_path = OUTPUT_DIR / vehicle_name
        cv2.imencode(".jpg", crop)[1].tofile(str(vehicle_path))
        vehicle_images.append(f"/outputs/{vehicle_name}")

    return jsonify(
        {
            "mode": result["mode"],
            "status": result.get("status", "recognized"),
            "message": result.get("message", ""),
            "summary": result["summary"],
            "vehicles": result["vehicles"],
            "plates": result["plates"],
            "vehicleType": result["vehicleType"],
            "resultImage": f"/outputs/{output_path.name}",
            "plateImages": plate_images,
            "vehicleImages": vehicle_images,
        }
    )


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)




