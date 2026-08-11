from pathlib import Path
from uuid import uuid4
import json
import os
import re

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
PROVINCE_RE = re.compile(r"^[\u4e00-\u9fff]")
AMBIGUOUS = {
    "4": {"A"},
    "A": {"4"},
    "0": {"O", "D", "Q"},
    "O": {"0", "D", "Q"},
    "D": {"0", "O", "Q"},
    "Q": {"0", "O", "D"},
    "1": {"I", "L", "T"},
    "I": {"1", "L", "T"},
    "L": {"1", "I", "T"},
    "T": {"1", "I", "L", "7"},
    "7": {"T"},
    "5": {"S"},
    "S": {"5"},
    "8": {"B"},
    "B": {"8"},
    "2": {"Z"},
    "Z": {"2"},
    "6": {"G"},
    "G": {"6"},
}

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


def plate_core(value):
    text = re.sub(r"[^\u4e00-\u9fffA-Z0-9]", "", str(value or "").upper())
    return PROVINCE_RE.sub("", text)


def raw_plate_text(value):
    return str(value or "").upper().strip()


def is_low_quality_ocr(value):
    raw = raw_plate_text(value)
    clean = plate_core(raw)
    if not raw or raw in {"???", "未识别"}:
        return True
    if "?" in raw:
        return True
    if len(clean) < 4:
        return True
    if sum(ch.isdigit() for ch in clean) < 2 and not re.search(r"[\u4e00-\u9fff]", raw):
        return True
    return False


def reliable_suffix_match(candidate, target):
    cand = plate_core(candidate)
    tgt = plate_core(target)
    if len(cand) < 5 or len(tgt) < 5:
        return False
    return tgt.endswith(cand) or cand.endswith(tgt)


def has_same_chinese_prefix(candidate, target):
    cand = raw_plate_text(candidate)
    tgt = raw_plate_text(target)
    return len(cand) >= 2 and len(tgt) >= 2 and cand[0] == tgt[0] and cand[1] == tgt[1]


def safe_auto_match(candidate, target, score, low_quality):
    if low_quality:
        return score <= 0.35
    cand = plate_core(candidate)
    tgt = plate_core(target)
    if cand == tgt:
        return True
    if reliable_suffix_match(candidate, target) and score <= 0.35:
        return True
    if has_same_chinese_prefix(candidate, target) and abs(len(cand) - len(tgt)) <= 1:
        return score <= 1.05
    return score <= 1.25


def unique_plate_numbers(numbers):
    clean = []
    for value in numbers:
        text = raw_plate_text(value)
        if not text or text in {"???", "未识别"} or "?" in text:
            continue
        if any(text != old and text in old for old in clean):
            continue
        clean = [old for old in clean if not (old != text and old in text)]
        if text not in clean:
            clean.append(text)
    return clean


def visual_equal(a, b):
    return a == b or b in AMBIGUOUS.get(a, set())


def edit_distance_score(candidate, target):
    cand = plate_core(candidate)
    tgt = plate_core(target)
    if not cand or not tgt:
        return 999
    rows = len(cand) + 1
    cols = len(tgt) + 1
    dp = [[0.0] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = dp[i - 1][0] + 1.0
    for j in range(1, cols):
        dp[0][j] = dp[0][j - 1] + 1.0
    for i in range(1, rows):
        for j in range(1, cols):
            if cand[i - 1] == tgt[j - 1]:
                sub = 0.0
            elif visual_equal(cand[i - 1], tgt[j - 1]):
                sub = 0.35
            else:
                sub = 1.0
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + sub,
            )
    return dp[-1][-1]


def similar_plate_score(candidate, target):
    cand = plate_core(candidate)
    tgt = plate_core(target)
    if not cand or not tgt:
        return 999
    if cand == tgt:
        return 0
    if len(cand) >= 4 and tgt.endswith(cand):
        return 0.15
    if len(cand) >= 5 and cand.endswith(tgt):
        return 0.25

    score = edit_distance_score(cand, tgt)

    if has_same_chinese_prefix(candidate, target):
        score = min(score, edit_distance_score(raw_plate_text(candidate)[1:], raw_plate_text(target)[1:]) - 0.25)

    # OCR often drops the province/city prefix on Chinese blue plates.
    for start in range(1, min(3, len(tgt)) + 1):
        tail = tgt[start:]
        if len(tail) >= 4:
            score = min(score, 0.25 + edit_distance_score(cand, tail))

    # If the OCR result is a damaged suffix, still allow archive correction.
    for tail_len in range(4, min(len(cand), len(tgt)) + 1):
        score = min(score, 0.35 + edit_distance_score(cand[-tail_len:], tgt[-tail_len:]))

    return score


def correct_plate_with_fleet(result):
    fleet = read_json("fleet.json", [])
    known = [item.get("plate", "") for item in fleet]
    corrected_numbers = []
    for plate in result.get("plates", []):
        text = plate.get("text")
        if not text or text in {"???", "未识别"}:
            continue
        if "?" in raw_plate_text(text):
            plate["rawText"] = text
            plate["text"] = "未识别"
            plate["ocrMethod"] = f"{plate.get('ocrMethod', 'OCR')} + 低可信待复核"
            continue
        low_quality = is_low_quality_ocr(text)
        best = None
        best_score = 999
        for target in known:
            if low_quality and not reliable_suffix_match(text, target):
                continue
            score = similar_plate_score(text, target)
            if score < best_score:
                best = target
                best_score = score
        if best and safe_auto_match(text, best, best_score, low_quality) and text != best:
            plate["rawText"] = text
            plate["text"] = best
            plate["ocrMethod"] = f"{plate.get('ocrMethod', 'OCR')} + 车牌档案校正"
        corrected_numbers.append(plate.get("text"))
    corrected_numbers = [
        num for num in corrected_numbers
        if num and num not in {"???", "未识别"} and "?" not in raw_plate_text(num)
    ]
    if corrected_numbers:
        result["summary"]["plateNumbers"] = unique_plate_numbers(corrected_numbers)
        result["status"] = "recognized"
        result["message"] = "已识别车牌"
    else:
        result["summary"]["plateNumbers"] = []
        if result.get("plates"):
            result["status"] = "plate-found"
            result["message"] = "已检测到车牌，OCR 结果需要人工复核"
    return result


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
        return jsonify({"error": "请上传车辆图片"}), 400
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
        return jsonify({"error": "图片读取失败，请换一张清晰图片"}), 400

    result = correct_plate_with_fleet(detector.detect(image, original_name))
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
    vehicle_source = result["vehicles"] or [{"box": [0, 0, image.shape[1], image.shape[0]], "label": "整车图"}]
    for index, vehicle in enumerate(vehicle_source[:4], start=1):
        crop = detector.crop(image, vehicle["box"])
        if crop.size == 0:
            continue
        vehicle_name = f"{job_id}_vehicle_{index}.jpg"
        vehicle_path = OUTPUT_DIR / vehicle_name
        cv2.imencode(".jpg", crop)[1].tofile(str(vehicle_path))
        vehicle_images.append(f"/outputs/{vehicle_name}")

    return jsonify({
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
    })


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
