from datetime import datetime
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


def write_json(name, data):
    path = DATA_DIR / name
    DATA_DIR.mkdir(exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


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


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def today_str():
    return datetime.now().strftime("%Y-%m-%d")


def date_only(value):
    """将可能带时刻的时间规范化为仅日期（YYYY-MM-DD）。"""
    text = str(value or "").strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return text[:10]
    return text


def valid_date8(value):
    """校验 8 位数字日期（YYYYMMDD，即 4 位年 + 2 位月 + 2 位日）。"""
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{8}", text):
        return False
    try:
        datetime.strptime(text, "%Y%m%d")
        return True
    except ValueError:
        return False


def current_gate():
    settings = read_json("settings.json", {})
    return {
        "gate": str(settings.get("gate") or "南门 01").strip(),
        "direction": str(settings.get("direction") or "入场").strip(),
    }


def upsert_fleet_from_detect(plates, detected_type, gate):
    """识别到车牌后自动记入车辆档案；新车辆状态统一为「需核验」。"""
    fleet = read_json("fleet.json", [])
    existing = {raw_plate_text(item.get("plate", "")): item for item in fleet}
    added = []
    updated = []
    now = now_str()
    for plate in plates:
        text = raw_plate_text(plate)
        if not text or text in {"???", "未识别"}:
            continue
        key = text.upper()
        if key in existing:
            item = existing[key]
            item["lastSeen"] = f"{now} · {gate}"
            updated.append(text)
        else:
            fleet.append({
                "plate": text,
                "type": detected_type or "未知",
                "owner": "未登记车辆",
                "brand": "待完善",
                "permit": "待审批",
                "status": "需核验",
                "lastSeen": f"{now} · {gate}",
                "phone": "未登记",
                "color": "未知",
                "purpose": "识别后自动登记，待人工完善",
            })
            existing[key] = fleet[-1]
            added.append(text)
    if added or updated:
        write_json("fleet.json", fleet)
    return added, updated


def record_pass(plates, gate, direction):
    """识别到车牌后自动写入一条通行记录。"""
    fleet = read_json("fleet.json", [])
    owner_map = {
        raw_plate_text(item.get("plate", "")): str(item.get("owner") or "未登记车辆")
        for item in fleet
    }
    records = read_json("pass_records.json", [])
    now = now_str()
    saved = []
    for plate in plates:
        text = raw_plate_text(plate)
        if not text or text in {"???", "未识别"}:
            continue
        records.append({
            "time": now,
            "gate": gate,
            "plate": text,
            "owner": owner_map.get(text.upper(), "未登记车辆"),
            "direction": direction,
            "result": "识别后自动登记",
        })
        saved.append(text)
    if saved:
        write_json("pass_records.json", records)
    return saved


def best_plate_text(plates, plate_numbers):
    """从识别结果中挑出置信度最高的有效车牌，仅该车牌用于档案与通行记录入库。"""
    numbers = set(plate_numbers or [])
    best = None
    best_conf = -1.0
    for plate in plates:
        text = plate.get("text")
        if not text or text in {"???", "未识别"}:
            continue
        if "?" in raw_plate_text(text):
            continue
        if text not in numbers:
            continue
        conf = float(plate.get("confidence", 0) or 0)
        if conf > best_conf:
            best = text
            best_conf = conf
    if best is None and plate_numbers:
        best = plate_numbers[0]
    return best


def upsert_pending_from_detect(plates):
    """识别成功的车牌自动进入待审核列表（若尚未在任何名单中）。"""
    lists = read_json("access_lists.json", {})
    lists.setdefault("white", [])
    lists.setdefault("black", [])
    lists.setdefault("pending", [])
    known = set()
    for name in ("white", "black", "pending"):
        for item in lists[name]:
            known.add(raw_plate_text(item.get("plate", "")))
    added = []
    today = today_str()
    for plate in plates:
        text = raw_plate_text(plate)
        if not text or text in {"???", "未识别"} or "?" in text or text in known:
            continue
        lists["pending"].append({
            "plate": text,
            "reason": "识别成功，待人工归类",
            "created": today,
        })
        known.add(text)
        added.append(text)
    if added:
        write_json("access_lists.json", lists)
    return added


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/fleet")
def fleet():
    return jsonify(read_json("fleet.json", []))


FLEET_EDITABLE = {"type", "owner", "brand", "permit", "status", "phone", "color", "purpose"}


def normalize_fleet_record(data, plate):
    return {
        "plate": plate,
        "type": str(data.get("type") or "未知").strip(),
        "owner": str(data.get("owner") or "未登记车辆").strip(),
        "brand": str(data.get("brand") or "待完善").strip(),
        "permit": str(data.get("permit") or "待审批").strip(),
        "status": str(data.get("status") or "需核验").strip(),
        "lastSeen": str(data.get("lastSeen") or "无记录").strip(),
        "phone": str(data.get("phone") or "未登记").strip(),
        "color": str(data.get("color") or "未知").strip(),
        "purpose": str(data.get("purpose") or "手动登记").strip(),
    }


@app.route("/api/fleet", methods=["POST"])
def create_fleet_record():
    data = request.get_json(silent=True) or {}
    plate = raw_plate_text(data.get("plate", ""))
    if not plate or plate in {"???", "未识别"}:
        return jsonify({"error": "缺少有效的车牌号"}), 400
    fleet = read_json("fleet.json", [])
    for item in fleet:
        if raw_plate_text(item.get("plate", "")) == plate:
            return jsonify({"error": "该车牌已存在档案，请直接编辑"}), 409
    record = normalize_fleet_record(data, plate)
    fleet.append(record)
    write_json("fleet.json", fleet)
    return jsonify({"ok": True, "record": record}), 201


@app.route("/api/fleet/<plate>", methods=["PUT"])
def update_fleet_record(plate):
    data = request.get_json(silent=True) or {}
    fleet = read_json("fleet.json", [])
    for item in fleet:
        if raw_plate_text(item.get("plate", "")) == raw_plate_text(plate):
            for key, value in data.items():
                if key in FLEET_EDITABLE:
                    item[key] = str(value or "").strip()
            write_json("fleet.json", fleet)
            return jsonify({"ok": True, "record": item})
    return jsonify({"error": "车辆档案不存在"}), 404


@app.route("/api/pass-records")
def pass_records():
    return jsonify(read_json("pass_records.json", []))


@app.route("/api/access-lists", methods=["GET", "POST"])
def access_lists():
    lists = read_json("access_lists.json", {})
    lists.setdefault("white", [])
    lists.setdefault("black", [])
    lists.setdefault("pending", [])

    if request.method == "POST":
        # 人工将待审核车辆归类到白名单 / 黑名单，或移回待审核
        data = request.get_json(silent=True) or {}
        plate = raw_plate_text(data.get("plate", ""))
        target = str(data.get("target") or "").strip()
        if not plate:
            return jsonify({"error": "缺少车牌号"}), 400
        if target not in {"white", "black", "pending"}:
            return jsonify({"error": "目标名单无效"}), 400

        source = None
        record = None
        for name in ("white", "black", "pending"):
            for item in lists[name]:
                if raw_plate_text(item.get("plate", "")) == plate:
                    source = name
                    record = item
                    break
            if record is not None:
                break
        if record is None:
            return jsonify({"error": "列表中不存在该车辆"}), 404
        if source == target:
            return jsonify({"ok": True, "record": record, "from": source, "to": target})
        if any(raw_plate_text(item.get("plate", "")) == plate for item in lists[target]):
            return jsonify({"error": "目标名单中已存在该车辆"}), 409

        lists[source].remove(record)
        if target == "white":
            # 白名单有效期三选项之一：默认 / 具体日期(8位数字) / 长期
            record["expire"] = record.get("expire") or "默认"
            record.pop("created", None)
        else:  # black / pending
            # 加入时间仅到日期，不精确到时刻
            record["created"] = date_only(record.get("created")) or today_str()
            record.pop("expire", None)
        lists[target].append(record)
        write_json("access_lists.json", lists)
        return jsonify({"ok": True, "record": record, "from": source, "to": target})

    return jsonify(lists)


@app.route("/api/access-lists/<plate>", methods=["PUT"])
def update_access_vehicle(plate):
    """人工补充/编辑名单中车辆的信息（用途说明、日期字段）。"""
    data = request.get_json(silent=True) or {}
    lists = read_json("access_lists.json", {})
    lists.setdefault("white", [])
    lists.setdefault("black", [])
    lists.setdefault("pending", [])
    target = None
    record = None
    for name in ("white", "black", "pending"):
        for item in lists[name]:
            if raw_plate_text(item.get("plate", "")) == raw_plate_text(plate):
                target = name
                record = item
                break
        if record is not None:
            break
    if record is None:
        return jsonify({"error": "列表中不存在该车辆"}), 404

    reason = str(data.get("reason") or "").strip()
    if reason:
        record["reason"] = reason
    if target == "white":
        expire = str(data.get("expire") or "").strip()
        if expire:
            if expire not in {"默认", "长期"} and not valid_date8(expire):
                return jsonify({"error": "有效期格式错误：默认 / 长期 / 8位数字（YYYYMMDD）"}), 400
            record["expire"] = expire
    else:
        created = str(data.get("created") or "").strip()
        if created:
            record["created"] = date_only(created)
    write_json("access_lists.json", lists)
    return jsonify({"ok": True, "record": record, "list": target})


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

    auto = {"passRecordSaved": False, "fleetAdded": [], "fleetUpdated": [], "pendingAdded": []}
    recognized = result["summary"].get("plateNumbers", [])
    # 只登记置信度最高的一个车牌，避免正确识别与误识别同时入库
    primary_plate = best_plate_text(result["plates"], recognized)
    if primary_plate:
        gate_info = current_gate()
        auto["fleetAdded"], auto["fleetUpdated"] = upsert_fleet_from_detect(
            [primary_plate], result.get("vehicleType"), gate_info["gate"]
        )
        auto["passRecordSaved"] = bool(
            record_pass([primary_plate], gate_info["gate"], gate_info["direction"])
        )
        # 识别成功的车牌自动进入待审核列表（若尚未在任何名单中）
        auto["pendingAdded"] = upsert_pending_from_detect([primary_plate])

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
        "primaryPlate": primary_plate or "",
        "auto": auto,
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
