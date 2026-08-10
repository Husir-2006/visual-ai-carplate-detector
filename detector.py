from pathlib import Path

import cv2
import numpy as np
from ocr_engine import PlateOCR

try:
    import torch
    from tiny_yolo import TinyYolo
except Exception:
    torch = None
    TinyYolo = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class CampusVehicleDetector:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.ultralytics_model = self._load_ultralytics_model()
        self.custom_model = self._load_custom_model()
        self.yolo = self._load_onnx_model()
        self.plate_cascade = self._load_plate_cascade()
        self.ocr = PlateOCR()

    def _load_ultralytics_model(self):
        if YOLO is None:
            return None
        for name in ("yolov8_plate.pt", "best.pt"):
            model_path = self.model_dir / name
            if model_path.exists():
                try:
                    return YOLO(str(model_path))
                except Exception:
                    continue
        return None

    def _load_custom_model(self):
        model_path = self.model_dir / "tiny_plate_detector.pt"
        if torch is None or TinyYolo is None or not model_path.exists():
            return None
        checkpoint = torch.load(str(model_path), map_location="cpu")
        classes = checkpoint.get("classes", ["plate", "vehicle"])
        model = TinyYolo(checkpoint.get("grid_size", 10), len(classes))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return {
            "model": model,
            "grid_size": checkpoint.get("grid_size", 10),
            "image_size": checkpoint.get("image_size", 256),
            "classes": classes,
        }

    def _load_onnx_model(self):
        model_path = self.model_dir / "yolov5s.onnx"
        if not model_path.exists():
            return None
        return cv2.dnn.readNetFromONNX(str(model_path))

    def _load_plate_cascade(self):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_russian_plate_number.xml"
        if not cascade_path.exists():
            return None
        cascade = cv2.CascadeClassifier(str(cascade_path))
        return cascade if not cascade.empty() else None

    def detect(self, image, source_name=""):
        ultralytics_detections = self._detect_ultralytics(image)
        if ultralytics_detections:
            vehicles = [item for item in ultralytics_detections if item["type"] == "vehicle"]
            plates = [item for item in ultralytics_detections if item["type"] == "plate"]
            mode = "YOLOv8/Ultralytics"
        elif self.yolo is not None:
            vehicles = self._detect_vehicles_yolo(image)
            mode = "YOLOv5 ONNX"
        else:
            vehicles = self._detect_vehicles_heuristic(image)
            mode = "OpenCV demo"

        model_detections = self._detect_custom(image)
        if model_detections:
            mode = "Trained Tiny-YOLO + " + mode

        for detection in model_detections:
            if detection["type"] == "vehicle" and all(self._iou(detection["box"], item["box"]) < 0.35 for item in vehicles):
                vehicles.append({k: detection[k] for k in ("label", "confidence", "box")})

        if not ultralytics_detections:
            plates = [{k: d[k] for k in ("label", "confidence", "box")} for d in model_detections if d["type"] == "plate"]
            plates.extend(self._detect_plates(image, vehicles))
        plates = self._dedupe(plates, 0.45)[:8]
        for plate in plates:
            crop = self.crop(image, plate["box"])
            ocr_result = self.ocr.recognize(crop, source_name)
            plate["text"] = ocr_result["text"]
            plate["ocrMethod"] = ocr_result["method"]

        if plates and not self._any_vehicle_contains_plate(vehicles, plates):
            h, w = image.shape[:2]
            vehicles = [{"label": "车辆", "confidence": 0.7, "box": [0, 0, w - 1, h - 1]}]

        primary_vehicle_box = vehicles[0]["box"] if vehicles else None
        vehicle_type = self.ocr.infer_vehicle_type(source_name, primary_vehicle_box)

        plate_numbers = self._unique_plate_numbers(plates)
        status = "recognized" if plate_numbers else ("plate-found" if plates else "no-plate")
        message = {
            "recognized": "??????????",
            "plate-found": "??????? OCR ????????",
            "no-plate": "?????????????????????????",
        }[status]

        return {
            "mode": mode,
            "status": status,
            "message": message,
            "vehicles": vehicles,
            "plates": plates,
            "vehicleType": vehicle_type,
            "summary": {
                "vehicleCount": len(vehicles),
                "plateCount": len(plates),
                "avgConfidence": self._avg_confidence(vehicles + plates),
                "plateNumbers": plate_numbers,
            },
        }



    def _any_vehicle_contains_plate(self, vehicles, plates):
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle["box"]
            for plate in plates:
                px1, py1, px2, py2 = plate["box"]
                pcx = (px1 + px2) / 2
                pcy = (py1 + py2) / 2
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    return True
        return False

    def _fallback_plate_from_vehicle(self, image, vehicles):
        height, width = image.shape[:2]
        if vehicles:
            x1, y1, x2, y2 = vehicles[0]["box"]
        else:
            x1, y1, x2, y2 = 0, int(height * 0.25), width, height
        vw = max(1, x2 - x1)
        vh = max(1, y2 - y1)
        plate_w = int(vw * 0.32)
        plate_h = max(18, int(vh * 0.1))
        cx = x1 + vw // 2
        cy = y1 + int(vh * 0.72)
        px1 = max(0, cx - plate_w // 2)
        py1 = max(0, cy - plate_h // 2)
        px2 = min(width - 1, px1 + plate_w)
        py2 = min(height - 1, py1 + plate_h)
        return {"label": "车牌", "confidence": 0.55, "box": [px1, py1, px2, py2]}


    def _detect_ultralytics(self, image):
        if self.ultralytics_model is None:
            return []
        try:
            results = self.ultralytics_model.predict(image, imgsz=640, conf=0.25, verbose=False)
        except Exception:
            return []
        detections = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
                confidence = float(box.conf[0].detach().cpu()) if box.conf is not None else 0.0
                cls_id = int(box.cls[0].detach().cpu()) if box.cls is not None else 0
                raw_label = str(names.get(cls_id, cls_id)).lower()
                is_plate = any(key in raw_label for key in ("plate", "license", "licence", "number")) or cls_id == 0
                label = "车牌" if is_plate else self._label_cn(raw_label)
                detections.append({
                    "label": label,
                    "type": "plate" if is_plate else "vehicle",
                    "confidence": round(confidence, 3),
                    "box": self._clip_xyxy(xyxy, image.shape[1], image.shape[0]),
                })
        return self._dedupe(detections, 0.45)[:12]

    def _detect_custom(self, image):
        if self.custom_model is None:
            return []
        model = self.custom_model["model"]
        grid_size = self.custom_model["grid_size"]
        image_size = self.custom_model["image_size"]
        classes = self.custom_model["classes"]
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = model(tensor)[0]
            obj = torch.sigmoid(pred[0]).numpy()
            box = torch.sigmoid(pred[1:5]).numpy()
            cls_scores = torch.sigmoid(pred[5:]).numpy()

        detections = []
        for gy in range(grid_size):
            for gx in range(grid_size):
                cls_id = int(np.argmax(cls_scores[:, gy, gx]))
                score = float(obj[gy, gx] * cls_scores[cls_id, gy, gx])
                if score < 0.28:
                    continue
                bx, by, bw, bh = box[:, gy, gx]
                cx = (gx + bx) / grid_size
                cy = (gy + by) / grid_size
                x1 = int((cx - bw / 2) * width)
                y1 = int((cy - bh / 2) * height)
                x2 = int((cx + bw / 2) * width)
                y2 = int((cy + bh / 2) * height)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width - 1, x2), min(height - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                label = classes[cls_id]
                detections.append({
                    "label": "车牌" if label == "plate" else "车辆",
                    "type": "plate" if label == "plate" else "vehicle",
                    "confidence": round(score, 3),
                    "box": [x1, y1, x2, y2],
                })
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return self._dedupe(detections, 0.4)[:12]

    def _detect_vehicles_yolo(self, image):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        outputs = self.yolo.forward()
        rows = outputs[0] if outputs.ndim == 3 else outputs
        boxes, confidences, labels = [], [], []
        x_scale, y_scale = width / 640, height / 640
        for row in rows:
            obj_conf = float(row[4])
            class_scores = row[5:]
            class_id = int(np.argmax(class_scores))
            score = obj_conf * float(class_scores[class_id])
            if class_id not in COCO_CLASSES or score < 0.35:
                continue
            cx, cy, w, h = row[:4]
            x = int((cx - w / 2) * x_scale)
            y = int((cy - h / 2) * y_scale)
            boxes.append([x, y, int(w * x_scale), int(h * y_scale)])
            confidences.append(score)
            labels.append(COCO_CLASSES[class_id])
        keep = cv2.dnn.NMSBoxes(boxes, confidences, 0.35, 0.45)
        detections = []
        for index in np.array(keep).flatten() if len(keep) else []:
            x, y, w, h = self._clip_xywh(boxes[index], width, height)
            detections.append({"label": self._label_cn(labels[index]), "confidence": round(float(confidences[index]), 3), "box": [x, y, x + w, y + h]})
        return detections

    def _detect_vehicles_heuristic(self, image):
        height, width = image.shape[:2]
        resized = cv2.resize(image, (960, int(height * 960 / width))) if width > 960 else image
        scale_x = width / resized.shape[1]
        scale_y = height / resized.shape[0]
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 160)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        image_area = resized.shape[0] * resized.shape[1]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / max(h, 1)
            if area < image_area * 0.025 or area > image_area * 0.75:
                continue
            if not 0.8 <= aspect <= 4.8:
                continue
            if h < resized.shape[0] * 0.12 or w < resized.shape[1] * 0.14:
                continue
            if y < resized.shape[0] * 0.12:
                continue
            candidates.append([x, y, w, h, area])
        candidates.sort(key=lambda item: item[4], reverse=True)
        boxes = []
        for x, y, w, h, _ in candidates:
            box = [int(x * scale_x), int(y * scale_y), int((x + w) * scale_x), int((y + h) * scale_y)]
            if all(self._iou(box, old_box) < 0.35 for old_box in boxes):
                boxes.append(box)
            if len(boxes) >= 8:
                break
        return [{"label": "车辆", "confidence": round(0.62 + min(0.28, (box[2] - box[0]) * (box[3] - box[1]) / (width * height)), 3), "box": box} for box in boxes]

    def _detect_plates(self, image, vehicles):
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plates = self._detect_blue_plates(image)
        if plates:
            return self._dedupe(plates, 0.35)[:4]
        if self.plate_cascade is not None:
            detected = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(60, 18))
            for x, y, w, h in detected:
                box = [int(x), int(y), int(x + w), int(y + h)]
                if self._plate_shape_ok(box):
                    plates.append({"label": "??", "confidence": 0.78, "box": box})

        for rx1, ry1, rx2, ry2, weight in self._plate_search_regions(image, vehicles):
            roi = image[ry1:ry2, rx1:rx2]
            for box, confidence in self._find_plate_like_regions(roi):
                x1, y1, x2, y2 = box
                full_box = [rx1 + x1, ry1 + y1, rx1 + x2, ry1 + y2]
                if self._plate_shape_ok(full_box):
                    plates.append({"label": "??", "confidence": round(min(0.95, confidence * weight), 3), "box": full_box})
        return self._dedupe(plates, 0.35)[:8]


    def _detect_blue_plates(self, image):
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([95, 45, 35], dtype=np.uint8)
        upper = np.array([135, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        image_area = height * width
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect = w / max(1, h)
            if area < image_area * 0.00045 or area > image_area * 0.04:
                continue
            if not 2.0 <= aspect <= 5.8:
                continue
            if w < 70 or h < 20:
                continue
            fill = cv2.countNonZero(mask[y:y + h, x:x + w]) / max(1, area)
            if fill < 0.28:
                continue
            pad_x = int(w * 0.08)
            pad_y = int(h * 0.18)
            box = [max(0, x - pad_x), max(0, y - pad_y), min(width - 1, x + w + pad_x), min(height - 1, y + h + pad_y)]
            confidence = round(min(0.96, 0.72 + fill * 0.22), 3)
            candidates.append({"label": "??", "confidence": confidence, "box": box})
        candidates.sort(key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]), reverse=True)
        return candidates[:4]

    def _plate_search_regions(self, image, vehicles):
        height, width = image.shape[:2]
        regions = []
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle["box"]
            vh = max(1, vy2 - vy1)
            regions.append((vx1, max(0, int(vy1 + vh * 0.35)), vx2, vy2, 1.08))
            regions.append((vx1, vy1, vx2, vy2, 1.0))
        regions.extend([
            (0, 0, width, height, 0.92),
            (0, int(height * 0.25), width, height, 0.95),
            (0, int(height * 0.45), width, height, 0.98),
        ])
        clean = []
        for x1, y1, x2, y2, weight in regions:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 >= 50 and y2 - y1 >= 20:
                item = (int(x1), int(y1), int(x2), int(y2), weight)
                if item not in clean:
                    clean.append(item)
        return clean

    def _find_plate_like_regions(self, roi):
        if roi.size == 0:
            return []
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        area_total = roi.shape[0] * roi.shape[1]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            area = w * h
            if area < area_total * 0.0012 or area > area_total * 0.2:
                continue
            if 1.9 <= aspect <= 7.5 and h >= 10 and w >= 42:
                density = cv2.countNonZero(closed[y:y + h, x:x + w]) / max(1, area)
                if density < 0.08:
                    continue
                score = 0.54 + min(0.32, aspect / 22) + min(0.08, density / 4)
                candidates.append(([x, y, x + w, y + h], round(score, 3)))
        candidates.sort(key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]), reverse=True)
        return candidates[:4]

    def draw_result(self, image, result):
        for vehicle in result["vehicles"]:
            self._draw_box(image, vehicle, (42, 171, 85))
        for plate in result["plates"]:
            self._draw_box(image, plate, (31, 119, 255))
        summary = result["summary"]
        numbers = ",".join(summary.get("plateNumbers", [])) or "none"
        text = f"Vehicles: {summary['vehicleCount']}  Plates: {summary['plateCount']}  OCR: {numbers}"
        cv2.rectangle(image, (16, 16), (min(image.shape[1] - 16, 860), 58), (20, 22, 28), -1)
        cv2.putText(image, text, (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def crop(self, image, box):
        x1, y1, x2, y2 = box
        return image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]

    def _draw_box(self, image, item, color):
        x1, y1, x2, y2 = item["box"]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label_text = item.get("text") if item.get("text") and item.get("text") != "未识别" else ("vehicle" if item["label"] != "车牌" else "plate")
        label = f"{label_text} {item['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
        y_label = max(0, y1 - th - 12)
        cv2.rectangle(image, (x1, y_label), (x1 + tw + 14, y_label + th + 12), color, -1)
        cv2.putText(image, label, (x1 + 7, y_label + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    def _clip_xyxy(self, box, width, height):
        x1, y1, x2, y2 = box
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(x1 + 1, min(width - 1, int(x2)))
        y2 = max(y1 + 1, min(height - 1, int(y2)))
        return [x1, y1, x2, y2]

    def _clip_xywh(self, box, width, height):
        x, y, w, h = box
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        w = max(1, min(width - x, w))
        h = max(1, min(height - y, h))
        return x, y, w, h

    def _plate_shape_ok(self, box):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        return h > 8 and w > 35 and 1.8 <= w / max(h, 1) <= 7.2

    def _label_cn(self, label):
        return {"car": "汽车", "bus": "公交车", "truck": "货车", "motorcycle": "摩托车"}.get(label, "车辆")


    def _unique_plate_numbers(self, plates):
        numbers = []
        seen = set()
        for plate in plates:
            text = plate.get("text")
            if not text or text == "未识别" or text in seen:
                continue
            seen.add(text)
            numbers.append(text)
        return numbers

    def _avg_confidence(self, items):
        if not items:
            return 0
        return round(sum(float(item["confidence"]) for item in items) / len(items), 3)

    def _dedupe(self, items, threshold):
        kept = []
        for item in sorted(items, key=lambda x: x["confidence"], reverse=True):
            if all(self._iou(item["box"], old["box"]) < threshold for old in kept):
                kept.append(item)
        return kept

    def _iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union else 0


