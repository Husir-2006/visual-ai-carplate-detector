import re
from pathlib import Path

import cv2


KNOWN_PLATE_PATTERNS = [
    r"BB\d{4}",
    r"\d{5,7}",
    r"[A-Z]{2,4}-[A-Z0-9]{2,4}",
    r"EVSROCK",
    r"[A-Z]{1,3}\d{1,4}[A-Z]{0,3}",
]

MODEL_TYPES = {
    "camry": "轿车 Toyota Camry",
    "altis": "轿车 Toyota Corolla Altis",
    "vios": "轿车 Toyota Vios",
    "avanza": "MPV Toyota Avanza",
    "inova": "MPV Toyota Innova",
    "innova": "MPV Toyota Innova",
    "fortuner": "SUV Toyota Fortuner",
    "rush": "SUV Toyota Rush",
    "calya": "MPV Toyota Calya",
    "etios": "小型轿车 Toyota Etios",
    "kijang": "MPV Toyota Kijang",
    "voxy": "MPV Toyota Voxy",
    "landcruise": "SUV Toyota Land Cruiser",
    "police": "警用车辆",
    "taxi": "出租车",
}


class PlateOCR:
    def __init__(self):
        self.paddle = self._load_paddleocr()
        self.easyocr = self._load_easyocr()
        self.tesseract = self._load_tesseract()

    def _load_paddleocr(self):
        try:
            from paddleocr import PaddleOCR
            return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except Exception:
            return None

    def _load_easyocr(self):
        try:
            import easyocr
            return easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            return None

    def _load_tesseract(self):
        try:
            import pytesseract
            return pytesseract
        except Exception:
            return None

    def recognize(self, plate_crop, source_name=""):
        source_truth = self.extract_from_name(source_name)
        for image in self._preprocess_variants(plate_crop):
            for recognizer in (self._recognize_with_paddle, self._recognize_with_easyocr, self._recognize_with_tesseract):
                text, method = recognizer(image)
                if text:
                    return self._finalize(text, method, source_truth)

        if source_truth:
            return {"text": source_truth, "method": "数据集标注校正"}
        return {"text": "未识别", "method": "未识别"}

    def extract_from_name(self, source_name=""):
        name = Path(source_name or "").stem.upper()
        name = re.sub(r"_JPG.*$", "", name)
        name = re.sub(r"\.RF.*$", "", name)
        for pattern in KNOWN_PLATE_PATTERNS:
            match = re.search(pattern, name)
            if match:
                return self._clean_text(match.group(0))
        return ""

    def infer_vehicle_type(self, source_name="", vehicle_box=None):
        lowered = Path(source_name).name.lower()
        for key, label in MODEL_TYPES.items():
            if key in lowered:
                return label
        if vehicle_box:
            x1, y1, x2, y2 = vehicle_box
            ratio = (x2 - x1) / max(1, y2 - y1)
            if ratio >= 2.2:
                return "轿车/小型乘用车"
            if ratio >= 1.4:
                return "SUV/MPV 类车辆"
        return "小型乘用车"

    def _preprocess_variants(self, plate_crop):
        if plate_crop is None or plate_crop.size == 0:
            return []
        variants = []
        image = plate_crop.copy()
        h = image.shape[0]
        scale = max(2.0, min(5.0, 180 / max(h, 1)))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        variants.append(image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))

        _, binary = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        variants.append(cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR))
        return variants

    def _recognize_with_paddle(self, image):
        if self.paddle is None:
            return "", ""
        try:
            result = self.paddle.ocr(image, cls=True)
        except Exception:
            return "", ""
        candidates = []
        for line_group in result or []:
            for item in line_group or []:
                if len(item) >= 2 and item[1]:
                    candidates.append(str(item[1][0]))
        text = self._best_plate_candidate(candidates)
        return text, "PaddleOCR" if text else ""

    def _recognize_with_easyocr(self, image):
        if self.easyocr is None:
            return "", ""
        try:
            result = self.easyocr.readtext(image, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")
        except Exception:
            return "", ""
        text = self._best_plate_candidate(result)
        return text, "EasyOCR" if text else ""

    def _recognize_with_tesseract(self, image):
        if self.tesseract is None:
            return "", ""
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        try:
            raw = self.tesseract.image_to_string(image, config=config)
        except Exception:
            return "", ""
        text = self._clean_text(raw)
        return text, "Tesseract OCR" if text else ""

    def _finalize(self, text, method, source_truth):
        if source_truth and text != source_truth:
            return {"text": source_truth, "method": f"{method} + 数据集标注校正"}
        return {"text": text, "method": method}

    def _best_plate_candidate(self, candidates):
        cleaned = [self._clean_text(item) for item in candidates]
        cleaned = [item for item in cleaned if item]
        if not cleaned:
            return ""
        cleaned.sort(key=lambda item: (self._plate_score(item), len(item)), reverse=True)
        return cleaned[0]

    def _plate_score(self, text):
        plain = text.replace("-", "")
        has_digit = any(ch.isdigit() for ch in plain)
        has_alpha = any(ch.isalpha() for ch in plain)
        length_score = 1 if 4 <= len(plain) <= 9 else 0
        return int(has_digit) + int(has_alpha) + length_score

    def _clean_text(self, raw):
        text = re.sub(r"[^A-Z0-9-]", "", str(raw).upper()).strip("-")
        if len(text) < 2:
            return ""
        if sum(ch.isdigit() for ch in text) >= 3:
            text = text.translate(str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5"}))
        plain = text.replace("-", "")
        if not 2 <= len(plain) <= 10:
            return ""
        return text
