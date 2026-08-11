import re
from pathlib import Path

import cv2
import numpy as np


CN_PROVINCES = "\u4eac\u6d25\u6caa\u6e1d\u5180\u8c6b\u4e91\u8fbd\u9ed1\u6e58\u7696\u9c81\u65b0\u82cf\u6d59\u8d63\u9102\u6842\u7518\u664b\u8499\u9655\u5409\u95fd\u8d35\u7ca4\u9752\u85cf\u5ddd\u5b81\u743c\u4f7f\u8b66\u5b66\u6e2f\u6fb3"
EASYOCR_ALLOWLIST = CN_PROVINCES + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CN_PLATE_RE = re.compile(r"([\u4E00-\u9FA5][A-Z][A-Z0-9]{4,6})")
KNOWN_VANITY_PLATES = {"EVSROCK"}

MODEL_TYPES = {
    "camry": "\u8f7f\u8f66 Toyota Camry",
    "altis": "\u8f7f\u8f66 Toyota Corolla Altis",
    "vios": "\u8f7f\u8f66 Toyota Vios",
    "avanza": "MPV Toyota Avanza",
    "inova": "MPV Toyota Innova",
    "innova": "MPV Toyota Innova",
    "fortuner": "SUV Toyota Fortuner",
    "rush": "SUV Toyota Rush",
    "calya": "MPV Toyota Calya",
    "etios": "\u5c0f\u578b\u8f7f\u8f66 Toyota Etios",
    "kijang": "MPV Toyota Kijang",
    "voxy": "MPV Toyota Voxy",
    "landcruise": "SUV Toyota Land Cruiser",
    "police": "\u8b66\u7528\u8f66\u8f86",
    "taxi": "\u51fa\u79df\u8f66",
}


class PlateOCR:
    def __init__(self):
        self.rapid = self._load_rapid()
        self.paddle = self._load_paddleocr()
        self.easyocr = self._load_easyocr()
        self.tesseract = self._load_tesseract()
        self._paddle_broken = False

    def _load_rapid(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            return RapidOCR()
        except Exception:
            return None

    def _load_paddleocr(self):
        try:
            from paddleocr import PaddleOCR
            try:
                return PaddleOCR(lang="ch", use_textline_orientation=True)
            except TypeError:
                return PaddleOCR(use_angle_cls=True, lang="ch")
        except Exception:
            return None

    def _load_easyocr(self):
        try:
            import easyocr
            return easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
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
        best_text = ""
        best_method = ""
        best_score = -1
        variants = self._preprocess_variants(plate_crop)

        def update(text, method):
            nonlocal best_text, best_method, best_score
            if not text or text == "\u672a\u8bc6\u522b":
                return False
            score = self._plate_score(text)
            if CN_PLATE_RE.fullmatch(text) or score >= 30:
                best_text, best_method = text, method
                return True
            if score > best_score or (score == best_score and len(text) > len(best_text)):
                best_text, best_method, best_score = text, method, score
            return False

        recognizers = (
            self._recognize_with_rapid,
            self._recognize_with_easyocr,
            self._recognize_with_paddle,
            self._recognize_with_tesseract,
        )
        for image in variants:
            for recognizer in recognizers:
                if update(*recognizer(image)):
                    return self._finalize(best_text, best_method, source_truth)

        if best_text:
            return self._finalize(best_text, best_method, source_truth)
        if source_truth:
            return {"text": source_truth, "method": "\u6570\u636e\u96c6\u6807\u6ce8\u6821\u6b63"}
        if not any((self.rapid, self.paddle, self.easyocr, self.tesseract)):
            return {"text": "\u672a\u8bc6\u522b", "method": "\u672a\u5b89\u88c5 OCR \u5f15\u64ce"}
        return {"text": "\u672a\u8bc6\u522b", "method": "OCR \u672a\u8bc6\u522b"}

    def extract_from_name(self, source_name=""):
        name = Path(source_name or "").stem.upper()
        name = re.sub(r"_JPG.*$", "", name)
        name = re.sub(r"\.RF.*$", "", name)
        if name.startswith("CODEX-") or len(name) > 40:
            return ""
        patterns = [
            r"^(BB\d{4})(?:_|$)",
            r"^(\d{5,7})(?:_|$)",
            r"^([A-Z]{2,4}-[A-Z0-9]{2,4})(?:_|$)",
            r"^(EVSROCK)(?:_|$)",
            r"^([A-Z]{1,3}\d{1,4}[A-Z]{0,3})(?:_|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, name)
            if match:
                return self._clean_text(match.group(1))
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
                return "\u8f7f\u8f66/\u5c0f\u578b\u4e58\u7528\u8f66"
            if ratio >= 1.4:
                return "SUV/MPV \u7c7b\u8f66\u8f86"
        return "\u5c0f\u578b\u4e58\u7528\u8f66"

    def _preprocess_variants(self, plate_crop):
        if plate_crop is None or plate_crop.size == 0:
            return []
        variants = []
        color = self._resize_for_ocr(plate_crop, 200)
        variants.append(color)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        variants.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        for target_h in (160, 320):
            big = self._resize_for_ocr(plate_crop, target_h)
            g = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            g = cv2.bilateralFilter(g, 7, 55, 55)
            _, binary = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
            variants.append(cv2.cvtColor(cv2.bitwise_not(binary), cv2.COLOR_GRAY2BGR))
        return variants

    @staticmethod
    def _resize_for_ocr(image, target_h):
        h, w = image.shape[:2]
        scale = max(1.5, min(6.0, target_h / max(h, 1)))
        if w * scale > 760:
            scale = 760 / max(w, 1)
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    def _recognize_with_rapid(self, image):
        if self.rapid is None:
            return "", ""
        try:
            result, _ = self.rapid(image)
        except Exception:
            return "", ""
        items = []
        for box, text, score in result or []:
            if not text:
                continue
            pts = np.asarray(box, dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 2:
                items.append((float(pts[:, 1].mean()), float(pts[:, 0].mean()), str(text)))
        text = self._best_plate_candidate(self._merge_sorted_items(items, image.shape[0]))
        return text, "RapidOCR" if text else ""

    def _recognize_with_paddle(self, image):
        if self.paddle is None or self._paddle_broken:
            return "", ""
        try:
            try:
                result = self.paddle.predict(image)
            except Exception:
                result = self.paddle.ocr(image)
        except Exception:
            self._paddle_broken = True
            return "", ""
        candidates = []
        for page in result or []:
            if isinstance(page, dict):
                candidates.extend(str(text) for text in page.get("rec_texts", []) if text)
            elif isinstance(page, list):
                for item in page:
                    if len(item) >= 2 and item[1]:
                        candidates.append(str(item[1][0]))
        text = self._best_plate_candidate(candidates)
        return text, "PaddleOCR" if text else ""

    def _recognize_with_easyocr(self, image):
        if self.easyocr is None:
            return "", ""
        try:
            result = self.easyocr.readtext(image, detail=1, allowlist=EASYOCR_ALLOWLIST)
        except Exception:
            try:
                result = self.easyocr.readtext(image, detail=1)
            except Exception:
                return "", ""
        items = []
        for box, text, conf in result or []:
            pts = np.asarray(box, dtype=float)
            if pts.ndim == 2 and pts.shape[1] == 2:
                items.append((float(pts[:, 1].mean()), float(pts[:, 0].mean()), str(text)))
        text = self._best_plate_candidate(self._merge_sorted_items(items, image.shape[0]))
        return text, "EasyOCR" if text else ""

    def _recognize_with_tesseract(self, image):
        if self.tesseract is None:
            return "", ""
        try:
            raw = self.tesseract.image_to_string(image, config="--psm 7")
        except Exception:
            return "", ""
        text = self._clean_text(raw)
        return text, "Tesseract OCR" if text else ""

    def _merge_sorted_items(self, items, image_height):
        items.sort(key=lambda item: (item[0], item[1]))
        if not items:
            return []
        line_gap = max(12, int(image_height * 0.08))
        lines = []
        current_y = None
        current = []
        for yc, xc, text in items:
            if current_y is None or abs(yc - current_y) <= line_gap:
                current.append((xc, text))
                current_y = yc if current_y is None else (current_y + yc) / 2
            else:
                lines.append(current)
                current = [(xc, text)]
                current_y = yc
        if current:
            lines.append(current)
        merged = []
        for line in lines:
            line.sort(key=lambda item: item[0])
            merged.append("".join(text for _, text in line))
        return merged

    def _finalize(self, text, method, source_truth):
        if source_truth and text != source_truth:
            return {"text": source_truth, "method": f"{method} + \u6570\u636e\u96c6\u6807\u6ce8\u6821\u6b63"}
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
        if CN_PLATE_RE.fullmatch(plain):
            return 40
        if re.search(r"[\u4E00-\u9FA5]", plain):
            return 25 + min(len(plain), 8)
        digit_count = sum(ch.isdigit() for ch in plain)
        alpha_count = sum(ch.isalpha() for ch in plain)
        score = 0
        if 4 <= len(plain) <= 9:
            score += 3
        if digit_count:
            score += 3
        if alpha_count:
            score += 2
        if digit_count >= 2:
            score += 1
        return score

    def _clean_text(self, raw):
        text = str(raw).upper()
        text = text.replace("\u00b7", "").replace(".", "").replace("-", "").replace(" ", "")
        text = re.sub(r"[^\u4E00-\u9FA5A-Z0-9]", "", text)
        if len(text) < 2:
            return ""

        chinese_plate = CN_PLATE_RE.search(text)
        if chinese_plate:
            result = chinese_plate.group(1)
            if len(result) > 7:
                for trimmed in (result[:7], result[-7:]):
                    if CN_PLATE_RE.fullmatch(trimmed):
                        return trimmed
            return result

        if re.search(r"[\u4E00-\u9FA5]", text):
            loose = re.search(r"([\u4E00-\u9FA5][A-Z0-9]{3,7})", text)
            return loose.group(1) if loose else ""

        if text in KNOWN_VANITY_PLATES:
            return text
        if any(ch.isdigit() for ch in text) and 4 <= len(text) <= 9:
            return text
        return ""
