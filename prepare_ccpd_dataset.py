import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PROVINCES = [
    "\u7696", "\u6caa", "\u6d25", "\u6e1d", "\u5180", "\u664b", "\u8499", "\u8fbd",
    "\u5409", "\u9ed1", "\u82cf", "\u6d59", "\u4eac", "\u95fd", "\u8d63", "\u9c81",
    "\u8c6b", "\u9102", "\u6e58", "\u7ca4", "\u6842", "\u743c", "\u5ddd", "\u8d35",
    "\u4e91", "\u85cf", "\u9655", "\u7518", "\u9752", "\u5b81", "\u65b0",
]
ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ADS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def stable_split(path, train_ratio=0.82, valid_ratio=0.10):
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < train_ratio:
        return "train"
    if value < train_ratio + valid_ratio:
        return "valid"
    return "test"


def parse_box(token):
    left_top, right_bottom = token.split("_")
    x1, y1 = [int(v) for v in left_top.split("&")]
    x2, y2 = [int(v) for v in right_bottom.split("&")]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def parse_plate_text(token):
    values = [int(v) for v in token.split("_")]
    if len(values) < 2:
        return ""
    chars = []
    chars.append(PROVINCES[values[0]] if values[0] < len(PROVINCES) else "")
    chars.append(ALPHABETS[values[1]] if values[1] < len(ALPHABETS) else "")
    for value in values[2:]:
        chars.append(ADS[value] if value < len(ADS) else "")
    return "".join(chars)


def parse_ccpd_filename(image_path):
    parts = image_path.stem.split("-")
    if len(parts) < 5:
        return None
    try:
        x1, y1, x2, y2 = parse_box(parts[2])
        plate_text = parse_plate_text(parts[4])
    except Exception:
        return None
    if not plate_text:
        return None
    return x1, y1, x2, y2, plate_text


def write_yolo_label(label_path, box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    label_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}\n", encoding="utf-8")


def collect_images(source):
    images = []
    for path in source.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTENSIONS and parse_ccpd_filename(path):
            images.append(path)
    return sorted(images)


def prepare(args):
    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")
    if output.exists() and args.clean:
        shutil.rmtree(output)
    for split in ("train", "valid", "test"):
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    images = collect_images(source)
    if args.limit and len(images) > args.limit:
        random.Random(args.seed).shuffle(images)
        images = sorted(images[:args.limit])

    rows = []
    copied = 0
    for index, image_path in enumerate(images):
        parsed = parse_ccpd_filename(image_path)
        if not parsed:
            continue
        x1, y1, x2, y2, plate_text = parsed
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        split = stable_split(image_path)
        out_name = f"ccpd_{index:07d}{image_path.suffix.lower()}"
        out_image = output / split / "images" / out_name
        out_label = output / split / "labels" / f"{Path(out_name).stem}.txt"
        shutil.copy2(image_path, out_image)
        write_yolo_label(out_label, (x1, y1, x2, y2), width, height)
        rows.append([split, out_name, plate_text, x1, y1, x2, y2])
        copied += 1

    (output / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n\nnc: 1\nnames: ['plate']\n",
        encoding="utf-8",
    )
    with (output / "ocr_labels.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "image", "plate", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)
    print(f"CCPD YOLO dataset ready: {output} images={copied}")


def main():
    parser = argparse.ArgumentParser(description="Convert CCPD Chinese license plate dataset to YOLO format.")
    parser.add_argument("--source", default="datasets/CCPD", help="Folder containing extracted CCPD images.")
    parser.add_argument("--output", default="datasets/ccpd_yolo", help="YOLO output folder.")
    parser.add_argument("--limit", type=int, default=50000, help="Limit images for course-project training; set 0 for all.")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--clean", action="store_true", help="Remove old output first.")
    args = parser.parse_args()
    if args.limit == 0:
        args.limit = None
    prepare(args)


if __name__ == "__main__":
    main()
