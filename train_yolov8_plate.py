from pathlib import Path
import argparse
import shutil


def train(args):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit("请先安装：pip install -r requirements-ai.txt") from exc

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"找不到数据配置：{data_yaml}")

    model = YOLO(args.model)
    result = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project="runs",
        name="plate_yolov8",
        exist_ok=True,
    )

    best = Path(result.save_dir) / "weights" / "best.pt"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy2(best, output)
        print(f"已复制最佳模型到：{output}")
    else:
        print(f"训练完成，但没有找到 best.pt；请检查：{result.save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="训练更稳定的 YOLOv8 车牌检测模型")
    parser.add_argument("--data", default="datasets/combined/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cpu", help="有 NVIDIA 显卡可改成 0")
    parser.add_argument("--output", default="models/yolov8_plate.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
