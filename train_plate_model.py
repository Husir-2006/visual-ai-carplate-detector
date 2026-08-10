import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tiny_yolo import TinyYolo


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class YoloDataset(Dataset):
    def __init__(self, root, split, image_size=256, grid_size=10, augment=False):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.grid_size = grid_size
        self.augment = augment
        image_dir = self.root / split / "images"
        self.images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.root / self.split / "labels" / f"{image_path.stem}.txt"
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.augment:
            image = self._augment_image(image)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        target = torch.zeros(7, self.grid_size, self.grid_size)
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = map(float, parts[:5])
                cls = int(cls)
                if cls not in (0, 1):
                    continue
                gx = min(self.grid_size - 1, max(0, int(cx * self.grid_size)))
                gy = min(self.grid_size - 1, max(0, int(cy * self.grid_size)))
                target[0, gy, gx] = 1.0
                target[1, gy, gx] = cx * self.grid_size - gx
                target[2, gy, gx] = cy * self.grid_size - gy
                target[3, gy, gx] = bw
                target[4, gy, gx] = bh
                target[5 + cls, gy, gx] = 1.0
        return image, target

    def _augment_image(self, image):
        image = image.astype(np.float32)
        if random.random() < 0.75:
            alpha = random.uniform(0.78, 1.24)
            beta = random.uniform(-22, 22)
            image = image * alpha + beta
        if random.random() < 0.35:
            noise = np.random.normal(0, random.uniform(2, 9), image.shape)
            image = image + noise
        image = np.clip(image, 0, 255).astype(np.uint8)
        if random.random() < 0.25:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        return image


def detection_loss(pred, target):
    obj_target = target[:, 0]
    obj_pred = pred[:, 0]
    mask = obj_target > 0.5
    obj_loss = F.binary_cross_entropy_with_logits(obj_pred, obj_target)
    noobj_loss = F.binary_cross_entropy_with_logits(obj_pred[~mask], obj_target[~mask]) if (~mask).any() else 0
    if mask.any():
        pred_box = torch.sigmoid(pred[:, 1:5].permute(0, 2, 3, 1)[mask])
        true_box = target[:, 1:5].permute(0, 2, 3, 1)[mask]
        box_loss = F.smooth_l1_loss(pred_box, true_box)
        pred_cls = pred[:, 5:].permute(0, 2, 3, 1)[mask]
        true_cls = target[:, 5:].permute(0, 2, 3, 1)[mask]
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls, true_cls)
    else:
        box_loss = torch.tensor(0.0, device=pred.device)
        cls_loss = torch.tensor(0.0, device=pred.device)
    return obj_loss + 0.4 * noobj_loss + 6.0 * box_loss + cls_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        total += float(detection_loss(model(images), targets).item())
    return total / max(1, len(loader))


def train(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    data_root = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_set = YoloDataset(data_root, "train", args.image_size, args.grid_size, augment=True)
    valid_set = YoloDataset(data_root, "valid", args.image_size, args.grid_size)
    test_set = YoloDataset(data_root, "test", args.image_size, args.grid_size)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=0)

    model = TinyYolo(args.grid_size, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_loss = float("inf")
    history = []

    print(f"device={device} train={len(train_set)} valid={len(valid_set)} test={len(test_set)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = detection_loss(model(images), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
        scheduler.step()
        train_loss = total / max(1, len(train_loader))
        valid_loss = evaluate(model, valid_loader, device)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 6), "valid_loss": round(valid_loss, 6)})
        print(f"epoch={epoch:02d} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "grid_size": args.grid_size,
                    "image_size": args.image_size,
                    "classes": ["plate", "vehicle"],
                    "best_valid_loss": best_loss,
                    "dataset": str(data_root),
                    "history": history,
                },
                output_path,
            )

    checkpoint = torch.load(str(output_path), map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_loss = evaluate(model, test_loader, device)
    checkpoint["test_loss"] = test_loss
    checkpoint["history"] = history
    torch.save(checkpoint, output_path)
    print(f"saved={output_path} best_valid_loss={best_loss:.4f} test_loss={test_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/anpr-model-1")
    parser.add_argument("--output", default="models/tiny_plate_detector.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
