"""
Kịch bản Huấn luyện Mô hình Phân loại U não trên ảnh MRI 2D (Giai đoạn 2):
- Hỗ trợ chọn kiến trúc: 'basic_cnn', 'mobilenet_v2', 'resnet18', 'resnet50', 'vgg16'.
- Tự động lưu checkpoint tốt nhất vào `outputs/checkpoints/best_cls_<model>.pth`.
- Xuất kết quả Accuracy, Precision, Recall, F1-Score trên tập Test.
"""

import argparse
from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLASS_NAMES,
    LEARNING_RATE,
    PROCESSED_DATA_DIR,
    WEIGHT_DECAY,
)
from src.dataset import BrainMRIClassificationDataset
from src.models.classifiers import build_classifier_2d


def evaluate_classifier(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(loader.dataset))
    acc = accuracy_score(all_targets, all_preds) * 100.0
    return avg_loss, acc, all_targets, all_preds


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mạng CNN 2D phân loại u não MRI")
    parser.add_argument("--model", type=str, default="mobilenet_v2", choices=["basic_cnn", "mobilenet_v2", "resnet18", "resnet50", "vgg16"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--no-pretrained", action="store_true", help="Không tải trọng số ImageNet")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị huấn luyện: {device} | Mô hình 2D: {args.model}")

    cls_dir = PROCESSED_DATA_DIR / "classification"
    train_ds = BrainMRIClassificationDataset(cls_dir / "train", augment=True)
    val_ds = BrainMRIClassificationDataset(cls_dir / "val", augment=False)
    test_ds = BrainMRIClassificationDataset(cls_dir / "test", augment=False)

    if len(train_ds) == 0:
        print("[!] Chưa có dữ liệu trong data/processed. Hãy chạy: python scripts/prepare_data.py --create-demo")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = build_classifier_2d(args.model, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt = CHECKPOINT_DIR / f"best_cls_{args.model}.pth"
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss /= len(train_ds)
        val_loss, val_acc, _, _ = evaluate_classifier(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)

    print(f"\n[+] Đã lưu mô hình tốt nhất tại: {best_ckpt} (Val Acc: {best_val_acc:.2f}%)")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    _, test_acc, y_true, y_pred = evaluate_classifier(model, test_loader, criterion, device)
    print(f"[+] KẾT QUẢ TRÊN TẬP TEST ĐỘC LẬP — Accuracy: {test_acc:.2f}%")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
