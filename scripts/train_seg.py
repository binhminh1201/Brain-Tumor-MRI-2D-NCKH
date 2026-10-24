"""
Kịch bản Huấn luyện Mô hình Khoanh vùng Khối u 2D U-Net & Multi-Task U-Net (Giai đoạn 3 & 4):
- Hỗ trợ 2 mô hình đối sánh:
  1. 'standard_unet': U-Net 2D truyền thống (Baseline nặng).
  2. 'light_multitask_unet': U-Net 2D siêu nhẹ kết hợp Đa nhiệm (Vừa khoanh vùng Mask, vừa phân loại 4 lớp u).
"""

import argparse
import json
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import BATCH_SIZE, CHECKPOINT_DIR, LEARNING_RATE, PROCESSED_DATA_DIR, WEIGHT_DECAY
from src.dataset import BrainMRISegmentationDataset
from src.models.unet_2d import LightweightMultiTaskUNet2D, StandardUNet2D
from src.utils.losses import BCEDiceLoss2D, MultiTaskLoss2D
from src.utils.metrics import compute_segmentation_metrics_2d


def load_seg_split(split: str, augment: bool) -> BrainMRISegmentationDataset:
    seg_dir = PROCESSED_DATA_DIR / "segmentation" / split
    labels_path = seg_dir / "labels.json"
    labels_dict = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    return BrainMRISegmentationDataset(
        images_dir=seg_dir / "images",
        masks_dir=seg_dir / "masks",
        labels_dict=labels_dict,
        augment=augment,
    )


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mạng 2D U-Net khoanh vùng & phân loại u não")
    parser.add_argument("--model", type=str, default="light_multitask_unet", choices=["standard_unet", "light_multitask_unet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE * 3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị: {device} | Mô hình Khoanh vùng 2D: {args.model}")

    train_ds = load_seg_split("train", augment=True)
    val_ds = load_seg_split("val", augment=False)
    test_ds = load_seg_split("test", augment=False)

    if len(train_ds) == 0:
        print("[!] Chưa có dữ liệu. Hãy chạy: python scripts/prepare_data.py --create-demo")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    is_multitask = args.model == "light_multitask_unet"
    if is_multitask:
        model = LightweightMultiTaskUNet2D(enable_classification_head=True).to(device)
        criterion = MultiTaskLoss2D(seg_weight=1.0, cls_weight=0.4)
    else:
        model = StandardUNet2D().to(device)
        criterion = BCEDiceLoss2D()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt = CHECKPOINT_DIR / f"best_seg_{args.model}.pth"
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks, labels in train_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()

            if is_multitask:
                seg_logits, cls_logits = model(images)
                loss, _, _ = criterion(seg_logits, cls_logits, masks, labels)
            else:
                seg_logits = model(images)
                loss = criterion(seg_logits, masks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)

        # Đánh giá trên tập Validation
        model.eval()
        val_dice, val_iou, val_correct = 0.0, 0.0, 0
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)
                if is_multitask:
                    seg_logits, cls_logits = model(images)
                    preds_cls = torch.argmax(cls_logits, dim=1)
                    val_correct += (preds_cls == labels).sum().item()
                else:
                    seg_logits = model(images)

                m = compute_segmentation_metrics_2d(seg_logits, masks)
                val_dice += m["dice"] * images.size(0)
                val_iou += m["iou"] * images.size(0)

        val_dice = (val_dice / len(val_ds)) * 100.0
        val_iou = (val_iou / len(val_ds)) * 100.0
        cls_info = f" | Val Cls Acc: {(val_correct / len(val_ds)) * 100.0:.2f}%" if is_multitask else ""

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Loss: {train_loss:.4f} | Val Dice: {val_dice:.2f}% | Val IoU: {val_iou:.2f}%{cls_info}")

        if val_dice >= best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_ckpt)

    print(f"\n[+] Đã lưu mô hình tốt nhất tại: {best_ckpt} (Best Val Dice: {best_dice:.2f}%)")


if __name__ == "__main__":
    main()
