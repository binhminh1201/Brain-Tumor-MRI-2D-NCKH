"""
Module tính toán các chỉ số đánh giá (Evaluation Metrics) cho Báo cáo & Bài báo NCKH:
1. Chỉ số Phân loại: Accuracy, Precision, Recall, F1-Score.
2. Chỉ số Khoanh vùng 2D: Dice Similarity Coefficient (DSC), IoU (Jaccard Index).
3. Chỉ số Y khoa trên ảnh 2D: Diện tích khối u (số pixel), Tỷ lệ diện tích u (%), Tâm khối u (x, y).
"""

from typing import Dict
import numpy as np
import torch


@torch.no_grad()
def compute_segmentation_metrics_2d(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """Tính chỉ số Dice Score và IoU (Jaccard) trên batch ảnh 2D."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    dice = (2.0 * intersection + smooth) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth)
    iou = (intersection + smooth) / (union + smooth)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
    }


def analyze_tumor_geometry_2d(binary_mask_2d: np.ndarray) -> Dict[str, float]:
    """
    Phân tích hình học khối u trên mặt nạ 2D (kích thước H x W, giá trị 0 hoặc 1/255):
    - Trả về diện tích theo pixel, tỷ lệ phần trăm so với toàn ảnh, và tọa độ tâm khối u.
    """
    mask_bool = binary_mask_2d > 0
    total_pixels = mask_bool.size
    tumor_pixels = int(np.sum(mask_bool))
    area_ratio_percent = (tumor_pixels / total_pixels) * 100.0

    if tumor_pixels > 0:
        y_indices, x_indices = np.where(mask_bool)
        center_x = float(np.mean(x_indices))
        center_y = float(np.mean(y_indices))
    else:
        center_x, center_y = -1.0, -1.0

    return {
        "tumor_pixels": tumor_pixels,
        "area_ratio_percent": round(area_ratio_percent, 2),
        "center_x": round(center_x, 1),
        "center_y": round(center_y, 1),
    }
