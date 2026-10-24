"""
Module các Hàm Mất Mát (Loss Functions) cho bài toán Khoanh vùng & Phân loại U não 2D:
- DiceLoss2D: Đo độ chồng lấp trực tiếp của vùng khối u (xử lý mất cân bằng giữa vùng u nhỏ và nền đen).
- BCEDiceLoss2D: Kết hợp Binary Cross Entropy + Dice Loss giúp hội tụ nhanh và viền u sắc nét.
- MultiTaskLoss2D: Hàm lỗi tổng hợp cho mô hình Đa nhiệm (Khoanh vùng + Phân loại).
"""

import torch
import torch.nn as nn


class DiceLoss2D(nn.Module):
    """Hàm mất mát Dice Loss cho mặt nạ nhị phân 2D."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class BCEDiceLoss2D(nn.Module):
    """
    Hàm mất mát kết hợp: L_total = bce_weight * L_BCE + dice_weight * L_Dice.
    Chuẩn mực trong phân đoạn ảnh y khoa 2D.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss2D()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice


class MultiTaskLoss2D(nn.Module):
    """
    Hàm mất mát cho mô hình Đa nhiệm (LightweightMultiTaskUNet2D):
    Kết hợp lỗi Khoanh vùng (BCEDiceLoss) và lỗi Phân loại (CrossEntropyLoss).
    """

    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 0.5):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.seg_loss_fn = BCEDiceLoss2D()
        self.cls_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,
        cls_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        cls_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l_seg = self.seg_loss_fn(seg_logits, seg_targets)
        l_cls = self.cls_loss_fn(cls_logits, cls_targets)
        total = self.seg_weight * l_seg + self.cls_weight * l_cls
        return total, l_seg, l_cls
