"""
Module xây dựng kiến trúc Khoanh vùng 2D U-Net & Multi-Task U-Net (Giai đoạn 3 & 4):
1. StandardUNet2D: Kiến trúc U-Net 2D kinh điển (Encoder - Bottleneck - Decoder + Skip Connections).
2. LightweightMultiTaskUNet2D: Kiến trúc U-Net thu gọn kết hợp đồng thời 2 nhánh đầu ra:
   - Nhánh 1 (Segmentation Mask): Tách viền khối u theo từng pixel 2D.
   - Nhánh 2 (Classification Head): Phân loại 4 lớp u ngay từ tầng Bottleneck.
   -> Cực kỳ phù hợp để triển khai trên thiết bị hạn chế (chỉ cần 1 mô hình duy nhất, xuất chuẩn ONNX mượt mà).
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_CLASSES


class DoubleConv2D(nn.Module):
    """Khối cơ bản của U-Net 2D: (Conv2d 3x3 -> BatchNorm2d -> ReLU) x 2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DepthwiseSeparableConv2D(nn.Module):
    """
    Khối tích chập tách chiều 2D (Depthwise Separable Convolution):
    Giúp giảm 70-80% số lượng tham số và phép tính FLOPs so với Conv2d thường,
    là chìa khóa để tối ưu hóa tốc độ suy luận trên CPU/Edge.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise: tích chập riêng trên từng kênh
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise: tích chập 1x1 trộn các kênh
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightDoubleConv2D(nn.Module):
    """Khối tích chập kép siêu nhẹ dùng cho Lightweight U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            DepthwiseSeparableConv2D(in_channels, out_channels),
            DepthwiseSeparableConv2D(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class StandardUNet2D(nn.Module):
    """
    Mạng U-Net 2D tiêu chuẩn (Baseline) dùng để đối sánh dung lượng và tốc độ.
    Số kênh mặc định: [64, 128, 256, 512], Bottleneck = 1024 (~31 triệu tham số).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        f = base_filters
        # Nhánh thu hẹp (Encoder)
        self.inc = DoubleConv2D(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(f * 4, f * 8))

        # Đáy chữ U (Bottleneck)
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(f * 8, f * 16))

        # Nhánh mở rộng (Decoder)
        self.up1 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv2D(f * 16, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv2D(f * 8, f * 4)

        self.up3 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv2D(f * 4, f * 2)

        self.up4 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv2D(f * 2, f)

        # Đầu ra bản đồ phân đoạn (Logits)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bot(x4)

        u1 = self.up1(b)
        u1 = self.conv_up1(torch.cat([u1, x4], dim=1))

        u2 = self.up2(u1)
        u2 = self.conv_up2(torch.cat([u2, x3], dim=1))

        u3 = self.up3(u2)
        u3 = self.conv_up3(torch.cat([u3, x2], dim=1))

        u4 = self.up4(u3)
        u4 = self.conv_up4(torch.cat([u4, x1], dim=1))

        return self.outc(u4)


class LightweightMultiTaskUNet2D(nn.Module):
    """
    Mô hình Đề xuất (Proposed Model) cho đề tài NCKH:
    - Sử dụng tích chập tách chiều (Depthwise Separable Conv) với base_filters=32 giúp giảm >90% dung lượng.
    - Tích hợp Đa nhiệm (Multi-Task):
      + Đầu ra 1 (seg_logits): Khoanh vùng khối u (1 x 256 x 256).
      + Đầu ra 2 (cls_logits): Phân loại dạng u (num_classes) ngay từ tầng Bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 3,
        seg_out_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        base_filters: int = 32,
        enable_classification_head: bool = True,
    ):
        super().__init__()
        self.enable_classification_head = enable_classification_head
        f = base_filters

        # Encoder nhẹ
        self.inc = DoubleConv2D(in_channels, f)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), LightDoubleConv2D(f, f * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), LightDoubleConv2D(f * 2, f * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), LightDoubleConv2D(f * 4, f * 8))

        # Bottleneck
        self.bot = nn.Sequential(nn.MaxPool2d(2), LightDoubleConv2D(f * 8, f * 16))

        # Nhánh Phân loại (Classification Head tại đáy chữ U)
        if self.enable_classification_head:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(f * 16, f * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(f * 4, num_classes),
            )

        # Decoder nhẹ (Dùng Upsample Bilinear + LightDoubleConv để tương thích 100% ONNX/Edge)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_up1 = LightDoubleConv2D(f * 16 + f * 8, f * 8)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_up2 = LightDoubleConv2D(f * 8 + f * 4, f * 4)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_up3 = LightDoubleConv2D(f * 4 + f * 2, f * 2)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_up4 = LightDoubleConv2D(f * 2 + f, f)

        self.outc = nn.Conv2d(f, seg_out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        b = self.bot(x4)

        u1 = self.conv_up1(torch.cat([self.up1(b), x4], dim=1))
        u2 = self.conv_up2(torch.cat([self.up2(u1), x3], dim=1))
        u3 = self.conv_up3(torch.cat([self.up3(u2), x2], dim=1))
        u4 = self.conv_up4(torch.cat([self.up4(u3), x1], dim=1))

        seg_logits = self.outc(u4)

        if self.enable_classification_head:
            cls_logits = self.cls_head(b)
            return seg_logits, cls_logits

        return seg_logits
