"""
Module xây dựng các kiến trúc Mạng Học sâu Phân loại 2D (Giai đoạn 2):
Đi từ cơ bản đến nâng cao để đối sánh trong bài báo NCKH:
1. BasicCNN2D: Mạng CNN 4 lớp tự thiết kế (giúp hiểu rõ bản chất Conv2d -> BatchNorm -> ReLU -> MaxPool).
2. Baseline nặng: VGG16 / ResNet50.
3. Mô hình nhẹ đề xuất (Lightweight): ResNet18 / MobileNetV2 (tối ưu cho thiết bị cấu hình hạn chế).
"""

import torch
import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


class BasicCNN2D(nn.Module):
    """
    Mạng CNN 2D cơ bản gồm 4 khối tích chập (Convolutional Blocks).
    Dành cho bước khởi đầu tìm hiểu cơ chế trích xuất đặc trưng 2D và làm mốc so sánh cơ sở.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            # Khối 1: 256x256 -> 128x128
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Khối 2: 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Khối 3: 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Khối 4: 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Gom đặc trưng toàn cục về (B, 256, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_classifier_2d(
    model_name: str = "mobilenet_v2",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Hàm khởi tạo linh hoạt các kiến trúc phân loại 2D phục vụ thực nghiệm đối sánh.
    Hỗ trợ: 'basic_cnn', 'vgg16', 'resnet50', 'resnet18', 'mobilenet_v2'.
    """
    model_name = model_name.lower()

    if model_name == "basic_cnn":
        return BasicCNN2D(num_classes=num_classes)

    elif model_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(
            f"Model '{model_name}' không thuộc danh sách hỗ trợ: "
            "['basic_cnn', 'vgg16', 'resnet50', 'resnet18', 'mobilenet_v2']"
        )
