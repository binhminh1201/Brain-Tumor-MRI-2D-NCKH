"""
Module xử lý dữ liệu ảnh MRI 2D (Giai đoạn 1):
- Tiền xử lý tăng cường độ tương phản y khoa bằng thuật toán CLAHE (2D).
- Dataset cho bài toán Phân loại (Classification) và Khoanh vùng (2D Segmentation).
- Hỗ trợ sinh dữ liệu MRI 2D giả lập (Demo) để kiểm thử toàn bộ luồng code ngay lập tức.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import CLASS_NAMES, IMAGE_SIZE


def apply_clahe_2d(image_2d: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Áp dụng thuật toán CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lên ảnh MRI 2D giúp làm rõ ranh giới giữa mô khối u và mô não lành.
    """
    if len(image_2d.shape) == 3:
        gray = cv2.cvtColor(image_2d, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_2d.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    # Chuyển về 3 kênh (H, W, 3) để tương thích với các mạng CNN chuẩn
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def preprocess_mri_2d(
    image: np.ndarray,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    use_clahe: bool = True,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Tiền xử lý 1 bức ảnh MRI 2D đầu vào:
    1. Resize về 256x256
    2. Cân bằng sáng cục bộ CLAHE
    3. Chuẩn hóa pixel về [0, 1] và chuyển sang Tensor (3, H, W)
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    if use_clahe:
        processed_rgb = apply_clahe_2d(resized)
    else:
        if len(resized.shape) == 2:
            processed_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            processed_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Chuẩn hóa về khoảng [0.0, 1.0]
    norm_img = processed_rgb.astype(np.float32) / 255.0
    # Đổi trục từ (H, W, C) sang (C, H, W) cho PyTorch
    tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).contiguous()
    return tensor_img, processed_rgb


class BrainMRIClassificationDataset(Dataset):
    """
    Dataset cho bài toán Phân loại 4 lớp u não trên ảnh MRI 2D.
    Cấu trúc thư mục chuẩn:
        root_dir/
            no_tumor/   (*.jpg, *.png)
            glioma/     (*.jpg, *.png)
            meningioma/ (*.jpg, *.png)
            pituitary/  (*.jpg, *.png)
    """

    def __init__(
        self,
        root_dir: Path,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        use_clahe: bool = True,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.augment = augment

        self.samples = []
        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_path in sorted(class_dir.glob(ext)):
                    self.samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_2d(self, image: np.ndarray) -> np.ndarray:
        """Tăng cường dữ liệu 2D cơ bản (Lật ngang, xoay nhẹ +-15 độ)."""
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)  # Lật ngang đối xứng 2 bán cầu não
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REFLECT)
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh MRI tại: {img_path}")

        if self.augment:
            image = self._augment_2d(image)

        tensor_img, _ = preprocess_mri_2d(image, self.image_size, self.use_clahe)
        return tensor_img, label


class BrainMRISegmentationDataset(Dataset):
    """
    Dataset cho bài toán Khoanh vùng khối u 2D (U-Net) kết hợp Phân loại đa nhiệm.
    Cấu trúc thư mục:
        images_dir/ : Chứa ảnh MRI 2D (*.png)
        masks_dir/  : Chứa mặt nạ nhị phân vùng u tương ứng (*.png, nền=0, vùng u=255)
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        labels_dict: Optional[dict] = None,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        use_clahe: bool = True,
        augment: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.labels_dict = labels_dict or {}
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.augment = augment

        self.image_paths = sorted(list(self.images_dir.glob("*.png")) + list(self.images_dir.glob("*.jpg")))

    def __len__(self) -> int:
        return len(self.image_paths)

    def _augment_pair_2d(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Biến đổi đồng thời cả Ảnh MRI và Nhãn Mask để không bị lệch vị trí khối u."""
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = image.shape[:2]
            mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path = self.image_paths[idx]
        mask_path = self.masks_dir / img_path.name

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Thiếu cặp ảnh hoặc mask tại: {img_path.name}")

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self._augment_pair_2d(image, mask)

        tensor_img, _ = preprocess_mri_2d(image, self.image_size, self.use_clahe)

        # Chuẩn hóa Mask về nhị phân 0.0 và 1.0 với kích thước (1, H, W)
        binary_mask = (mask > 127).astype(np.float32)
        tensor_mask = torch.from_numpy(binary_mask).unsqueeze(0)

        label = self.labels_dict.get(img_path.name, 1)
        return tensor_img, tensor_mask, label
