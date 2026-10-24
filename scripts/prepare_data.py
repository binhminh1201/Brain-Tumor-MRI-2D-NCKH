"""
Kịch bản chuẩn bị dữ liệu MRI 2D (Giai đoạn 1):
1. Chế độ `--create-demo`: Tự động sinh bộ dữ liệu ảnh MRI 2D giả lập (gồm 4 lớp + Mask khoanh vùng u)
   để bạn có thể chạy thử nghiệm toàn bộ code Huấn luyện -> Xuất ONNX -> Giao diện ngay lập tức.
2. Chế độ `--convert-figshare`: Giải nén các file `.mat` từ bộ dữ liệu chuẩn Figshare Brain Tumor 2D
   thành ảnh `.png` và mặt nạ `_mask.png` đồng thời chia tập Train/Val/Test (70/15/15).
"""

import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

# Thêm thư mục gốc vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import CLASS_NAMES, IMAGE_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR


def generate_synthetic_mri_2d_dataset(num_samples_per_class: int = 25) -> None:
    """
    Tạo bộ dữ liệu ảnh MRI não 2D giả lập (kích thước 256x256) phục vụ kiểm thử nhanh luồng nghiên cứu.
    Hình hộp sọ elip 2D + cấu trúc mô não + khối u sáng (với Glioma, Meningioma, Pituitary).
    """
    print(f"[*] Đang khởi tạo dữ liệu MRI 2D mẫu tại: {PROCESSED_DATA_DIR}")
    np.random.seed(42)
    h, w = IMAGE_SIZE

    for split, ratio in [("train", 0.7), ("val", 0.15), ("test", 0.15)]:
        n_cls = max(4, int(num_samples_per_class * ratio))
        cls_root = PROCESSED_DATA_DIR / "classification" / split
        seg_img_dir = PROCESSED_DATA_DIR / "segmentation" / split / "images"
        seg_mask_dir = PROCESSED_DATA_DIR / "segmentation" / split / "masks"

        seg_img_dir.mkdir(parents=True, exist_ok=True)
        seg_mask_dir.mkdir(parents=True, exist_ok=True)
        labels_map = {}

        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_dir = cls_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for i in range(n_cls):
                img = np.zeros((h, w), dtype=np.uint8)
                mask = np.zeros((h, w), dtype=np.uint8)

                # 1. Vẽ hộp sọ và nhu mô não 2D
                center = (w // 2 + np.random.randint(-6, 6), h // 2 + np.random.randint(-6, 6))
                axes = (int(w * 0.36), int(h * 0.42))
                cv2.ellipse(img, center, axes, 0, 0, 360, 190, 4)  # Viền hộp sọ sáng
                inner_axes = (axes[0] - 8, axes[1] - 8)
                cv2.ellipse(img, center, inner_axes, 0, 0, 360, np.random.randint(75, 105), -1)

                # 2. Vẽ não thất giữa (Ventricles) tối màu
                cv2.ellipse(img, (center[0] - 12, center[1]), (10, 26), 12, 0, 360, 35, -1)
                cv2.ellipse(img, (center[0] + 12, center[1]), (10, 26), -12, 0, 360, 35, -1)

                # 3. Nếu thuộc nhóm có khối u -> Vẽ khối u tại vị trí đặc trưng từng loại
                if class_name != "no_tumor":
                    if class_name == "glioma":
                        # U thần kinh đệm: Nằm sâu trong bán cầu não, biên hơi bất đối xứng
                        tx = center[0] + np.random.choice([-35, 35]) + np.random.randint(-10, 10)
                        ty = center[1] + np.random.randint(-30, 30)
                        r1, r2 = np.random.randint(16, 28), np.random.randint(14, 24)
                    elif class_name == "meningioma":
                        # U màng não: Nằm sát rìa màng não/hộp sọ, tròn rõ
                        angle_rad = np.random.uniform(0, 2 * np.pi)
                        tx = int(center[0] + (inner_axes[0] - 16) * np.cos(angle_rad))
                        ty = int(center[1] + (inner_axes[1] - 16) * np.sin(angle_rad))
                        r1 = r2 = np.random.randint(14, 22)
                    else:  # pituitary
                        # U tuyến yên: Nằm vùng trung tâm/đáy sọ
                        tx = center[0] + np.random.randint(-8, 8)
                        ty = center[1] + np.random.randint(25, 45)
                        r1, r2 = np.random.randint(12, 18), np.random.randint(10, 16)

                    intensity = np.random.randint(195, 245)
                    cv2.ellipse(img, (tx, ty), (r1, r2), 0, 0, 360, intensity, -1)
                    cv2.ellipse(mask, (tx, ty), (r1, r2), 0, 0, 360, 255, -1)

                # Thêm nhiễu Gaussian nhẹ giống ảnh cộng hưởng từ thực tế
                noise = np.random.normal(0, 7, (h, w)).astype(np.int16)
                img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                img = cv2.GaussianBlur(img, (3, 3), 0)

                file_name = f"{class_name}_{i:03d}.png"
                cv2.imwrite(str(class_dir / file_name), img)
                cv2.imwrite(str(seg_img_dir / file_name), img)
                cv2.imwrite(str(seg_mask_dir / file_name), mask)
                labels_map[file_name] = label_idx

        with open(PROCESSED_DATA_DIR / "segmentation" / split / "labels.json", "w", encoding="utf-8") as f:
            json.dump(labels_map, f, indent=2)

    print("[+] Hoàn tất tạo dữ liệu MRI 2D mẫu! Sẵn sàng huấn luyện và kiểm thử.")


def convert_figshare_mat_files(raw_mat_dir: Path) -> None:
    """
    Hàm giải nén dữ liệu thực tế từ bộ Figshare Brain Tumor Dataset (các file .mat 2D).
    Cấu trúc file .mat của Figshare:
      - cjdata.image: Ma trận ảnh MRI 2D
      - cjdata.tumorMask: Mặt nạ nhị phân khối u 2D
      - cjdata.label: 1 (Meningioma), 2 (Glioma), 3 (Pituitary)
    """
    import h5py

    mat_files = sorted(list(raw_mat_dir.rglob("*.mat")))
    if not mat_files:
        print(f"[!] Không tìm thấy file .mat nào trong {raw_mat_dir}.")
        return

    print(f"[*] Tìm thấy {len(mat_files)} file .mat. Đang giải nén sang ảnh PNG 2D...")
    figshare_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}
    label_idx_map = {"meningioma": 2, "glioma": 1, "pituitary": 3}

    for i, mat_path in enumerate(mat_files):
        with h5py.File(mat_path, "r") as f:
            label_code = int(f["cjdata"]["label"][0][0])
            class_name = figshare_map[label_code]
            image = np.array(f["cjdata"]["image"]).T
            mask = np.array(f["cjdata"]["tumorMask"]).T

            # Chuẩn hóa ảnh 16-bit về 8-bit [0, 255]
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            mask = (mask > 0).astype(np.uint8) * 255

            image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

            # Chia Train (70%) / Val (15%) / Test (15%)
            mod = i % 20
            split = "train" if mod < 14 else ("val" if mod < 17 else "test")

            cls_dir = PROCESSED_DATA_DIR / "classification" / split / class_name
            seg_img_dir = PROCESSED_DATA_DIR / "segmentation" / split / "images"
            seg_mask_dir = PROCESSED_DATA_DIR / "segmentation" / split / "masks"
            for d in (cls_dir, seg_img_dir, seg_mask_dir):
                d.mkdir(parents=True, exist_ok=True)

            fname = f"{class_name}_{mat_path.stem}.png"
            cv2.imwrite(str(cls_dir / fname), image)
            cv2.imwrite(str(seg_img_dir / fname), image)
            cv2.imwrite(str(seg_mask_dir / fname), mask)

            labels_json_path = PROCESSED_DATA_DIR / "segmentation" / split / "labels.json"
            labels_data = {}
            if labels_json_path.exists():
                labels_data = json.loads(labels_json_path.read_text(encoding="utf-8"))
            labels_data[fname] = label_idx_map[class_name]
            labels_json_path.write_text(json.dumps(labels_data, indent=2), encoding="utf-8")

    print("[+] Giải nén và chia tập dữ liệu Figshare 2D hoàn tất!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu MRI 2D cho đề tài NCKH")
    parser.add_argument("--create-demo", action="store_true", help="Tạo dữ liệu MRI 2D mẫu để kiểm thử nhanh")
    parser.add_argument("--samples-per-class", type=int, default=30, help="Số ảnh mẫu mỗi lớp khi tạo demo")
    parser.add_argument("--convert-figshare", type=str, default="", help="Đường dẫn thư mục chứa file .mat của Figshare")
    args = parser.parse_args()

    if args.convert_figshare:
        convert_figshare_mat_files(Path(args.convert_figshare))
    else:
        generate_synthetic_mri_2d_dataset(args.samples_per_class)
