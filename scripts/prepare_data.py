"""
Kịch bản chuẩn bị dữ liệu MRI 2D (Giai đoạn 1):
1. Chế độ `--download-real`: Tự động tải và giải nén bộ dữ liệu MRI 2D THẬT chuẩn quốc tế:
   - Bộ dữ liệu 1: Figshare Brain Tumor Dataset (Jun Cheng et al. - 3.064 ảnh MRI 2D có sẵn Mask bác sĩ vẽ tay
     cho 3 loại u: Meningioma, Glioma, Pituitary + Mã bệnh nhân PID).
   - Bộ dữ liệu 2: Sartaj Brain Tumor Classification MRI Dataset (Bổ sung ảnh MRI 2D lớp Bình thường 'no_tumor'
     và tăng cường 4 lớp phân loại).
2. Chế độ `--create-demo`: Tự động sinh bộ dữ liệu ảnh MRI 2D giả lập để kiểm thử nhanh luồng code.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile
import cv2
import h5py
import numpy as np
from tqdm import tqdm

# Thêm thư mục gốc vào sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import CLASS_NAMES, IMAGE_SIZE, PROCESSED_DATA_DIR, RAW_DATA_DIR, SPLITS_DIR


FIGSHARE_FILES = [
    ("brainTumorDataPublic_1-766.zip", "https://ndownloader.figshare.com/files/3381290"),
    ("brainTumorDataPublic_767-1532.zip", "https://ndownloader.figshare.com/files/3381296"),
    ("brainTumorDataPublic_1533-2298.zip", "https://ndownloader.figshare.com/files/3381293"),
    ("brainTumorDataPublic_2299-3064.zip", "https://ndownloader.figshare.com/files/3381302"),
]

SARTAJ_GITHUB_ZIP = (
    "sartaj_brain_mri.zip",
    "https://codeload.github.com/Sartajbhuvaji/Brain-Tumor-Classification-DataSet/zip/refs/heads/master",
)


def download_file_with_progress(url: str, dest_path: Path) -> None:
    """Tải file từ URL với thanh tiến trình tqdm."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 100:
        print(f"[✓] Đã có sẵn file: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return

    print(f"[*] Đang tải: {dest_path.name} từ {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
            while True:
                chunk = response.read(1024 * 64)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))


def clear_processed_dir() -> None:
    """Xóa dữ liệu cũ (demo) trong data/processed trước khi nạp dữ liệu thật."""
    for subdir in ["classification", "segmentation"]:
        target = PROCESSED_DATA_DIR / subdir
        if target.exists():
            shutil.rmtree(target)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DATA_DIR / ".gitkeep").touch()


def read_pid_from_h5(cjdata) -> str:
    """Đọc mã bệnh nhân (Patient ID) từ cấu trúc HDF5 .mat của Figshare."""
    try:
        pid_arr = np.array(cjdata["PID"]).flatten()
        return "".join(chr(int(c)) for c in pid_arr if int(c) > 0)
    except Exception:
        return "unknown"


def download_and_prepare_real_mri_2d(max_figshare_zips: int = 4) -> None:
    """
    Tải bộ dữ liệu thật từ Figshare (3.064 ảnh MRI 2D + Mask bác sĩ vẽ)
    kết hợp bộ dữ liệu Sartaj MRI 2D (bổ sung lớp 'no_tumor' - Không có khối u),
    sau đó tiền xử lý về 256x256 và chia tập Train (70%) / Val (15%) / Test (15%).
    """
    figshare_raw_dir = RAW_DATA_DIR / "figshare"
    figshare_mat_dir = figshare_raw_dir / "mat_files"
    sartaj_raw_dir = RAW_DATA_DIR / "sartaj"
    figshare_mat_dir.mkdir(parents=True, exist_ok=True)
    sartaj_raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tải và giải nén các gói dữ liệu Figshare (.mat)
    for fname, url in FIGSHARE_FILES[:max_figshare_zips]:
        zip_path = figshare_raw_dir / fname
        download_file_with_progress(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(figshare_mat_dir)

    # 2. Tải và giải nén bộ dữ liệu Sartaj (để lấy ảnh MRI 2D thật lớp 'no_tumor')
    sartaj_zip_path = sartaj_raw_dir / SARTAJ_GITHUB_ZIP[0]
    download_file_with_progress(SARTAJ_GITHUB_ZIP[1], sartaj_zip_path)
    sartaj_extract_dir = sartaj_raw_dir / "extracted"
    if not sartaj_extract_dir.exists():
        with zipfile.ZipFile(sartaj_zip_path, "r") as zf:
            zf.extractall(sartaj_extract_dir)

    # 3. Xóa dữ liệu demo giả lập cũ và chuẩn bị thư mục processed mới
    print("[*] Đang làm sạch thư mục data/processed để nạp dữ liệu MRI 2D bệnh nhân thật...")
    clear_processed_dir()

    figshare_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}
    label_idx_map = {"no_tumor": 0, "glioma": 1, "meningioma": 2, "pituitary": 3}

    mat_files = sorted(list(figshare_mat_dir.rglob("*.mat")), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    print(f"[*] Đang trích xuất {len(mat_files)} hồ sơ MRI 2D + Mask từ Figshare...")

    labels_per_split = {"train": {}, "val": {}, "test": {}}
    stats_counts = {s: {c: 0 for c in CLASS_NAMES} for s in ["train", "val", "test"]}

    # Chia theo nhóm bệnh nhân (PID) hoặc định hướng cố định để chống Data Leakage
    patient_to_split = {}
    np.random.seed(42)

    for i, mat_path in enumerate(tqdm(mat_files, desc="Xử lý Figshare .mat")):
        if mat_path.name == "cvind.mat":
            continue
        try:
            with h5py.File(mat_path, "r") as f:
                cjdata = f["cjdata"]
                label_code = int(cjdata["label"][0][0])
                class_name = figshare_map[label_code]
                pid = read_pid_from_h5(cjdata)

                image = np.array(cjdata["image"]).T
                mask = np.array(cjdata["tumorMask"]).T
        except Exception as e:
            print(f"[!] Bỏ qua file lỗi {mat_path.name}: {e}")
            continue

        # Gán mỗi bệnh nhân (PID) vào cố định 1 tập (Train 70% / Val 15% / Test 15%)
        if pid not in patient_to_split:
            r = len(patient_to_split) % 20
            patient_to_split[pid] = "train" if r < 14 else ("val" if r < 17 else "test")
        split = patient_to_split[pid]

        # Chuẩn hóa ảnh MRI 16-bit về 8-bit [0, 255] và kích thước 256x256
        image_8u = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask_8u = (mask > 0).astype(np.uint8) * 255

        image_256 = cv2.resize(image_8u, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        mask_256 = cv2.resize(mask_8u, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        cls_dir = PROCESSED_DATA_DIR / "classification" / split / class_name
        seg_img_dir = PROCESSED_DATA_DIR / "segmentation" / split / "images"
        seg_mask_dir = PROCESSED_DATA_DIR / "segmentation" / split / "masks"
        for d in (cls_dir, seg_img_dir, seg_mask_dir):
            d.mkdir(parents=True, exist_ok=True)

        fname = f"{class_name}_fig_{mat_path.stem}.png"
        cv2.imwrite(str(cls_dir / fname), image_256)
        cv2.imwrite(str(seg_img_dir / fname), image_256)
        cv2.imwrite(str(seg_mask_dir / fname), mask_256)

        labels_per_split[split][fname] = label_idx_map[class_name]
        stats_counts[split][class_name] += 1

    # 4. Bổ sung ảnh MRI 2D thật lớp 'no_tumor' (Bình thường) từ bộ Sartaj
    no_tumor_imgs = sorted(list(sartaj_extract_dir.rglob("no_tumor/*.jpg")) + list(sartaj_extract_dir.rglob("no_tumor/*.png")))
    print(f"[*] Đang xử lý {len(no_tumor_imgs)} ảnh MRI 2D thật lớp 'no_tumor' (Não bình thường)...")

    empty_mask_256 = np.zeros(IMAGE_SIZE, dtype=np.uint8)
    for idx, img_path in enumerate(tqdm(no_tumor_imgs, desc="Xử lý ảnh No Tumor")):
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        img_256 = cv2.resize(img_gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        r = idx % 20
        split = "train" if r < 14 else ("val" if r < 17 else "test")

        cls_dir = PROCESSED_DATA_DIR / "classification" / split / "no_tumor"
        seg_img_dir = PROCESSED_DATA_DIR / "segmentation" / split / "images"
        seg_mask_dir = PROCESSED_DATA_DIR / "segmentation" / split / "masks"
        for d in (cls_dir, seg_img_dir, seg_mask_dir):
            d.mkdir(parents=True, exist_ok=True)

        fname = f"no_tumor_sartaj_{idx:04d}.png"
        cv2.imwrite(str(cls_dir / fname), img_256)
        cv2.imwrite(str(seg_img_dir / fname), img_256)
        cv2.imwrite(str(seg_mask_dir / fname), empty_mask_256)

        labels_per_split[split][fname] = 0
        stats_counts[split]["no_tumor"] += 1

    # Lưu file labels.json cho từng tập và thống kê tổng hợp
    for split in ["train", "val", "test"]:
        lbl_path = PROCESSED_DATA_DIR / "segmentation" / split / "labels.json"
        lbl_path.write_text(json.dumps(labels_per_split[split], indent=2), encoding="utf-8")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SPLITS_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(stats_counts, indent=2), encoding="utf-8")

    print("\n================ THỐNG KÊ BỘ DỮ LIỆU MRI 2D THẬT ĐÃ CHUẨN HÓA (256x256) ================")
    for split in ["train", "val", "test"]:
        total_s = sum(stats_counts[split].values())
        print(f"  - Tập {split.upper():5s}: Tổng {total_s:4d} ảnh | Chi tiết: {stats_counts[split]}")
    print(f"[+] Đã lưu thống kê phân bố dữ liệu tại: {summary_path}")


def generate_synthetic_mri_2d_dataset(num_samples_per_class: int = 25) -> None:
    """Tạo bộ dữ liệu ảnh MRI não 2D giả lập phục vụ kiểm thử nhanh luồng code."""
    print(f"[*] Đang khởi tạo dữ liệu MRI 2D mẫu tại: {PROCESSED_DATA_DIR}")
    clear_processed_dir()
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

                center = (w // 2 + np.random.randint(-6, 6), h // 2 + np.random.randint(-6, 6))
                axes = (int(w * 0.36), int(h * 0.42))
                cv2.ellipse(img, center, axes, 0, 0, 360, 190, 4)
                inner_axes = (axes[0] - 8, axes[1] - 8)
                cv2.ellipse(img, center, inner_axes, 0, 0, 360, np.random.randint(75, 105), -1)

                if class_name != "no_tumor":
                    tx = center[0] + np.random.randint(-30, 30)
                    ty = center[1] + np.random.randint(-30, 30)
                    r1, r2 = np.random.randint(14, 24), np.random.randint(14, 24)
                    cv2.ellipse(img, (tx, ty), (r1, r2), 0, 0, 360, 220, -1)
                    cv2.ellipse(mask, (tx, ty), (r1, r2), 0, 0, 360, 255, -1)

                file_name = f"{class_name}_{i:03d}.png"
                cv2.imwrite(str(class_dir / file_name), img)
                cv2.imwrite(str(seg_img_dir / file_name), img)
                cv2.imwrite(str(seg_mask_dir / file_name), mask)
                labels_map[file_name] = label_idx

        with open(PROCESSED_DATA_DIR / "segmentation" / split / "labels.json", "w", encoding="utf-8") as f:
            json.dump(labels_map, f, indent=2)

    print("[+] Hoàn tất tạo dữ liệu MRI 2D mẫu!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu MRI 2D cho đề tài NCKH")
    parser.add_argument("--download-real", action="store_true", help="Tải và giải nén bộ dữ liệu MRI 2D thật (Figshare + Sartaj)")
    parser.add_argument("--max-zips", type=int, default=4, help="Số gói zip Figshare cần tải (1 đến 4, mặc định 4 = đủ 3064 ảnh)")
    parser.add_argument("--create-demo", action="store_true", help="Tạo dữ liệu MRI 2D mẫu để kiểm thử nhanh")
    parser.add_argument("--samples-per-class", type=int, default=30, help="Số ảnh mẫu mỗi lớp khi tạo demo")
    args = parser.parse_args()

    if args.download_real:
        download_and_prepare_real_mri_2d(max_figshare_zips=args.max_zips)
    else:
        generate_synthetic_mri_2d_dataset(args.samples_per_class)
