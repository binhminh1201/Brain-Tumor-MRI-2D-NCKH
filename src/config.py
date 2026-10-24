"""
File cấu hình trung tâm (Central Configuration) cho đề tài NCKH:
Phát hiện, khoanh vùng và phân loại khối u não trên ảnh MRI 2D.
"""

from pathlib import Path

# 1. Đường dẫn thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# 2. Thông số ảnh MRI 2D chuẩn hóa
# Kích thước 256x256 là tỷ lệ vàng cho ảnh 2D: vừa giữ nét ranh giới u, vừa chạy cực nhanh trên CPU
IMAGE_SIZE = (256, 256)
IN_CHANNELS = 3  # Ảnh MRI xám được nhân bản 3 kênh (RGB) để tận dụng trọng số Pretrained (ImageNet)

# 3. Danh sách 4 lớp phân loại chuẩn (Kaggle + Figshare)
CLASS_NAMES = [
    "no_tumor",     # 0: Bình thường (Không có u)
    "glioma",       # 1: U thần kinh đệm
    "meningioma",   # 2: U màng não
    "pituitary",    # 3: U tuyến yên
]
CLASS_LABELS_VI = {
    "no_tumor": "Bình thường (Không có khối u)",
    "glioma": "U thần kinh đệm (Glioma)",
    "meningioma": "U màng não (Meningioma)",
    "pituitary": "U tuyến yên (Pituitary)",
}
NUM_CLASSES = len(CLASS_NAMES)

# 4. Siêu tham số huấn luyện mặc định (Hyperparameters)
SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS_CLS = 30
NUM_EPOCHS_SEG = 40
