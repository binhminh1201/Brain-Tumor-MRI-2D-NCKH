# 🧠 Đề Tài NCKH: Ứng Dụng Thị Giác Máy Tính & Học Sâu Trong Phát Hiện, Khoanh Vùng Và Phân Loại U Não Trên Ảnh MRI 2D

> 📌 **Theo dõi Tiến độ & Nhật ký Nghiên cứu:** Mọi công việc đã làm được, đang triển khai và chưa làm được qua từng phiên làm việc đều được ghi chép chi tiết tại [**TIMELINE_NCKH.md**](./TIMELINE_NCKH.md) để phục vụ trực tiếp cho việc viết Báo cáo tổng kết và Bài báo khoa học cấp trường.

---

## 1. Giới Thiệu Tổng Quan (Từ Cơ Bản Nhất)

Dự án giải quyết bài toán tự động hóa chẩn đoán khối u não trên **ảnh chụp cộng hưởng từ MRI 2D cắt lát** (kích thước chuẩn hóa $256 \times 256$ pixels) với **3 mục tiêu cốt lõi**:
1. **Phát hiện & Phân loại chính xác (>90%):** Phân biệt 4 lớp ảnh MRI 2D gồm:
   - `no_tumor`: Bình thường (Không có khối u)
   - `glioma`: U thần kinh đệm
   - `meningioma`: U màng não
   - `pituitary`: U tuyến yên
2. **Khoanh vùng chi tiết & Giải thích trực quan (XAI):** Tách đường viền khối u theo từng điểm ảnh (Pixel-level Segmentation Mask) bằng mạng **2D U-Net**, tính toán diện tích vùng u (theo pixel và %) và hiển thị bản đồ nhiệt **Grad-CAM 2D**.
3. **Tối ưu hóa cho thiết bị cấu hình hạn chế:** Sử dụng tích chập tách chiều (*Depthwise Separable Convolution 2D*) kết hợp xuất chuẩn trung gian **ONNX (`FP32` / `INT8`)** để chạy suy luận cực nhanh trên CPU thông thường mà không đòi hỏi card đồ họa đắt tiền khi triển khai.

---

## 2. Cấu Trúc Thư Mục Dự Án

```text
ComputerVision/
├── GEMINI.md                    # Quy tắc tự động cập nhật Timeline & Git cho dự án
├── TIMELINE_NCKH.md             # Nhật ký tiến độ chi tiết (Đã làm / Đang làm / Chưa làm)
├── README.md                    # Hướng dẫn cài đặt và vận hành cho cả 2 thành viên
├── requirements.txt             # Danh sách các gói thư viện Python
├── data/
│   ├── raw/                     # Nơi đặt dữ liệu gốc tải về (Figshare / Kaggle)
│   └── processed/               # Dữ liệu ảnh 2D & Mask sau khi chuẩn hóa 256x256
├── src/
│   ├── config.py                # Cấu hình đường dẫn, kích thước ảnh 2D, siêu tham số
│   ├── dataset.py               # Tiền xử lý CLAHE 2D, Augmentation & DataLoader
│   ├── models/
│   │   ├── classifiers.py       # Mạng CNN 2D: BasicCNN2D, VGG16, ResNet50, ResNet18, MobileNetV2
│   │   └── unet_2d.py           # StandardUNet2D & LightweightMultiTaskUNet2D (Đề xuất)
│   └── utils/
│       ├── losses.py            # Hàm lỗi DiceLoss2D, BCEDiceLoss2D, MultiTaskLoss2D
│       ├── metrics.py           # Chỉ số Dice Score, IoU, đo diện tích khối u 2D
│       └── gradcam.py           # Bản đồ nhiệt Grad-CAM 2D & vẽ viền Contour khối u
├── scripts/
│   ├── prepare_data.py          # Tạo dữ liệu 2D demo hoặc giải nén dữ liệu Figshare (.mat)
│   ├── train_cls.py             # Huấn luyện mô hình phân loại CNN 2D
│   ├── train_seg.py             # Huấn luyện mô hình khoanh vùng & đa nhiệm U-Net 2D
│   └── export_and_benchmark_onnx.py # Xuất file .onnx, nén INT8 và xuất bảng CSV Benchmark CPU
├── outputs/
│   ├── checkpoints/             # Nơi lưu trọng số mô hình (.pth, .onnx)
│   ├── figures/                 # Biểu đồ huấn luyện, ma trận nhầm lẫn, ảnh kết quả
│   └── tables/                  # Bảng số liệu CSV phục vụ viết bài báo khoa học
└── app/
    └── main_app.py              # Giao diện nguyên mẫu Web App (Streamlit)
```

---

## 3. Hướng Dẫn Triển Khai Chi Tiết (Dành Cho Nhóm 2 Thành Viên - Không Dùng Docker)

Nhóm chỉ gồm 2 thành viên nên sử dụng trực tiếp **Git + Môi trường ảo Python (`.venv`)** để đồng bộ nhanh, nhẹ và dễ kiểm soát nhất.

### 🖥️ A. Cài Đặt Trên Máy Local Chính (Windows + GPU NVIDIA Quadro T1000)

**Bước 1: Cài đặt Python 3.11 (Chuẩn ổn định nhất cho PyTorch & ONNX)**
Mở PowerShell và chạy:
```powershell
winget install -e --id Python.Python.3.11 --scope user
```
*(Đóng PowerShell và mở lại sau khi cài xong).*

**Bước 2: Khởi tạo và kích hoạt môi trường ảo `.venv`**
```powershell
cd d:\ComputerVision
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
.\.venv\Scripts\Activate.ps1
```

**Bước 3: Cài đặt PyTorch hỗ trợ GPU NVIDIA (CUDA 12.1) & các thư viện**
```powershell
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Bước 4: Kiểm tra nhận diện card đồ họa rời**
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

### 💻 B. Hướng Dẫn Cho Thành Viên Thứ 2 Trong Nhóm (Khi Clone Từ Git Về Máy Khác)

Khi người thứ 2 muốn lấy dự án về máy cá nhân để cùng chạy thử nghiệm, sửa code hoặc làm báo cáo:

**Bước 1: Cài đặt Git và Python 3.11** (tương tự Bước 1 ở trên).

**Bước 2: Tải mã nguồn từ Git về máy**
```powershell
git clone https://github.com/binhminh1201/Brain-Tumor-MRI-2D-NCKH.git
cd Brain-Tumor-MRI-2D-NCKH
```

**Bước 3: Tạo môi trường ảo `.venv` trên máy của thành viên thứ 2**
```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Bước 4: Cài đặt thư viện tùy theo cấu hình máy của thành viên thứ 2**
- **Trường hợp 1 — Máy của bạn cùng nhóm CÓ card rời NVIDIA:**
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ```
- **Trường hợp 2 — Máy của bạn cùng nhóm KHÔNG có card rời (Chỉ dùng CPU/Card onboard):**
  ```powershell
  pip install -r requirements.txt
  ```
  *(Code trong dự án đã có cơ chế tự động phát hiện `cuda` hoặc `cpu`, nên máy chỉ có CPU vẫn chạy được đầy đủ mọi kịch bản và giao diện nguyên mẫu!).*

---

## 4. Quy Trình Chạy Thực Nghiệm Từ Cơ Bản Đến Hoàn Thiện

Sau khi đã kích hoạt môi trường ảo `(.venv)`:

1. **Tạo dữ liệu MRI 2D mẫu để kiểm thử toàn bộ hệ thống ngay lập tức:**
   ```powershell
   python scripts/prepare_data.py --create-demo
   ```
   *(Hoặc khi đã tải bộ dữ liệu Figshare `.mat` vào `data/raw/figshare`: `python scripts/prepare_data.py --convert-figshare data/raw/figshare`)*.

2. **Huấn luyện mô hình Phân loại 2D (Classification):**
   ```powershell
   # Chạy mạng CNN 4 lớp cơ bản:
   python scripts/train_cls.py --model basic_cnn --epochs 10

   # Chạy mạng nhẹ MobileNetV2:
   python scripts/train_cls.py --model mobilenet_v2 --epochs 20
   ```

3. **Huấn luyện mô hình Khoanh vùng & Đa nhiệm 2D (U-Net Segmentation + Classification):**
   ```powershell
   python scripts/train_seg.py --model light_multitask_unet --epochs 20
   ```

4. **Xuất mô hình sang chuẩn `.onnx`, nén `INT8` và đo Benchmark CPU cho Bài báo:**
   ```powershell
   python scripts/export_and_benchmark_onnx.py
   ```
   -> Mở file `outputs/tables/benchmark_results.csv` để lấy bảng số liệu đưa vào báo cáo.

5. **Khởi chạy Hệ thống Nguyên mẫu Trực quan (Web App):**
   ```powershell
   streamlit run app/main_app.py
   ```

---

## 5. Quy Trình Đồng Bộ Git Giữa 2 Thành Viên

- **Trước khi bắt đầu làm việc:** Luôn kéo bản cập nhật mới nhất về máy:
  ```powershell
  git pull
  ```
- **Sau khi kết thúc 1 phiên làm việc:** Cập nhật tiến độ vào `TIMELINE_NCKH.md` và đẩy lên Git:
  ```powershell
  git add -A
  git commit -m "Cap nhat tien do phien lam viec"
  git push
  ```
