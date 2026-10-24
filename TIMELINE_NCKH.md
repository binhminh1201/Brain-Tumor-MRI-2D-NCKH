# 📅 TIMELINE & NHẬT KÝ NGHIÊN CỨU KHOA HỌC (NCKH)
**Tên đề tài:** *Nghiên cứu ứng dụng thị giác máy tính và học sâu trong phát hiện, phân loại khối u não trên ảnh MRI* (Tên rút gọn: *Ứng dụng học sâu trong phát hiện và phân loại u não trên ảnh chụp cộng hưởng từ MRI*)  
**Cấp quản lý:** Đề tài NCKH & Bài báo khoa học cấp Trường Đại học  
**Quy mô nhóm:** 02 thành viên (Triển khai qua Git & Python `.venv`, không dùng Docker)  
**Đối tượng dữ liệu:** 100% Ảnh chụp cộng hưởng từ **MRI 2D cắt lát** ($256 \times 256$ pixels)

---

## 📌 I. TỔNG QUAN NỀN TẢNG KỸ THUẬT (TỪ CƠ BẢN NHẤT - DÙNG ĐỂ VIẾT BÁO CÁO)

Để thuận tiện cho việc viết chương *Cơ sở lý thuyết* và *Phương pháp đề xuất* trong báo cáo tổng kết cuối kỳ, dưới đây là lý giải từ gốc rễ vì sao nhóm lựa chọn từng thành phần kỹ thuật:

1. **Vì sao chọn Ảnh MRI 2D ($256 \times 256$) thay vì 3D?**
   - Trong thực tế khám chữa bệnh ban đầu và mục tiêu chạy trên **thiết bị cấu hình hạn chế**, dữ liệu lát cắt 2D giúp giảm hàng chục lần khối lượng tính toán so với khối 3D Voxel.
   - Kích thước $256 \times 256$ giữ được trọn vẹn ranh giới mô mềm của khối u nhưng chỉ mất vài chục mili-giây để suy luận trên CPU thông thường.
2. **Vì sao dùng thuật toán tiền xử lý CLAHE 2D?**
   - Ảnh MRI thô thường có độ tương phản không đồng đều giữa các máy chụp. Thuật toán *Contrast Limited Adaptive Histogram Equalization (CLAHE)* chia ảnh 2D thành các ô lưới nhỏ ($8 \times 8$) để cân bằng sáng cục bộ, giúp ranh giới giữa vùng u và nhu mô não lành hiện lên sắc nét trước khi đưa vào mạng học sâu.
3. **Vì sao kết hợp mạng CNN 2D và U-Net 2D (Lightweight Multi-Task U-Net)?**
   - **CNN (Convolutional Neural Network):** Giỏi trích xuất đặc trưng để trả lời câu hỏi *"Ảnh này thuộc loại u nào trong 4 loại: Bình thường, Glioma, Meningioma, Pituitary?"* (Mục tiêu Accuracy > 90%).
   - **U-Net 2D:** Có cấu trúc đối xứng hình chữ U (Encoder thu nhỏ ảnh lấy ngữ cảnh - Decoder phóng to ảnh khôi phục vị trí + đường nối tắt *Skip Connections* giữ chi tiết biên) để trả lời câu hỏi *"Khối u nằm ở những pixel nào và diện tích bao nhiêu?"*.
   - **Điểm mới tối ưu của đề tài:** Thay vì dùng phép tích chập thường nặng nề, nhóm sử dụng **Tích chập tách chiều (Depthwise Separable Convolution 2D)** kết hợp nhánh phân loại ngay tại đáy chữ U (*Bottleneck*), tạo ra mô hình đa nhiệm siêu nhẹ.
4. **Vì sao xuất mô hình sang chuẩn trung gian ONNX (`.onnx`)?**
   - Giúp tách rời quá trình huấn luyện (bằng PyTorch trên máy có GPU) và quá trình triển khai (chạy bằng `ONNX Runtime` trên bất kỳ máy tính CPU yếu hoặc thiết bị biên nào mà không cần cài đặt bộ thư viện PyTorch nặng nề).

---

## 🕒 II. NHẬT KÝ TIẾN ĐỘ THEO TỪNG PHIÊN LÀM VIỆC (PROJECT TIMELINE)

### 🔹 PHIÊN 1 (Ngày 24/10/2026): Định hình Đề tài, Thiết kế Kiến trúc 2D & Khởi tạo Mã nguồn Cốt lõi

#### 1. Những gì ĐÃ LÀM ĐƯỢC (Completed ✅)
- Xác lập mục tiêu & phạm vi nghiên cứu: Tập trung 100% vào **ảnh MRI 2D** ($256 \times 256$) với 4 lớp phân loại (*No tumor, Glioma, Meningioma, Pituitary*) và phân đoạn mặt nạ nhị phân (*Binary Segmentation Mask*).
- Khảo sát phần cứng máy Local số 1: Phát hiện GPU rời **NVIDIA Quadro T1000** hỗ trợ CUDA.
- Xây dựng hoàn chỉnh bộ khung mã nguồn tại `d:\ComputerVision` (`src/`, `scripts/`, `app/`) kèm quy tắc dự án [GEMINI.md](file:///d:/ComputerVision/GEMINI.md) và hướng dẫn nhóm 2 người [README.md](file:///d:/ComputerVision/README.md).

---

### 🔹 PHIÊN 2 (Ngày 24/10/2026): Hoàn tất Cài đặt Môi trường Local (`.venv` + CUDA), Kiểm thử Pipeline 2D & Đo Benchmark CPU/ONNX Đầu Tiên

#### 1. Những gì ĐÃ LÀM ĐƯỢC (Completed ✅)
- **Hoàn tất cài đặt môi trường ảo `.venv` (Python 3.11) và thư viện học sâu trên máy Local số 1:**
  - Cài đặt thành công `torch-2.5.1+cu121` và `torchvision-0.20.1+cu121`, xác nhận kết nối trực tiếp với card rời **NVIDIA Quadro T1000** (`CUDA available: True`).
  - Cài đặt đầy đủ toàn bộ các gói trong `requirements.txt` (`opencv-python`, `albumentations`, `scikit-learn`, `onnx`, `onnxruntime`, `streamlit`, `h5py`...).
- **Khắc phục lỗi hiển thị tiếng Việt (`UnicodeEncodeError` trên Windows PowerShell):**
  - Bổ sung cấu hình `sys.stdout.reconfigure(encoding="utf-8")` vào file trung tâm [src/config.py](file:///d:/ComputerVision/src/config.py#L9-L12) giúp mọi kịch bản Python in tiếng Việt có dấu mượt mà trên Windows.
- **Chạy kiểm thử thành công toàn bộ Pipeline 2D (End-to-End Verification):**
  - Sinh bộ dữ liệu MRI 2D mẫu (`scripts/prepare_data.py --create-demo`).
  - Chạy kiểm thử huấn luyện trên GPU `Quadro T1000` cho cả bài toán Phân loại (`scripts/train_cls.py`) và Khoanh vùng đa nhiệm (`scripts/train_seg.py`).
- **Thu thập Bảng kết quả Thực nghiệm Benchmark Tối ưu hóa Mô hình 2D trên CPU ([outputs/tables/benchmark_results.csv](file:///d:/ComputerVision/outputs/tables/benchmark_results.csv)):**
  - Đây là số liệu thực nghiệm cực kỳ đắt giá chứng minh mục tiêu *"tối ưu hóa dung lượng và tốc độ suy luận trên thiết bị cấu hình hạn chế"*:

| Mô hình 2D ($256 \times 256$) | Định dạng | Số tham số (Triệu) | Dung lượng (MB) | Thời gian CPU (ms/ảnh) | Tốc độ (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Standard 2D U-Net (Baseline Nặng)** | PyTorch / ONNX (FP32) | 31.038 M | 118.37 MB | 478.63 ms | 2.1 FPS |
| **Lightweight Multi-Task 2D U-Net (Đề xuất)** | PyTorch / ONNX (FP32) | **0.990 M** | **3.76 MB** | **36.35 ms** | **27.5 FPS** |
| **Lightweight Multi-Task 2D U-Net [Nén INT8]** | ONNX Quantized (INT8) | **0.990 M** | **1.03 MB** | 148.23 ms | 6.7 FPS |
| **MobileNetV2 2D Classifier** | PyTorch / ONNX (FP32) | 2.229 M | 8.43 MB | **4.86 ms** | **205.9 FPS** |
| **MobileNetV2 2D Classifier [Nén INT8]** | ONNX Quantized (INT8) | 2.229 M | **2.26 MB** | 37.72 ms | 26.5 FPS |
| **Basic 4-Layer 2D CNN** | PyTorch / ONNX (FP32) | 0.390 M | 1.49 MB | 8.38 ms | 119.4 FPS |

> 💡 **Nhận xét khoa học rút ra để đưa vào Bài báo NCKH:**
> 1. So với mạng `Standard 2D U-Net` truyền thống ($31.038$ triệu tham số, nặng $118.37\text{ MB}$), mô hình đề xuất **`Lightweight Multi-Task 2D U-Net`** chỉ có **$0.990$ triệu tham số** (giảm **96.8%** lượng tham số), dung lượng bản `FP32` giảm xuống chỉ còn **$3.76\text{ MB}$** (và bản nén `INT8` chỉ còn **$1.03\text{ MB}$** — nhẹ hơn **115 lần**!).
> 2. Về tốc độ suy luận thực tế trên **CPU thường**, mô hình đề xuất chạy chỉ mất **$36.35\text{ ms/ảnh}$** (đạt **$27.5\text{ FPS}$**, tức thời gian thực Real-time), **nhanh gấp $13.1$ lần** so với U-Net truyền thống ($478.63\text{ ms/ảnh}$), trong khi lại giải quyết được đồng thời cả 2 nhiệm vụ (Khoanh vùng + Phân loại).

#### 2. Những gì ĐANG TRIỂN KHAI (In Progress 🔄)
- Chuẩn bị tải và tích hợp bộ dữ liệu MRI 2D thực tế (**Figshare Brain Tumor Dataset** và **Kaggle Brain Tumor MRI Dataset**) vào `data/raw/` để huấn luyện lấy độ chính xác thực tế (>90%).

#### 3. Những gì CHƯA LÀM ĐƯỢC / KẾ HOẠCH TIẾP THEO (Pending / Backlog ⏳)
- [ ] **Giai đoạn 1B (Dữ liệu thật):** Tải bộ dữ liệu thực tế Figshare & Kaggle MRI 2D vào `data/raw/` và giải nén sang `data/processed/`.
- [ ] **Giai đoạn 2 (Huấn luyện Phân loại trên dữ liệu thật):** Huấn luyện `BasicCNN2D`, `ResNet18`, `MobileNetV2` với 30 epochs trên GPU Quadro T1000 để đạt **Accuracy > 90%** (mục tiêu 94%–98%), xuất biểu đồ Learning Curve và Confusion Matrix vào `outputs/figures/`.
- [ ] **Giai đoạn 3 & 4 (Huấn luyện Khoanh vùng & Đa nhiệm trên dữ liệu thật):** Huấn luyện `StandardUNet2D` và `LightweightMultiTaskUNet2D` với 40 epochs để đạt **Dice Score > 80%**, xuất ảnh minh họa Grad-CAM + Contour khối u.
- [ ] **Giai đoạn 6 (Nguyên mẫu & Viết Báo cáo):** Chạy thử nghiệm giao diện `streamlit run app/main_app.py` với trọng số đã huấn luyện trên dữ liệu thật và tổng hợp Bài báo khoa học cấp trường.

---

## 📊 III. BẢNG TỔNG HỢP TRẠNG THÁI CÁC HẠNG MỤC (MILESTONE TRACKER)

| Giai đoạn | Hạng mục công việc | Trạng thái | Ghi chú / Kết quả |
| :--- | :--- | :---: | :--- |
| **GĐ 0** | Chốt đề tài, mục tiêu, kiến trúc 2D & Khởi tạo Git/Codebase | ✅ **Hoàn thành** | Đã tạo đầy đủ cấu trúc code tại `d:\ComputerVision` |
| **GĐ 1A** | Cài đặt `.venv` (Python 3.11 + CUDA GPU) & Kiểm thử luồng dữ liệu 2D Demo | ✅ **Hoàn thành** | Đã nhận GPU Quadro T1000 & chạy thông suốt toàn bộ code |
| **GĐ 1B** | Tải & Tiền xử lý bộ dữ liệu MRI 2D thực tế (Figshare / Kaggle) | 🔄 **Đang làm** | Bước tiếp theo ngay sau khi kiểm thử |
| **GĐ 2** | Huấn luyện & Đánh giá các mô hình Phân loại CNN 2D (>90%) trên dữ liệu thật | ⏳ **Chưa làm** | Đã kiểm thử thành công code `scripts/train_cls.py` |
| **GĐ 3** | Huấn luyện & Đánh giá mô hình Khoanh vùng U-Net 2D trên dữ liệu thật | ⏳ **Chưa làm** | Đã kiểm thử thành công code `scripts/train_seg.py` |
| **GĐ 4** | Tích hợp Đa nhiệm + Đo diện tích u + Bản đồ nhiệt Grad-CAM | ⏳ **Chưa làm** | Module `gradcam.py` & `metrics.py` đã sẵn sàng |
| **GĐ 5** | Xuất chuẩn ONNX, nén INT8 & Benchmark tốc độ trên CPU | ✅ **Hoàn thành bước 1** | Đã có bảng `outputs/tables/benchmark_results.csv` (nhanh gấp 13.1 lần, nhẹ hơn 96.8%) |
| **GĐ 6** | Chạy thử nghiệm Ứng dụng Nguyên mẫu & Viết Bài báo cấp trường | ⏳ **Chưa làm** | Giao diện `app/main_app.py` đã sẵn sàng chạy thử |
