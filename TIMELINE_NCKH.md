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
- Hoàn tất cài đặt môi trường ảo `.venv` (Python 3.11), `torch-2.5.1+cu121` kết nối trực tiếp với card rời **NVIDIA Quadro T1000** (`CUDA available: True`) và toàn bộ các gói trong `requirements.txt`.
- Bổ sung cấu hình `sys.stdout.reconfigure(encoding="utf-8")` vào [src/config.py](file:///d:/ComputerVision/src/config.py#L9-L12) giúp Terminal Windows hiển thị tiếng Việt có dấu mượt mà.
- Thu thập **Bảng kết quả Thực nghiệm Benchmark Tối ưu hóa Mô hình 2D trên CPU** ([outputs/tables/benchmark_results.csv](file:///d:/ComputerVision/outputs/tables/benchmark_results.csv)):

| Mô hình 2D ($256 \times 256$) | Định dạng | Số tham số (Triệu) | Dung lượng (MB) | Thời gian CPU (ms/ảnh) | Tốc độ (FPS) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Standard 2D U-Net (Baseline Nặng)** | PyTorch / ONNX (FP32) | 31.038 M | 118.37 MB | 478.63 ms | 2.1 FPS |
| **Lightweight Multi-Task 2D U-Net (Đề xuất)** | PyTorch / ONNX (FP32) | **0.990 M** | **3.76 MB** | **36.35 ms** | **27.5 FPS** |
| **Lightweight Multi-Task 2D U-Net [Nén INT8]** | ONNX Quantized (INT8) | **0.990 M** | **1.03 MB** | 148.23 ms | 6.7 FPS |
| **MobileNetV2 2D Classifier** | PyTorch / ONNX (FP32) | 2.229 M | 8.43 MB | **4.86 ms** | **205.9 FPS** |
| **MobileNetV2 2D Classifier [Nén INT8]** | ONNX Quantized (INT8) | 2.229 M | **2.26 MB** | 37.72 ms | 26.5 FPS |
| **Basic 4-Layer 2D CNN** | PyTorch / ONNX (FP32) | 0.390 M | 1.49 MB | 8.38 ms | 119.4 FPS |

---

### 🔹 PHIÊN 3 (Ngày 24/10/2026): Tải, Giải Nén & Chuẩn Hóa Bộ Dữ Liệu MRI 2D Bệnh Nhân Thật (Figshare + Sartaj — 3.564 Ảnh & Mask)

#### 1. Những gì ĐÃ LÀM ĐƯỢC (Completed ✅)
- **Nâng cấp kịch bản [scripts/prepare_data.py](file:///d:/ComputerVision/scripts/prepare_data.py) với chế độ `--download-real`:**
  - Tự động tải 4 gói dữ liệu y khoa chuẩn quốc tế **Figshare Brain Tumor Dataset (Jun Cheng et al., ID: 1512427)** gồm **3.064 hồ sơ bệnh án `.mat` (HDF5)** chụp từ 233 bệnh nhân tại Bệnh viện Nanfang và Bệnh viện Đa khoa Đại học Y Thiên Tân.
  - Tự động tải bổ sung **500 ảnh MRI 2D não bình thường (`no_tumor`)** từ bộ dữ liệu **Sartaj Brain Tumor MRI Classification Dataset**.
- **Tiền xử lý & Chia tập chống rò rỉ dữ liệu theo Mã bệnh nhân (`PID` - Patient-level Split):**
  - Giải nén ma trận ảnh MRI 16-bit sang ảnh `.png` 8-bit chuẩn hóa $256 \times 256$ và mặt nạ khoanh vùng khối u nhị phân (`0` và `255`) do bác sĩ chẩn đoán hình ảnh vẽ tay.
  - Đọc trực tiếp trường `cjdata.PID` (Mã bệnh nhân) trong từng file `.mat` để đảm bảo **tất cả các lát cắt MRI của cùng 1 bệnh nhân chỉ nằm ở duy nhất 1 tập** (`Train`, `Val` hoặc `Test`), loại bỏ hoàn toàn hiện tượng rò rỉ dữ liệu (*Data Leakage*) — điểm cộng rất lớn về phương pháp luận khi viết bài báo khoa học.
- **Bảng Thống kê Phân bố Bộ dữ liệu MRI 2D Thật ([data/splits/dataset_summary.json](file:///d:/ComputerVision/data/splits/dataset_summary.json)):**

| Tập dữ liệu (Split) | Bình thường (`no_tumor`) | U thần kinh đệm (`glioma`) | U màng não (`meningioma`) | U tuyến yên (`pituitary`) | **Tổng số ảnh & Mask 2D** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Huấn luyện (Train ~70%)** | 350 | 1.044 | 536 | 644 | **2.574 ảnh** |
| **Kiểm thử (Val ~15%)** | 75 | 176 | 96 | 138 | **485 ảnh** |
| **Đánh giá độc lập (Test ~15%)** | 75 | 206 | 76 | 148 | **505 ảnh** |
| **TỔNG CỘNG** | **500** | **1.426** | **708** | **930** | **3.564 ảnh & Mask** |

#### 2. Những gì ĐANG TRIỂN KHAI (In Progress 🔄)
- Chuẩn bị bước vào **Giai đoạn 2 & 3**: Huấn luyện chính thức các mô hình Phân loại CNN 2D (`BasicCNN2D`, `MobileNetV2`, `ResNet18`) và mô hình Khoanh vùng đa nhiệm (`LightweightMultiTaskUNet2D`, `StandardUNet2D`) trên **3.564 ảnh MRI 2D bệnh nhân thật** bằng GPU **NVIDIA Quadro T1000**.

#### 3. Những gì CHƯA LÀM ĐƯỢC / KẾ HOẠCH TIẾP THEO (Pending / Backlog ⏳)
- [ ] **Giai đoạn 2 (Huấn luyện Phân loại 2D trên dữ liệu thật):** Chạy `scripts/train_cls.py` trên 3.564 ảnh thật để đạt mục tiêu **Accuracy > 90%** trên tập Test (505 ảnh độc lập), lưu trọng số tốt nhất và xuất bảng Precision, Recall, F1-Score.
- [ ] **Giai đoạn 3 & 4 (Huấn luyện Khoanh vùng & Đa nhiệm 2D U-Net trên dữ liệu thật):** Chạy `scripts/train_seg.py` trên 3.564 cặp ảnh & Mask bác sĩ vẽ để đạt **Dice Score > 80%**, tích hợp đo diện tích u và bản đồ nhiệt Grad-CAM.
- [ ] **Giai đoạn 5 (Xuất trọng số ONNX đã huấn luyện):** Cập nhật file `.onnx` với trọng số đã học từ dữ liệu thật.
- [ ] **Giai đoạn 6 (Nguyên mẫu & Viết Báo cáo):** Chạy giao diện `streamlit run app/main_app.py` chẩn đoán trên ảnh MRI bệnh nhân thật và hoàn thiện Bài báo NCKH cấp trường.

---

## 📊 III. BẢNG TỔNG HỢP TRẠNG THÁI CÁC HẠNG MỤC (MILESTONE TRACKER)

| Giai đoạn | Hạng mục công việc | Trạng thái | Ghi chú / Kết quả |
| :--- | :--- | :---: | :--- |
| **GĐ 0** | Chốt đề tài, mục tiêu, kiến trúc 2D & Khởi tạo Git/Codebase | ✅ **Hoàn thành** | Đã đồng bộ lên GitHub `binhminh1201/Brain-Tumor-MRI-2D-NCKH` |
| **GĐ 1A** | Cài đặt `.venv` (Python 3.11 + CUDA GPU) & Kiểm thử luồng dữ liệu 2D | ✅ **Hoàn thành** | Đã nhận GPU Quadro T1000 & chạy thông suốt toàn bộ code |
| **GĐ 1B** | Tải & Tiền xử lý bộ dữ liệu MRI 2D bệnh nhân thật (Figshare + Sartaj) | ✅ **Hoàn thành** | **3.564 ảnh & Mask 2D thật** (`Train: 2574`, `Val: 485`, `Test: 505`) chia theo `PID` |
| **GĐ 2** | Huấn luyện & Đánh giá các mô hình Phân loại CNN 2D (>90%) trên dữ liệu thật | 🔄 **Sẵn sàng chạy** | Dữ liệu thật đã nằm trong `data/processed/classification` |
| **GĐ 3** | Huấn luyện & Đánh giá mô hình Khoanh vùng U-Net 2D trên dữ liệu thật | 🔄 **Sẵn sàng chạy** | Dữ liệu thật + Mask bác sĩ vẽ đã nằm trong `data/processed/segmentation` |
| **GĐ 4** | Tích hợp Đa nhiệm + Đo diện tích u + Bản đồ nhiệt Grad-CAM | ⏳ **Chưa làm** | Module `gradcam.py` & `metrics.py` đã sẵn sàng |
| **GĐ 5** | Xuất chuẩn ONNX, nén INT8 & Benchmark tốc độ trên CPU | ✅ **Hoàn thành bước 1** | Đã có bảng `outputs/tables/benchmark_results.csv` (nhanh gấp 13.1 lần, nhẹ hơn 96.8%) |
| **GĐ 6** | Chạy thử nghiệm Ứng dụng Nguyên mẫu & Viết Bài báo cấp trường | ⏳ **Chưa làm** | Giao diện `app/main_app.py` đã sẵn sàng |
