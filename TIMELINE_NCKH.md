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
- **Xác lập mục tiêu & phạm vi nghiên cứu:**
  - Chốt hướng nghiên cứu tập trung 100% vào **ảnh MRI 2D** với 4 lớp phân loại (*No tumor, Glioma, Meningioma, Pituitary*) và phân đoạn mặt nạ nhị phân (*Binary Segmentation Mask*).
  - Chốt phương thức làm việc nhóm 2 người: Đồng bộ qua Git + môi trường ảo `.venv` chuẩn (không dùng Docker để giữ sự gọn nhẹ).
- **Khảo sát phần cứng máy Local số 1:**
  - Hệ điều hành: Windows (PowerShell).
  - Phát hiện máy có sẵn GPU rời **NVIDIA Quadro T1000** — đủ khả năng huấn luyện trực tiếp các mô hình ảnh 2D ($256 \times 256$) ngay trên máy cá nhân với tốc độ cao.
- **Xây dựng hoàn chỉnh bộ khung mã nguồn (Codebase) từ cơ bản đến nâng cao tại `d:\ComputerVision`:**
  - [src/config.py](file:///d:/ComputerVision/src/config.py): Thiết lập tham số chuẩn hóa ảnh 2D ($256 \times 256$), nhãn tiếng Anh/tiếng Việt và siêu tham số huấn luyện.
  - [src/dataset.py](file:///d:/ComputerVision/src/dataset.py): Xây dựng bộ nạp dữ liệu `BrainMRIClassificationDataset`, `BrainMRISegmentationDataset`, tích hợp tiền xử lý **CLAHE 2D** và tăng cường dữ liệu (Lật ngang đối xứng bán cầu não, xoay nhẹ $\pm 15^\circ$).
  - [src/models/classifiers.py](file:///d:/ComputerVision/src/models/classifiers.py): Xây dựng mạng `BasicCNN2D` (4 lớp cơ bản để học lý thuyết) và các mạng `VGG16`, `ResNet50` (Baseline nặng), `ResNet18`, `MobileNetV2` (Mô hình nhẹ).
  - [src/models/unet_2d.py](file:///d:/ComputerVision/src/models/unet_2d.py): Xây dựng `StandardUNet2D` (~31M tham số) và mô hình đề xuất `LightweightMultiTaskUNet2D` (kết hợp khoanh vùng Mask 2D và phân loại 4 lớp u với tích chập tách chiều).
  - [src/utils/losses.py](file:///d:/ComputerVision/src/utils/losses.py) & [src/utils/metrics.py](file:///d:/ComputerVision/src/utils/metrics.py): Cài đặt hàm mất mát kết hợp $\mathcal{L}_{BCE} + \mathcal{L}_{Dice}$, độ đo `Dice Score`, `IoU`, và hàm đo diện tích khối u theo pixel.
  - [src/utils/gradcam.py](file:///d:/ComputerVision/src/utils/gradcam.py): Cài đặt giải thuật **Grad-CAM 2D** vẽ bản đồ nhiệt giải thích vùng mô bất thường và vẽ đường viền Contour khối u.
  - [scripts/](file:///d:/ComputerVision/scripts): Hoàn thiện 4 kịch bản tự động: `prepare_data.py`, `train_cls.py`, `train_seg.py`, `export_and_benchmark_onnx.py`.
  - [app/main_app.py](file:///d:/ComputerVision/app/main_app.py): Xây dựng giao diện nguyên mẫu bằng `Streamlit` hiển thị 4 khung chẩn đoán trực quan.
  - Khởi tạo kho chứa Git (`git init`), quy tắc dự án [GEMINI.md](file:///d:/ComputerVision/GEMINI.md) và tài liệu hướng dẫn triển khai cho thành viên thứ 2 trong [README.md](file:///d:/ComputerVision/README.md).

#### 2. Những gì ĐANG TRIỂN KHAI (In Progress 🔄)
- **Thiết lập môi trường Python 3.11 & PyTorch CUDA trên máy Local:**
  - Cài đặt Python 3.11, tạo môi trường ảo `.venv` và cài đặt các gói thư viện trong `requirements.txt` (kèm PyTorch hỗ trợ GPU NVIDIA Quadro T1000).
- **Chạy kiểm thử luồng dữ liệu 2D mẫu (`--create-demo`):**
  - Sinh ảnh MRI 2D mẫu để kiểm tra toàn bộ pipeline từ Huấn luyện -> Xuất ONNX -> Hiển thị lên giao diện Streamlit.

#### 3. Những gì CHƯA LÀM ĐƯỢC / KẾ HOẠCH TIẾP THEO (Pending / Backlog ⏳)
- [ ] **Giai đoạn 1 (Dữ liệu thực tế):** Tải bộ dữ liệu chuẩn quốc tế **Figshare Brain Tumor 2D (3.064 ảnh `.mat` có mask)** và **Kaggle Brain Tumor MRI 2D (7.023 ảnh 4 lớp)** đưa vào `data/raw/` và chạy giải nén sang `data/processed/`.
- [ ] **Giai đoạn 2 (Thực nghiệm Phân loại CNN 2D):** Huấn luyện đối sánh `BasicCNN2D`, `VGG16`, `ResNet50`, `ResNet18`, `MobileNetV2` trên dữ liệu thật, thu thập **Bảng kết quả 1** (đạt Accuracy > 90%, mục tiêu 95%–98%) và vẽ Confusion Matrix.
- [ ] **Giai đoạn 3 & 4 (Thực nghiệm Khoanh vùng & Đa nhiệm 2D U-Net):** Huấn luyện đối sánh `StandardUNet2D` và `LightweightMultiTaskUNet2D`, thu thập **Bảng kết quả 2** (`Dice Score`, `IoU`) và xuất ảnh minh họa Grad-CAM + Contour khối u.
- [ ] **Giai đoạn 5 (Tối ưu hóa ONNX & Đo tốc độ thiết bị hạn chế):** Xuất trọng số đã huấn luyện sang `.onnx` (`FP32` & `INT8`), đo tốc độ suy luận thực tế trên CPU (`ms/ảnh`, `FPS`, dung lượng `MB`) để lập **Bảng kết quả 3**.
- [ ] **Giai đoạn 6 (Hoàn thiện Nguyên mẫu & Viết Báo cáo/Bài báo):** Đóng gói ứng dụng nguyên mẫu chạy mượt và viết hoàn chỉnh Báo cáo tổng kết NCKH + Bài báo khoa học cấp trường dựa trên số liệu và Timeline này.

---

## 📊 III. BẢNG TỔNG HỢP TRẠNG THÁI CÁC HẠNG MỤC (MILESTONE TRACKER)

| Giai đoạn | Hạng mục công việc | Trạng thái | Ghi chú / Kết quả |
| :--- | :--- | :---: | :--- |
| **GĐ 0** | Chốt đề tài, mục tiêu, kiến trúc 2D & Khởi tạo Git/Codebase | ✅ **Hoàn thành** | Đã tạo đầy đủ cấu trúc code tại `d:\ComputerVision` |
| **GĐ 1** | Cài đặt môi trường `.venv` + Chuẩn bị dữ liệu MRI 2D (Demo & Thật) | 🔄 **Đang làm** | Đang cài đặt Python 3.11 + PyTorch CUDA trên máy Local |
| **GĐ 2** | Huấn luyện & Đánh giá các mô hình Phân loại CNN 2D (>90%) | ⏳ **Chưa làm** | Code `scripts/train_cls.py` đã sẵn sàng |
| **GĐ 3** | Huấn luyện & Đánh giá mô hình Khoanh vùng U-Net 2D | ⏳ **Chưa làm** | Code `scripts/train_seg.py` đã sẵn sàng |
| **GĐ 4** | Tích hợp Đa nhiệm + Đo diện tích u + Bản đồ nhiệt Grad-CAM | ⏳ **Chưa làm** | Module `gradcam.py` & `metrics.py` đã sẵn sàng |
| **GĐ 5** | Xuất chuẩn ONNX, nén INT8 & Benchmark tốc độ trên CPU | ⏳ **Chưa làm** | Code `scripts/export_and_benchmark_onnx.py` đã sẵn sàng |
| **GĐ 6** | Chạy thử nghiệm Ứng dụng Nguyên mẫu & Viết Bài báo cấp trường | ⏳ **Chưa làm** | Giao diện `app/main_app.py` đã sẵn sàng |
