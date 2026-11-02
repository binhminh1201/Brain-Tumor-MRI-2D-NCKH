# 📚 TỔNG HỢP & PHÂN TÍCH CÁC BÀI BÁO KHOA HỌC THAM KHẢO (LITERATURE REVIEW)
*(Tài liệu phục vụ học bản chất theo lời dặn của Giảng viên hướng dẫn và trích dẫn trực tiếp vào Mục "Tổng quan nghiên cứu" & "Tài liệu tham khảo" của Báo cáo NCKH cấp trường)*

---

## 🎯 I. BỨC TRANH TỔNG THỂ: ĐỀ TÀI CỦA NHÓM ĐỨNG Ở ĐÂU TRONG CÁC NGHIÊN CỨU HIỆN NAY?

Theo đúng lời khuyên của Giảng viên hướng dẫn trong buổi họp ngày 03/11/2026: *"Hãy xem 1–2 bài báo đi trước đúng chủ đề của mình để biết họ dùng dữ liệu gì, mô hình của họ sinh ra như thế nào, họ huấn luyện và nén mô hình ra sao"*.

Dưới đây là **5 bài báo khoa học quốc tế chuẩn mực nhất** gắn chặt với từng mảnh ghép trong đề tài của nhóm:
1. **Nhóm bài báo về Kiến trúc Đa nhiệm (U-Net + CNN vừa Khoanh vùng vừa Phân loại trên bộ Figshare 3.064 ảnh 2D).**
2. **Nhóm bài báo về Tối ưu hóa & Nén mô hình (Lightweight & Quantization) cho Thiết bị cấu hình hạn chế.**
3. **Nhóm bài báo gốc về Bộ dữ liệu MRI 2D & Kiến trúc U-Net Y sinh.**
4. **Nhóm bài báo về Học liên kết (Federated Learning) bảo mật dữ liệu bệnh viện cho ảnh MRI U não.**

---

## 📖 II. PHÂN TÍCH CHI TIẾT 5 BÀI BÁO KHOA HỌC CỐT LÕI (BẰNG TIẾNG VIỆT DỄ HIỂU)

### 📄 Bài báo số 1 (Sát nhất với kiến trúc Đa nhiệm của nhóm)
* **Tên bài báo:** *UAC-Net: a unified multi-task network with aggregated contextual attention and cross-layer fusion for brain tumor segmentation and classification*
* **Tác giả:** Chuanqi Cai, Jincan Zhang, Dongyan Zhang, Wenna Chen, Xinghua Tan, Ganqin Du, Hongwei Jiang
* **Nơi công bố:** Tạp chí *Frontiers in Neuroscience* (Tháng 10/2026)
* **🔗 Link đọc Full-text miễn phí:** https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1986322/full
* **🔗 Link DOI:** https://doi.org/10.3389/fnins.2026.1986322
* **1. Họ giải quyết vấn đề gì?**
  * Trước đây, để vừa **khoanh vùng khối u (Segmentation)** vừa **phân loại loại u (Classification)** trên ảnh MRI 2D, người ta phải huấn luyện 2 mạng riêng biệt (1 mạng U-Net + 1 mạng CNN). Việc chạy 2 mạng riêng gây tốn gấp đôi bộ nhớ và thời gian tính toán, đồng thời bỏ phí mối liên hệ giữa hình dáng vị trí khối u và loại khối u.
* **2. Họ dùng dữ liệu gì?**
  * Sử dụng chính bộ **Figshare Brain Tumor Dataset (3.064 ảnh MRI 2D của 233 bệnh nhân)** gồm 3 loại u (*Glioma, Meningioma, Pituitary*) kèm mặt nạ Mask bác sĩ vẽ — **giống 100% bộ dữ liệu trong thư mục `data/` của nhóm mình!**
* **3. Mô hình của họ hoạt động như thế nào?**
  * Thiết kế một mạng chữ U thống nhất (**Unified Multi-Task U-Net**):
    * **Nhánh Encoder (Bên trái chữ U):** Trích xuất các đặc trưng (*features*) từ ảnh MRI 2D.
    * **Nhánh Phân loại (Classification Head ở đáy chữ U):** Gom đặc trưng ở tầng sâu nhất để dự đoán nhãn loại u.
    * **Nhánh Khoanh vùng (Decoder bên phải chữ U):** Giải tích chập/phóng to đặc trưng kết hợp đường nối tắt (*Skip Connections*) để vẽ mặt nạ nhị phân từng pixel của khối u.
* **4. Nhóm mình học và kế thừa được gì từ bài này?**
  * Đây chính là cơ sở khoa học vững chắc giải thích vì sao nhóm thiết kế mô hình [`LightweightMultiTaskUNet2D`](file:///d:/ComputerVision/src/models/unet_2d.py) gộp cả Khoanh vùng và Phân loại vào chung 1 mạng duy nhất!

---

### 📄 Bài báo số 2 (Sát nhất với mục tiêu "Nén mô hình & Tìm Điểm Vàng trên Thiết bị hạn chế" thầy hướng dẫn dặn)
* **Tên bài báo:** *Quantized Machine Learning Models for Medical Imaging in Low-Resource Healthcare Settings*
* **Tác giả:** Sumanth Meenan Kanneti, Aryan Shah
* **Nơi công bố:** *arXiv Preprint* (Tháng 05/2026)
* **🔗 Link trang chính (arXiv):** https://arxiv.org/abs/2605.19207
* **🔗 Link đọc/tải trực tiếp PDF:** https://arxiv.org/pdf/2605.19207
* **🔗 Bài tham khảo bổ sung (Lightweight U-Net trên MDPI Bioengineering):** https://www.mdpi.com/2306-5354/11/12/1302
* **1. Họ giải quyết vấn đề gì?**
  * Các bệnh viện tuyến cơ sở hoặc phòng khám nghèo tài nguyên (*Low-resource healthcare settings*) không có máy chủ GPU đắt tiền. Các mô hình học sâu thông thường quá nặng nên không thể chạy trực tiếp trên máy tính CPU yếu hoặc thiết bị biên tại phòng khám.
* **2. Họ dùng dữ liệu và mô hình gì?**
  * Bài toán phân loại **4 lớp ảnh MRI u não 2D** (*Glioma, Meningioma, Pituitary, và Healthy/No tumor* — giống hệt 4 lớp của nhóm mình).
  * Sử dụng kiến trúc mạng tích chập tách chiều siêu nhẹ **MobileNetV2** làm khung xương sống (*Backbone*).
* **3. Họ thực hiện "Nén mô hình" (Quantization) như thế nào và kết quả "Điểm vàng" ra sao?**
  * Sau khi huấn luyện xong bộ tham số `state_dict` ở định dạng số thực 32-bit (`FP32`), họ áp dụng kỹ thuật **Lượng tử hóa sau huấn luyện (Post-training Quantization)** sang định dạng nén (`FP16` / `INT8`).
  * **Kết quả Điểm vàng (Trade-off):** Dung lượng file mô hình giảm từ **35.34 MB xuống chỉ còn 5.76 MB** (nén nhỏ đi **6.14 lần**), tốc độ suy luận trên thiết bị biên tăng vọt trong khi **độ chính xác chẩn đoán gần như không suy giảm (< 1%)** so với bản gốc!
* **4. Nhóm mình học và kế thừa được gì từ bài này?**
  * Trả lời trọn vẹn câu hỏi của thầy hướng dẫn ở phút `14:05 – 15:40` trong buổi họp: *"Nén mô hình là làm cái gì và tìm điểm vàng đánh đổi giữa dung lượng và độ chính xác như thế nào?"*. Nhóm mình dùng chính phương pháp xuất chuẩn **ONNX + Quantization** này trong kịch bản [`scripts/export_and_benchmark_onnx.py`](file:///d:/ComputerVision/scripts/export_and_benchmark_onnx.py).

---

### 📄 Bài báo số 3 (Bài báo gốc của Bộ dữ liệu Figshare 3.064 ảnh MRI 2D — Bắt buộc trích dẫn)
* **Tên bài báo:** *Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition*
* **Tác giả:** Jun Cheng, Wei Huang, Shuangliang Cao, Ru Yang, Wei Yang, Zhaoqiang Yun, Zhijian Wang, Qianjin Feng
* **Nơi công bố:** Tạp chí *PLOS ONE* (Năm 2015), DOI: `10.1371/journal.pone.0140381`
* **🔗 Link đọc Full-text miễn phí:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140381
* **🔗 Link bộ dữ liệu gốc trên Figshare:** https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
* **Nội dung cốt lõi:**
  * Công bố bộ dữ liệu chuẩn quốc tế gồm **3.064 lát cắt MRI 2D T1-weighted contrast-enhanced** từ **233 bệnh nhân** với 3 loại u (*Meningioma: 708 ảnh, Glioma: 1.426 ảnh, Pituitary: 930 ảnh*).
  * Chỉ ra rằng vùng mô xung quanh ranh giới khối u chứa rất nhiều thông tin đặc trưng (*features*) quan trọng để phân biệt 3 loại u này, đồng thời nhấn mạnh việc chia tập dữ liệu theo **Mã bệnh nhân (`PID`)** để đánh giá khách quan (không bị *Data Leakage*).

---

### 📄 Bài báo số 4 (Bài báo gốc khai sinh kiến trúc U-Net trong Y khoa — Bắt buộc trích dẫn)
* **Tên bài báo:** *U-Net: Convolutional Networks for Biomedical Image Segmentation*
* **Tác giả:** Olaf Ronneberger, Philipp Fischer, Thomas Brox
* **Nơi công bố:** Hội nghị *MICCAI / Springer* (Năm 2015), DOI: `10.1007/978-3-319-24574-4_28`
* **🔗 Link đọc/tải trực tiếp PDF:** https://arxiv.org/pdf/1505.04597
* **🔗 Link trang chính (arXiv):** https://arxiv.org/abs/1505.04597
* **Nội dung cốt lõi:**
  * Giới thiệu kiến trúc hình chữ U đối xứng gồm nhánh thu hẹp (**Contracting Path / Encoder** dùng tích chập + MaxPooling để bắt ngữ cảnh) và nhánh mở rộng (**Expansive Path / Decoder** dùng giải tích chập/Upsampling để định vị chính xác từng pixel), nối với nhau bằng các **đường nối tắt (Skip Connections)**.
  * Chứng minh U-Net có thể học phân đoạn ảnh y sinh 2D cực kỳ chính xác chỉ với vài nghìn bức ảnh nhờ kỹ thuật tăng cường dữ liệu (*Data Augmentation*).

---

### 📄 Bài báo số 5 (Bài báo nền tảng về Federated Learning cho U não MRI — Hướng mở rộng thầy gợi ý)
* **Tên bài báo:** *Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data*
* **Tác giả:** Micah J. Sheller, Brandon Edwards, G. Anthony Reina, Spyridon Bakas et al.
* **Nơi công bố:** Tạp chí *Nature Scientific Reports* (Năm 2020), DOI: `10.1038/s41598-020-69250-1`
* **🔗 Link đọc Full-text miễn phí:** https://www.nature.com/articles/s41598-020-69250-1
* **🔗 Link tải trực tiếp PDF:** https://www.nature.com/articles/s41598-020-69250-1.pdf
* **1. Họ giải quyết vấn đề gì (Đúng câu chuyện thầy kể ở phút 00:01 – 06:15)?**
  * Các bệnh viện không thể chia sẻ ảnh chụp MRI của bệnh nhân cho nhau vì luật bảo mật dữ liệu y tế (*Patient Privacy & Legality*). Nhưng nếu mỗi bệnh viện chỉ tự train trên vài trăm ảnh của riêng mình thì mô hình bị **Bias (thiên lệch)** và kém chính xác.
* **2. Giải pháp Federated Learning (Học liên kết) hoạt động ra sao?**
  * Tại mỗi bệnh viện (đóng vai trò là 1 *Worker / Thiết bị biên*), mô hình **U-Net** tự học trên ảnh MRI của bệnh viện đó.
  * Kết thúc mỗi vòng (*Round*), các bệnh viện **chỉ gửi bộ từ điển tham số (`state_dict`)** về máy chủ tổng hợp (**Aggregator**).
  * Máy chủ `Aggregator` cộng trung bình có trọng số các bộ tham số (**FedAvg**) để tạo ra mô hình tổng (**Global Model**) rồi gửi trả lại cho các bệnh viện.
* **3. Kết quả đạt được:**
  * Mô hình U-Net huấn luyện theo kiểu **Federated Learning** qua 10 bệnh viện đạt **99% chất lượng khoanh vùng khối u** so với việc gom toàn bộ ảnh bệnh nhân về một máy chủ trung tâm, trong khi **không một bức ảnh bệnh nhân nào bị lộ ra khỏi bệnh viện**!

---

## 🎓 III. TÓM TẮT "TỪ ĐIỂN BẢN CHẤT" ĐỂ TRẢ LỜI HỘI ĐỒNG (THEO DẶN DÒ CỦA THẦY)

| Câu hỏi của Thầy / Hội đồng | Câu trả lời ngắn gọn, chuẩn bản chất (Dành cho cả 3 thành viên) |
| :--- | :--- |
| **1. "Model" thực chất là cái gì?** | Kiến trúc mạng (như CNN, U-Net) là bộ khung công thức toán học, còn **Model sau khi train xong** thực chất là **một từ điển chứa các con số tham số/trọng số (`state_dict`)** đã được tối ưu. Khi đưa ảnh MRI vào, máy tính nhân các điểm ảnh với bộ tham số này để rút ra đặc trưng (*features*) và so khớp với mẫu (*pattern*) của từng loại u. |
| **2. "Feature" trong AI nhận diện ảnh là gì?** | Là **Đặc trưng hình ảnh** (không phải chức năng phần mềm). Ở các tầng đầu của CNN, *feature* là cạnh, đường viền sáng tối; ở các tầng sâu, *feature* là hình thái, kết cấu và vị trí của khối u. |
| **3. "Nén mô hình" là làm gì? Có giống WinRAR không?** | Không giống WinRAR. Nén mô hình AI là **giảm số lượng tham số** (dùng tích chập tách chiều *Depthwise Separable Conv*, cắt tỉa *Pruning*) hoặc **làm tròn độ chính xác của con số trong `state_dict`** (Lượng tử hóa *Quantization* từ số thực 32-bit `FP32` xuống 16-bit `FP16` hoặc số nguyên 8-bit `INT8`). |
| **4. Tại sao nén quá tay lại giảm độ chính xác? "Điểm vàng" là gì?** | Vì khi làm tròn số hoặc cắt bớt tham số quá nhiều, một số đặc trưng tinh vi (như ranh giới mờ của khối u nhỏ) bị mất đi (*mất feature*). **"Điểm vàng" (Sweet spot)** là mức nén giúp dung lượng giảm mạnh (3–10 lần), chạy cực nhanh trên thiết bị biên nhưng độ chính xác chỉ giảm rất nhẹ (1%–3%), vẫn giữ ở mức cao (>90% phân loại, >80% khoanh vùng). |
| **5. "Data Bias" (Thiên lệch dữ liệu) trên bộ dữ liệu công khai là gì?** | Là hiện tượng mô hình học quá sát với đặc trưng độ sáng/độ nét của các máy chụp trong bộ dữ liệu mẫu (Figshare/Kaggle), nên khi gặp ảnh từ máy chụp đời khác có thể giảm độ chính xác. Nhóm khắc phục bằng tiền xử lý cân bằng sáng cục bộ **CLAHE** và biến đổi **Data Augmentation**. |
| **6. Trong "Federated Learning", thiết bị biên gửi cái gì lên Server?** | Thiết bị biên **không gửi ảnh MRI bệnh nhân**, mà chỉ gửi **bộ tham số mô hình (`state_dict`)** nặng vài MB lên máy chủ `Aggregator`. Máy chủ tính trung bình cộng có trọng số theo số lượng ảnh của từng nơi (**FedAvg**) để cập nhật mô hình chung. |
