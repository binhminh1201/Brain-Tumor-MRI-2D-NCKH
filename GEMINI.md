# Quy Tắc Hoạt Động Riêng Cho Dự Án `d:\ComputerVision` (NCKH U Não MRI 2D)

Quy tắc này **chỉ áp dụng riêng cho các cuộc hội thoại trong workspace `d:\ComputerVision`**:

1. **Phạm vi kỹ thuật cố định:**
   - Dự án chỉ xử lý **ảnh MRI 2D** (kích thước chuẩn `256x256`, định dạng `.png`/`.jpg`/`.mat`). Tuyệt đối không sử dụng mô hình 3D phức tạp.
   - Giải thích và ghi chép mọi kiến thức, thuật toán, câu lệnh từ mức **cơ bản nhất** để phục vụ việc học tập và viết báo cáo NCKH cấp trường.
   - Nhóm nghiên cứu gồm **02 thành viên**, làm việc qua **Git + Môi trường ảo Python (`.venv`)**, **chưa sử dụng Docker** ở giai đoạn hiện tại.

2. **Cập nhật Nhật ký & Timeline dự án (`TIMELINE_NCKH.md` & `README.md`):**
   - Sau mỗi phiên làm việc hoặc mỗi khi hoàn thành một hạng mục (cài đặt, xử lý dữ liệu, huấn luyện, tối ưu, sửa lỗi...), BẮT BUỘC phải cập nhật vào file `TIMELINE_NCKH.md` (và đồng bộ thông tin trên `README.md`) đầy đủ 3 mục:
     - **Đã làm được (Completed):** Ghi chi tiết những gì đã hoàn thành, lý do chọn giải pháp, cấu hình, kết quả đạt được.
     - **Đang triển khai (In Progress):** Những công việc đang thực hiện dở dang.
     - **Chưa làm được / Kế hoạch tiếp theo (Pending / Next Steps):** Những hạng mục còn lại trong lộ trình 6 giai đoạn.
   - Nội dung ghi chép phải rõ ràng, mạch lạc để cuối kỳ nhóm có thể trích xuất trực tiếp làm **Báo cáo tổng kết đề tài NCKH** và **Bài báo khoa học cấp trường**.

3. **Tự động đồng bộ Git cuối mỗi phiên/tác vụ:**
   - Sau khi cập nhật code và `TIMELINE_NCKH.md`, tự động thực hiện:
     - `git add -A`
     - `git commit -m "<Mô tả rõ ràng phiên làm việc>"`
     - `git push` (nếu kho chứa từ xa `remote origin` đã được kết nối).
