# BẮT BUỘC: Hướng Dẫn Tác Nghiệp AI Trong Dự Án NCKH U Não MRI 2D

Mọi trợ lý AI (Gemini, Antigravity, Claude, ChatGPT, GitHub Copilot...) khi tham gia vào workspace này cùng bất kỳ thành viên nào trong nhóm **BẮT BUỘC TUÂN THỦ** các nguyên tắc sau:

1. **Phạm Vi Kỹ Thuật Cố Định:**
   - 100% Ảnh MRI 2D kích thước chuẩn `256x256` (.png/.jpg/.mat). Tuyệt đối không dùng mô hình 3D.
   - Giải thích cặn kẽ từ mức cơ bản nhất phục vụ viết báo cáo NCKH cấp trường.
   - Nhóm 2-3 thành viên, triển khai qua Git + `.venv`, **không dùng Docker**.

2. **Chủ Động Cập Nhật `TIMELINE_NCKH.md` & `README.md`:**
   - Khi kết thúc phiên làm việc hoặc hoàn thành một tính năng/thực nghiệm, AI **phải tự động cập nhật** file `TIMELINE_NCKH.md` đầy đủ 3 phần:
     + **Đã làm được (Completed ✅):** Chi tiết công việc, nguyên nhân chọn giải pháp, số liệu đạt được.
     + **Đang triển khai (In Progress 🔄):** Việc đang làm dở.
     + **Chưa làm được / Kế hoạch tiếp theo (Pending ⏳):** Các bước tiếp theo.

3. **Chủ Động Thực Hiện Git:**
   - Luôn chạy:
     ```bash
     git add -A
     git commit -m "<Mô tả ngắn gọn, rõ ràng phiên làm việc>"
     git push origin main
     ```
