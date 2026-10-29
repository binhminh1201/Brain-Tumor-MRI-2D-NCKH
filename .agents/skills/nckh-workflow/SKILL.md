---
name: nckh-workflow
description: >-
  Kỹ năng tự động hóa quy trình nghiên cứu, chuẩn hóa mô hình MRI 2D, ghi nhật ký NCKH
  (TIMELINE_NCKH.md, README.md) và đồng bộ Git sau mỗi phiên làm việc.
---

# Kỹ Năng Điều Hành & Quản Lý Dự Án NCKH U Não MRI 2D

Kỹ năng này định hướng cho AI (và các trợ lý mã nguồn) hỗ trợ bất kỳ thành viên nào trong nhóm phát triển dự án này đúng chuẩn mực học thuật và quy trình quản lý dự án.

## 1. Nguyên Tắc Cốt Lõi Bất Di Bất Dịch
- **100% Ảnh MRI 2D:** Kích thước chuẩn hóa `256x256` (RGB/Grayscale). Không dùng mô hình 3D (Voxel/3D Conv) để đảm bảo mô hình nhẹ, phù hợp mục tiêu chạy trên thiết bị biên/cấu hình hạn chế.
- **Giải thích từ cơ bản:** Mọi giải thích lý thuyết, mã nguồn, thuật toán đều phải xuất phát từ mức cơ bản nhất (Toán học, cấu trúc lớp CNN/U-Net, hàm mất mát...) để nhóm có thể trích xuất trực tiếp vào Báo cáo tổng kết đề tài và Bài báo khoa học cấp trường.
- **Môi trường & Công cụ:** Dự án chỉ dùng Python `.venv` + Git, **không dùng Docker** ở giai đoạn hiện tại.

## 2. Quy Trình Cập Nhật Nhật Ký Dự Án (`TIMELINE_NCKH.md`)
Bất cứ khi nào thành viên làm việc xong một phiên hoặc hoàn thành một tác vụ (huấn luyện, đánh giá, sửa lỗi, xuất mô hình...), AI **BẮT BUỘC** phải chủ động mở file `TIMELINE_NCKH.md` và ghi nhận một phiên mới với cấu trúc chuẩn:

```markdown
### 🔹 PHIÊN X (Ngày DD/MM/YYYY): [Tên ngắn gọn phiên làm việc]

#### 1. Những gì ĐÃ LÀM ĐƯỢC (Completed ✅)
- Ghi chi tiết các công việc đã hoàn thành.
- Lý do chọn thuật toán/giải pháp (từ cơ bản).
- Kết quả thực nghiệm cụ thể (Bảng số liệu: Accuracy, Dice Score, Latency ms, Dung lượng MB...).

#### 2. Những gì ĐANG TRIỂN KHAI (In Progress 🔄)
- Những tác vụ đang chạy dở dang hoặc đang cần kiểm tra tiếp.

#### 3. Những gì CHƯA LÀM ĐƯỢC / KẾ HOẠCH TIẾP THEO (Pending / Backlog ⏳)
- Các hạng mục còn lại bám sát 6 giai đoạn của đề tài.
```

Đồng thời, cập nhật lại bảng tiến độ tóm tắt tại **Mục III. BẢNG TỔNG HỢP TRẠNG THÁI CÁC HẠNG MỤC** trong `TIMELINE_NCKH.md`.

## 3. Quy Trình Tự Động Đồng Bộ Git
Sau khi cập nhật code và nhật ký `TIMELINE_NCKH.md`:
1. Kiểm tra trạng thái Git: `git status`
2. Đóng gói toàn bộ thay đổi: `git add -A`
3. Commit với thông điệp rõ ràng: `git commit -m "Phien X: <Mô tả chi tiết nội dung đã làm>"`
4. Đẩy lên nhánh chính: `git push origin main`
