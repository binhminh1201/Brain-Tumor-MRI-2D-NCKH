import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path

def create_document():
    doc = docx.Document()

    # Căn lề tiêu chuẩn A4 (2.54 cm = 1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Cấu hình phông chữ mặc định: Times New Roman, cỡ 12
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.25
    normal_style.paragraph_format.space_after = Pt(6)

    def add_heading_styled(text, level, color_rgb=(0, 0x33, 0x66)):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.color.rgb = RGBColor(*color_rgb)
        if level == 1:
            run.font.size = Pt(15)
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
        elif level == 2:
            run.font.size = Pt(13)
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
        return p

    # Tiêu đề tài liệu
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t_run = title_p.add_run(
        "ĐỀ TÀI NGHIÊN CỨU KHOA HỌC:\n"
        "ỨNG DỤNG THỊ GIÁC MÁY TÍNH VÀ HỌC SÂU TRONG PHÁT HIỆN, PHÂN LOẠI KHỐI U NÃO TRÊN ẢNH MRI 2D"
    )
    t_run.bold = True
    t_run.font.name = 'Times New Roman'
    t_run.font.size = Pt(15)
    t_run.font.color.rgb = RGBColor(0x00, 0x20, 0x60)
    title_p.paragraph_format.space_after = Pt(4)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s_run = sub_p.add_run("TỔNG QUAN HƯỚNG TRIỂN KHAI VÀ Ý NGHĨA THỰC TIỄN CỦA MÔ HÌNH")
    s_run.italic = True
    s_run.font.name = 'Times New Roman'
    s_run.font.size = Pt(12)
    s_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    sub_p.paragraph_format.space_after = Pt(16)

    # PHẦN 1
    add_heading_styled("I. Giai Đoạn 1: Nghiên Cứu & Huấn Luyện Tại Lab (Hiện Tại)", 1)

    p1 = doc.add_paragraph()
    p1.add_run("• ").bold = True
    p1.add_run("Huấn luyện mô hình cơ sở: ").bold = True
    p1.add_run(
        "Nhóm nghiên cứu sử dụng máy tính có trang bị bộ xử lý đồ họa rời (GPU NVIDIA) để \"dạy\" mô hình học sâu "
        "nhận diện đặc trưng bất thường từ "
    )
    p1.add_run("3.564 bức ảnh chụp MRI não 2D cắt lát chuẩn y khoa").bold = True
    p1.add_run(
        " (gồm dữ liệu Figshare có sẵn mặt nạ viền khối u do bác sĩ chẩn đoán hình ảnh vẽ tay và dữ liệu não bình thường Sartaj)."
    )

    p2 = doc.add_paragraph()
    p2.add_run("• ").bold = True
    p2.add_run("Đóng gói & Tối ưu hóa tri thức: ").bold = True
    p2.add_run("Sau khi huấn luyện đạt độ chính xác cao (>90%), nhóm tiến hành ")
    p2.add_run("\"đóng băng\" và nén toàn bộ kiến thức mạng").bold = True
    p2.add_run(" sang một file định dạng chuẩn mở duy nhất: ")
    code_r = p2.add_run("model_u_nao.onnx")
    code_r.bold = True
    code_r.font.color.rgb = RGBColor(0x00, 0x66, 0xCC)
    p2.add_run(" với dung lượng siêu nhỏ gọn chỉ khoảng ")
    p2.add_run("1.0 MB đến 3.7 MB").bold = True
    p2.add_run(" (giảm hơn 96% dung lượng so với mô hình U-Net thông thường).")

    p3 = doc.add_paragraph()
    p3.add_run("• ").bold = True
    p3.add_run("Ý nghĩa của file .onnx: ").bold = True
    p3.add_run("File .onnx này chính là ")
    p3.add_run("sản phẩm cốt lõi của đề tài").bold = True
    p3.add_run(
        ". Nó độc lập hoàn toàn với các phần mềm lập trình cồng kềnh, đóng vai trò như một bộ thư viện độc lập "
        "có thể nạp trực tiếp vào bất kỳ ứng dụng phần mềm hoặc thiết bị nhúng nào."
    )

    # PHẦN 2
    add_heading_styled("II. Giai Đoạn 2: Triển Khai Thực Tế & Thiết Bị Biên (Ý Nghĩa Thực Tiễn)", 1)

    p_intro = doc.add_paragraph()
    p_intro.add_run("Hãy đặt mô hình vào bối cảnh thực tế tại một ")
    p_intro.add_run("phòng khám nhỏ tuyến huyện, trạm y tế cơ sở hoặc xe y tế khám bệnh lưu động").bold = True
    p_intro.add_run(":")

    add_heading_styled("1. Điều kiện phần cứng thực tế tại cơ sở y tế:", 2)
    p_hw = doc.add_paragraph()
    p_hw.add_run("Tại các cơ sở này, bác sĩ chỉ có sẵn ")
    p_hw.add_run("máy tính văn phòng cấu hình bình thường (chỉ chạy chip CPU phổ thông, không có card đồ họa GPU đắt đỏ)").bold = True
    p_hw.add_run(", hoặc một bo mạch máy tính mini nhỏ gọn (như Raspberry Pi / máy tính nhúng). Đây chính là đại diện cho ")
    p_hw.add_run("\"Thiết bị cấu hình hạn chế / Thiết bị biên\"").bold = True
    p_hw.add_run(" mà đề tài đặt mục tiêu hướng tới.")

    add_heading_styled("2. Quy trình vận hành chẩn đoán tại chỗ (On-Device AI):", 2)
    bullets = [
        ("Bước 1: Thu nhận ảnh", "Bệnh nhân sau khi chụp cộng hưởng từ, file ảnh MRI 2D được xuất ra."),
        ("Bước 2: Mở phần mềm", "Bác sĩ mở phần mềm chẩn đoán của nhóm (bên trong đã nhúng sẵn file não model_u_nao.onnx)."),
        ("Bước 3: Nạp ảnh bệnh nhân", "Bác sĩ đưa ảnh MRI của bệnh nhân vào phần mềm."),
        ("Bước 4: Tự động suy luận siêu tốc", "Chỉ mất khoảng 0.03 giây (36 mili-giây), con chip CPU trên máy bác sĩ tự tính toán và hiển thị ngay kết quả trực quan lên màn hình:")
    ]

    for title, desc in bullets:
        bp = doc.add_paragraph()
        bp.add_run("• ").bold = True
        bp.add_run(title + ": ").bold = True
        bp.add_run(desc)

    sub_bullets = [
        ("Kết luận phân loại:", "Thông báo rõ loại tổn thương, ví dụ \"Phát hiện: U màng não (Độ tin cậy: 96%)\"."),
        ("Khoanh vùng tổn thương:", "Tự động tô màu đỏ bán trong suốt và vẽ đường viền vàng chính xác quanh khối u."),
        ("Đo lường định lượng:", "Tính toán diện tích khối u (theo số điểm ảnh pixel và tỷ lệ % so với hộp sọ)."),
        ("Giải thích AI (Grad-CAM):", "Xuất bản đồ nhiệt minh bạch vùng mô não mà mô hình dựa vào để chẩn đoán.")
    ]

    for title, desc in sub_bullets:
        sp = doc.add_paragraph()
        sp.paragraph_format.left_indent = Inches(0.4)
        sp.add_run("– ").bold = True
        sp.add_run(title + " ").bold = True
        sp.add_run(desc)

    p_doc = doc.add_paragraph()
    p_doc.paragraph_format.space_before = Pt(4)
    p_doc.add_run("• ").bold = True
    p_doc.add_run("Hỗ trợ bác sĩ: ").bold = True
    p_doc.add_run("Bác sĩ sử dụng các thông tin trực quan này để đối chiếu, hội chẩn sơ bộ nhanh và giải thích minh bạch tình trạng cho người bệnh.")

    # PHẦN 3
    add_heading_styled("III. Ba Điểm Mấu Chốt Cần Nắm Vững Khi Thuyết Minh Đề Tài", 1)

    points = [
        (
            "1. Không phải mô hình gửi dữ liệu đi đâu",
            "Mô hình của nhóm là dạng nhúng trực tiếp (On-device AI). Toàn bộ thuật toán chạy 100% nội bộ trên máy của bác sĩ/thiết bị biên, không phụ thuộc vào máy chủ trung tâm hay đường truyền Internet."
        ),
        (
            "2. Nhiệm vụ của thiết bị biên",
            "Thiết bị biên thu nhận ảnh thực tế của bệnh nhân và tự mình thực thi file mô hình của nhóm để đưa ra chẩn đoán ngay lập tức tại phòng khám."
        ),
        (
            "3. Vì sao giải pháp \"siêu nhẹ & tối ưu\" lại là bước đột phá?",
            "Nếu sử dụng các mô hình học sâu kinh điển (nặng hàng trăm MB, hàng chục triệu tham số), máy tính văn phòng của bác sĩ sẽ bị quá tải, đơ máy (mất 5–10 giây/ảnh) hoặc bắt buộc phải gửi ảnh lên Cloud Server (tốn chi phí duy trì đắt đỏ và tiềm ẩn nguy cơ rò rỉ dữ liệu y tế riêng tư của người bệnh).\n"
            "Ngược lại, mô hình đề xuất của nhóm chỉ nặng ~1–3.7 MB, xử lý trong 0.03 giây trên CPU thông thường, bảo mật tuyệt đối dữ liệu bệnh án tại chỗ. Đây chính là giá trị khoa học và thực tiễn cao nhất khi bảo vệ trước Hội đồng NCKH."
        )
    ]

    for p_title, p_desc in points:
        pp = doc.add_paragraph()
        run_t = pp.add_run(p_title + ": ")
        run_t.bold = True
        run_t.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
        pp.add_run(p_desc)

    # Lưu ra Desktop của người dùng
    desktop_file = Path("C:/Users/DELL/Desktop/Tong_quan_Huong_trien_khai_NCKH_U_nao.docx")
    doc.save(str(desktop_file))

    # Lưu thêm 1 bản dự phòng vào thư mục outputs/ của dự án
    workspace_file = Path("d:/ComputerVision/outputs/Tong_quan_Huong_trien_khai_NCKH_U_nao.docx")
    workspace_file.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(workspace_file))

    print(f"DONE: {desktop_file}")

if __name__ == '__main__':
    create_document()
