"""
Hệ thống Nguyên mẫu Trực quan Chẩn đoán U não trên ảnh MRI 2D (Giai đoạn 6):
Chạy bằng lệnh:
    streamlit run app/main_app.py
Tính năng:
1. Tải ảnh MRI 2D (.png, .jpg) hoặc chọn ảnh mẫu có sẵn.
2. Tiền xử lý CLAHE tăng cường độ tương phản 2D.
3. Chạy mô hình Đa nhiệm Lightweight Multi-Task 2D U-Net (hỗ trợ cả PyTorch & ONNX Runtime trên CPU).
4. Hiển thị trực quan 4 khung: Ảnh gốc -> Ảnh CLAHE -> Bản đồ nhiệt Grad-CAM -> Đường viền Khoanh vùng u + Diện tích u.
"""

from pathlib import Path
import sys
import time
import cv2
import numpy as np
import streamlit as st
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import CHECKPOINT_DIR, CLASS_LABELS_VI, CLASS_NAMES, IMAGE_SIZE, PROCESSED_DATA_DIR
from src.dataset import preprocess_mri_2d
from src.models.unet_2d import LightweightMultiTaskUNet2D
from src.utils.gradcam import generate_gradcam_2d, overlay_mask_and_contour_2d
from src.utils.metrics import analyze_tumor_geometry_2d


@st.cache_resource
def load_multitask_model() -> LightweightMultiTaskUNet2D:
    model = LightweightMultiTaskUNet2D(enable_classification_head=True)
    ckpt_path = CHECKPOINT_DIR / "best_seg_light_multitask_unet.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def main():
    st.set_page_config(page_title="Hệ thống Chẩn đoán U não MRI 2D", layout="wide")
    st.title("🧠 Hệ Thống Nguyên Mẫu: Phát Hiện, Khoanh Vùng & Phân Loại U Não Trên Ảnh MRI 2D")
    st.markdown(
        "**Đề tài NCKH:** *Ứng dụng thị giác máy tính và học sâu trong phát hiện, phân loại khối u não trên ảnh MRI 2D*"
    )

    st.sidebar.header("⚙️ Cấu hình & Đầu vào")
    use_clahe = st.sidebar.checkbox("Tăng cường tương phản y khoa (CLAHE 2D)", value=True)
    seg_threshold = st.sidebar.slider("Ngưỡng khoanh vùng khối u (Threshold)", 0.20, 0.80, 0.50, 0.05)

    # Cho phép tải ảnh hoặc chọn ảnh mẫu từ tập test
    uploaded_file = st.sidebar.file_uploader("Tải lên ảnh chụp MRI 2D (.png, .jpg)", type=["png", "jpg", "jpeg"])

    sample_images = sorted(list((PROCESSED_DATA_DIR / "segmentation" / "test" / "images").glob("*.png")))
    selected_sample = None
    if sample_images:
        sample_names = [p.name for p in sample_images]
        chosen_name = st.sidebar.selectbox("Hoặc chọn ảnh MRI 2D mẫu để thử nghiệm:", sample_names)
        selected_sample = PROCESSED_DATA_DIR / "segmentation" / "test" / "images" / chosen_name

    raw_img_gray = None
    gt_mask = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        raw_img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    elif selected_sample is not None and selected_sample.exists():
        raw_img_gray = cv2.imread(str(selected_sample), cv2.IMREAD_GRAYSCALE)
        gt_path = PROCESSED_DATA_DIR / "segmentation" / "test" / "masks" / selected_sample.name
        if gt_path.exists():
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    if raw_img_gray is None:
        st.info("👈 Vui lòng tải lên một bức ảnh MRI 2D ở thanh bên trái (hoặc chạy `python scripts/prepare_data.py --create-demo` để tạo ảnh mẫu).")
        return

    model = load_multitask_model()
    tensor_img, processed_rgb = preprocess_mri_2d(raw_img_gray, IMAGE_SIZE, use_clahe=use_clahe)
    input_batch = tensor_img.unsqueeze(0)

    # Đo thời gian suy luận thực tế trên CPU
    t0 = time.perf_counter()
    with torch.no_grad():
        seg_logits, cls_logits = model(input_batch)
        probs_cls = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
        pred_mask_prob = torch.sigmoid(seg_logits)[0, 0].cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_class_idx = int(np.argmax(probs_cls))
    pred_class_key = CLASS_NAMES[pred_class_idx]
    pred_label_vi = CLASS_LABELS_VI[pred_class_key]
    confidence = float(probs_cls[pred_class_idx] * 100.0)

    binary_mask = (pred_mask_prob >= seg_threshold).astype(np.uint8)
    if pred_class_key == "no_tumor":
        binary_mask = np.zeros_like(binary_mask)

    # Tạo bản đồ nhiệt Grad-CAM 2D tại tầng Bottleneck của U-Net
    heatmap_rgb, _, _ = generate_gradcam_2d(model, input_batch, target_layer=model.bot)
    gradcam_overlay = cv2.addWeighted(processed_rgb, 0.60, heatmap_rgb, 0.40, 0)

    # Vẽ viền khoanh vùng khối u
    contour_vis = overlay_mask_and_contour_2d(processed_rgb, binary_mask, gt_mask=gt_mask)
    geom = analyze_tumor_geometry_2d(binary_mask)

    # Hiển thị 4 ô trực quan
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("1. Ảnh MRI 2D Gốc")
        st.image(cv2.resize(raw_img_gray, IMAGE_SIZE), caption="Ảnh đầu vào (256x256)", use_container_width=True)
    with col2:
        st.subheader("2. Tăng Tương Phản CLAHE")
        st.image(processed_rgb, caption="Làm rõ ranh giới mô mềm", use_container_width=True)
    with col3:
        st.subheader("3. Bản Đồ Nhiệt Grad-CAM")
        st.image(gradcam_overlay, caption="Vùng mô AI tập trung chú ý", use_container_width=True)
    with col4:
        st.subheader("4. Khoanh Vùng Khối U (U-Net)")
        st.image(contour_vis, caption="Viền Vàng: AI | Viền Xanh: Nhãn thật", use_container_width=True)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Kết luận Phân loại", pred_label_vi)
    m2.metric("Độ tin cậy (Confidence)", f"{confidence:.2f}%")
    m3.metric("Diện tích vùng u 2D", f"{geom['tumor_pixels']} px ({geom['area_ratio_percent']}%)")
    m4.metric("Tốc độ suy luận (CPU)", f"{latency_ms:.2f} ms/ảnh")


if __name__ == "__main__":
    main()
