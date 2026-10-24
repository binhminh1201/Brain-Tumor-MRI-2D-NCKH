"""
Module Trực quan hóa Kết quả Chẩn đoán & Giải thích AI (XAI - Giai đoạn 4 & 6):
- Vẽ đường viền khối u (Contour) và vùng phủ bán trong suốt lên ảnh MRI 2D.
- Tạo bản đồ nhiệt Grad-CAM 2D để giải thích mô hình đang tập trung vào vùng mô nào.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn


def overlay_mask_and_contour_2d(
    rgb_image: np.ndarray,
    binary_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Vẽ vùng khối u dự đoán (màu Đỏ cam + viền Vàng) và nhãn thật nếu có (viền Xanh lá)
    lên bức ảnh MRI 2D gốc.
    """
    vis = rgb_image.copy()
    mask_uint8 = ((binary_mask > 0).astype(np.uint8)) * 255

    # Tô phủ màu đỏ bán trong suốt lên vùng khối u
    colored_overlay = vis.copy()
    colored_overlay[mask_uint8 > 0] = [255, 60, 60]
    vis = cv2.addWeighted(vis, 0.72, colored_overlay, 0.28, 0)

    # Vẽ đường viền sắc nét quanh khối u dự đoán (Màu vàng)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 230, 0), 2)

    # Nếu có Ground Truth Mask thì vẽ viền Xanh lá để so sánh trong bài báo
    if gt_mask is not None:
        gt_uint8 = ((gt_mask > 0).astype(np.uint8)) * 255
        gt_contours, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, gt_contours, -1, (0, 255, 100), 1)

    return vis


def generate_gradcam_2d(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, int, float]:
    """
    Tính toán bản đồ nhiệt Grad-CAM 2D cho mô hình phân loại hoặc nhánh phân loại của U-Net.
    Trả về: (heatmap_rgb, predicted_class_idx, confidence_prob).
    """
    model.eval()
    activations = []
    gradients = []

    def forward_hook(_module, _inp, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h_f = target_layer.register_forward_hook(forward_hook)
    h_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        output = model(input_tensor)
        # Hỗ trợ cả mô hình đơn nhiệm (logits) và đa nhiệm (seg_logits, cls_logits)
        cls_logits = output[1] if isinstance(output, tuple) else output

        probs = torch.softmax(cls_logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0, pred_idx].item())

        score_idx = target_class if target_class is not None else pred_idx
        model.zero_grad()
        cls_logits[0, score_idx].backward()

        act = activations[0][0]  # (C, H', W')
        grad = gradients[0][0]   # (C, H', W')
        weights = torch.mean(grad, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * act, dim=0)
        cam = torch.relu(cam).cpu().numpy()

        cam = cam - np.min(cam)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        return heatmap_rgb, pred_idx, conf
    finally:
        h_f.remove()
        h_b.remove()
