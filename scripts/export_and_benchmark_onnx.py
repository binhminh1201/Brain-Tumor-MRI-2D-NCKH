"""
Kịch bản Xuất chuẩn ONNX, Nén mô hình & Đo Benchmark trên CPU (Giai đoạn 5):
- Chuyển đổi các mô hình 2D từ PyTorch (.pth) sang chuẩn trung gian ONNX (.onnx).
- Thực hiện lượng tử hóa động (Dynamic Quantization INT8) để giảm dung lượng cho thiết bị biên.
- Tự động đo và xuất bảng số liệu (Số tham số, Dung lượng MB, Thời gian suy luận ms/ảnh, FPS)
  ra file `outputs/tables/benchmark_results.csv` để đưa thẳng vào Bài báo NCKH cấp trường.
"""

from pathlib import Path
import sys
import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import pandas as pd
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import CHECKPOINT_DIR, IMAGE_SIZE, TABLES_DIR
from src.models.classifiers import build_classifier_2d
from src.models.unet_2d import LightweightMultiTaskUNet2D, StandardUNet2D


def count_parameters_millions(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def export_model_to_onnx(model: nn.Module, onnx_path: Path, is_multitask: bool = False) -> None:
    model.eval()
    dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
    output_names = ["seg_logits", "cls_logits"] if is_multitask else ["output"]
    dynamic_axes = {"input": {0: "batch_size"}}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def benchmark_onnx_cpu(onnx_path: Path, num_runs: int = 30) -> tuple[float, float]:
    """Đo tốc độ suy luận thực tế trên CPU (ms/ảnh và FPS) qua ONNX Runtime."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options, providers=["CPUExecutionProvider"])

    dummy_np = np.random.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).astype(np.float32)

    # Chạy khởi động (Warm-up)
    for _ in range(5):
        session.run(None, {"input": dummy_np})

    start = time.perf_counter()
    for _ in range(num_runs):
        session.run(None, {"input": dummy_np})
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / num_runs) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    return round(latency_ms, 2), round(fps, 1)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    models_to_compare = [
        ("Standard 2D U-Net (Baseline Nặng)", StandardUNet2D(), False, "standard_unet_2d"),
        ("Lightweight Multi-Task 2D U-Net (Đề xuất)", LightweightMultiTaskUNet2D(), True, "light_multitask_unet_2d"),
        ("MobileNetV2 2D Classifier", build_classifier_2d("mobilenet_v2", pretrained=False), False, "mobilenet_v2_cls"),
        ("Basic 4-Layer 2D CNN", build_classifier_2d("basic_cnn", pretrained=False), False, "basic_cnn_cls"),
    ]

    records = []
    print("[*] Bắt đầu xuất chuẩn ONNX và đo Benchmark trên CPU cho ảnh MRI 2D (256x256)...")

    for display_name, model, is_multitask, short_name in models_to_compare:
        params_m = round(count_parameters_millions(model), 3)

        # Lưu PyTorch state_dict tạm để đo dung lượng gốc
        pth_path = CHECKPOINT_DIR / f"{short_name}.pth"
        torch.save(model.state_dict(), pth_path)
        pth_size_mb = round(pth_path.stat().st_size / (1024 * 1024), 2)

        # Xuất sang ONNX FP32
        onnx_fp32_path = CHECKPOINT_DIR / f"{short_name}_fp32.onnx"
        export_model_to_onnx(model, onnx_fp32_path, is_multitask=is_multitask)
        onnx_fp32_mb = round(onnx_fp32_path.stat().st_size / (1024 * 1024), 2)
        lat_fp32, fps_fp32 = benchmark_onnx_cpu(onnx_fp32_path)

        records.append({
            "Mô hình 2D": display_name,
            "Định dạng": "PyTorch / ONNX (FP32)",
            "Tham số (Triệu)": params_m,
            "Dung lượng (MB)": onnx_fp32_mb,
            "Thời gian CPU (ms/ảnh)": lat_fp32,
            "Tốc độ (FPS)": fps_fp32,
        })

        # Lượng tử hóa sang ONNX INT8 cho mô hình đề xuất
        if "Lightweight" in display_name or "MobileNetV2" in display_name:
            onnx_int8_path = CHECKPOINT_DIR / f"{short_name}_int8.onnx"
            quantize_dynamic(
                model_input=onnx_fp32_path,
                model_output=onnx_int8_path,
                weight_type=QuantType.QUInt8,
            )
            onnx_int8_mb = round(onnx_int8_path.stat().st_size / (1024 * 1024), 2)
            lat_int8, fps_int8 = benchmark_onnx_cpu(onnx_int8_path)
            records.append({
                "Mô hình 2D": f"{display_name} [Tối ưu INT8]",
                "Định dạng": "ONNX Quantized (INT8)",
                "Tham số (Triệu)": params_m,
                "Dung lượng (MB)": onnx_int8_mb,
                "Thời gian CPU (ms/ảnh)": lat_int8,
                "Tốc độ (FPS)": fps_int8,
            })

    df = pd.DataFrame(records)
    csv_out = TABLES_DIR / "benchmark_results.csv"
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")

    print("\n================ BẢNG KẾT QUẢ ĐỐI SÁNH DUNG LƯỢNG & TỐC ĐỘ SUY LUẬN 2D ================")
    print(df.to_string(index=False))
    print(f"\n[+] Đã lưu bảng số liệu phục vụ viết bài báo tại: {csv_out}")


if __name__ == "__main__":
    main()
