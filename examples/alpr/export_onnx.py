#!/usr/bin/env python3
"""
Export trained RF-DETR model to ONNX format (GPU, legacy exporter)
"""
import torch
import os

torch._dynamo.config.suppress_errors = True

from rfdetr import RFDETRMedium

CHECKPOINT_PATH = "./output/license_plate_detector/checkpoint_best_ema.pth"
OUTPUT_DIR = "./export"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = RFDETRMedium(
    resolution=576,
    num_classes=1,
    pretrain_weights=CHECKPOINT_PATH
)

# Get inner model, force ALL tensors to CUDA
inner_model = model.model.model.cuda()
inner_model.eval()
inner_model.export = True

# Dummy input on CUDA
dummy_input = torch.randn(1, 3, 576, 576, device='cuda')

print("Exporting with legacy ONNX exporter (CUDA)...")

torch.onnx.export(
    inner_model,
    dummy_input,
    f"{OUTPUT_DIR}/rfdetr_alpr.onnx",
    opset_version=18,
    input_names=["images"],
    output_names=["boxes", "scores"],
    dynamic_axes={
        "images": {0: "batch"},
        "boxes": {0: "batch"},
        "scores": {0: "batch"}
    },
    do_constant_folding=True,
    export_params=True,
    dynamo=False,
)

print(f"✓ Exported to {OUTPUT_DIR}/rfdetr_alpr.onnx")
