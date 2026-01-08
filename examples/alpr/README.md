# ALPR (Automatic License Plate Recognition) Example

This example demonstrates training RF-DETR for license plate detection and exporting for deployment.

## Deployment Targets

This pipeline supports multiple deployment paths depending on your target hardware:

| Target | Runtime | Model Format | Notes |
|--------|---------|--------------|-------|
| NVIDIA GPU | TensorRT | `.engine` (FP16/INT8) | Fastest inference, requires CUDA |
| i.MX8M Plus | TFLite + NPU delegate | `.tflite` (INT8) | Vivante VIP8000 NPU |
| i.MX93 | TFLite + Ethos delegate | `.tflite` (INT8) | Ethos-U65 NPU |
| Generic ARM | ONNX Runtime / TFLite | `.onnx` / `.tflite` | CPU fallback |

**GPU path (this repo):** PyTorch checkpoint -> ONNX -> TensorRT engine

**Edge NPU path:** PyTorch checkpoint -> ONNX -> TFLite INT8 (with representative dataset calibration)

For edge deployment on ARM + NPU devices, use TFLite with INT8 quantization. TensorRT engines are GPU-specific and won't run on ARM NPUs.

## Pipeline Overview

1. **Prepare Dataset** - Merge COCO-format datasets
2. **Train Model** - Fine-tune RF-DETR Medium
3. **Export ONNX** - Convert to ONNX format
4. **Optimize ONNX** - Simplify graph (optional)
5. **Build TensorRT** - Create FP16/INT8 engines (GPU) -or- Convert to TFLite INT8 (Edge NPU)

## Scripts

| Script | Description |
|--------|-------------|
| `prepare_dataset.py` | Merge multiple COCO datasets into one |
| `train.py` | Train RF-DETR with W&B/TensorBoard logging |
| `export_onnx.py` | Export trained model to ONNX |
| `calibrate_int8.py` | Build TensorRT INT8 engine with calibration (GPU) |
| `convert_tflite.py` | Convert ONNX to TFLite INT8 (Edge NPU) |
| `test_inference.py` | Test detection + OCR on sample images |

## Quick Start

### 1. Prepare Dataset

```bash
# Update DATASET_DIRS in prepare_dataset.py, then:
python prepare_dataset.py
```

### 2. Train Model

```bash
# Update CONFIG in train.py, then:
python train.py
```

### 3. Export to ONNX

```bash
python export_onnx.py
```

### 4. Optimize ONNX (Optional)

```bash
pip install onnx-simplifier
python -m onnxsim ./export/rfdetr_alpr.onnx ./export/rfdetr_alpr_optimized.onnx
```

### 5. Build TensorRT Engine

**FP16 Engine:**
```bash
trtexec \
  --onnx=./export/rfdetr_alpr_optimized.onnx \
  --saveEngine=./export/detector_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096M \
  --minShapes=images:1x3x576x576 \
  --optShapes=images:1x3x576x576 \
  --maxShapes=images:1x3x576x576
```

**INT8 Engine (with calibration):**
```bash
# Requires calibration images in ./merged_dataset/valid/
python calibrate_int8.py
```

### 5b. Convert to TFLite INT8 (Edge NPU)

For ARM + NPU deployment (i.MX93, i.MX8M Plus), convert to TFLite INT8 instead of TensorRT:

```bash
# Convert ONNX to TFLite with INT8 quantization
python convert_tflite.py
```

This uses a representative dataset for calibration to ensure accurate INT8 quantization. The output `.tflite` model can be deployed with:
- **i.MX8M Plus**: TFLite with Vivante NPU delegate
- **i.MX93**: TFLite with Ethos-U delegate via [eIQ](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ)

Note: Transformer-based models like RF-DETR may have limited NPU operator coverage. Profile on target hardware and expect some CPU fallback for unsupported ops.

### 6. Test Inference with OCR

```bash
# Install OCR dependency (requires v1.0.0+)
pip install "fast-plate-ocr[onnx-gpu]>=1.0.0"

# Run detection + OCR on sample images
python test_inference.py
```

The test script uses the CCT (Compact Convolutional Transformer) model for OCR, which provides good accuracy on global license plates. CLAHE preprocessing is applied for contrast enhancement.

Output images with bounding boxes and plate numbers will be saved to `./test_output/`.

## Pre-trained Models

Pre-trained models are available on HuggingFace:

```python
from huggingface_hub import hf_hub_download

# Download INT8 TensorRT engine
engine = hf_hub_download(
    repo_id="autolane/rfdetr-alpr",
    filename="license_plate_detector_int8.engine"
)
```

See: https://huggingface.co/autolane/rfdetr-alpr

## Requirements

```bash
pip install rfdetr tensorrt pycuda onnx onnx-simplifier "fast-plate-ocr[onnx-gpu]>=1.0.0"
```
