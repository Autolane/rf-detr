# ALPR (Automatic License Plate Recognition) Example

This example demonstrates training RF-DETR for license plate detection and exporting to TensorRT for edge deployment.

## Pipeline Overview

1. **Prepare Dataset** - Merge COCO-format datasets
2. **Train Model** - Fine-tune RF-DETR Medium
3. **Export ONNX** - Convert to ONNX format
4. **Optimize ONNX** - Simplify graph (optional)
5. **Build TensorRT** - Create FP16/INT8 engines

## Scripts

| Script | Description |
|--------|-------------|
| `prepare_dataset.py` | Merge multiple COCO datasets into one |
| `train.py` | Train RF-DETR with W&B/TensorBoard logging |
| `export_onnx.py` | Export trained model to ONNX |
| `calibrate_int8.py` | Build TensorRT INT8 engine with calibration |
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

### 6. Test Inference with OCR

```bash
# Install OCR dependency
pip install fast-plate-ocr[onnx-gpu]

# Run detection + OCR on sample images
python test_inference.py
```

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
pip install rfdetr tensorrt pycuda onnx onnx-simplifier fast-plate-ocr[onnx-gpu]
```
