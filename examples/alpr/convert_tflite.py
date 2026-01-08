#!/usr/bin/env python3
"""
Convert ONNX model to TFLite INT8 for i.MX93 / i.MX8M Plus NPU.

Pipeline: ONNX -> TensorFlow SavedModel -> TFLite INT8

Requirements:
    pip install onnx onnx-tf tensorflow

Usage:
    python convert_tflite.py
"""
import os
import glob
import numpy as np
import cv2

# =============================================================================
# CONFIGURATION
# =============================================================================

ONNX_MODEL = "./export/rfdetr_alpr.onnx"
OUTPUT_DIR = "./export/tflite"
CALIBRATION_IMAGES = "./merged_license_plates/valid"
NUM_CALIBRATION_SAMPLES = 100
IMAGE_SIZE = 576

# =============================================================================
# CALIBRATION DATA GENERATOR
# =============================================================================

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """Preprocess image for calibration (same as inference)."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top, left = pad_h // 2, pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, pad_h - top, left, pad_w - left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std

    return normalized


def calibration_data_gen():
    """Generator for calibration data."""
    image_paths = glob.glob(f"{CALIBRATION_IMAGES}/*.jpg")[:NUM_CALIBRATION_SAMPLES]

    for path in image_paths:
        data = preprocess_image(path)
        if data is not None:
            # Add batch dimension, keep as NHWC for TFLite
            yield [data[np.newaxis, ...]]


# =============================================================================
# CONVERSION PIPELINE
# =============================================================================

def convert_onnx_to_tflite():
    """Full conversion pipeline: ONNX -> TF -> TFLite INT8."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("ONNX to TFLite INT8 Conversion")
    print("=" * 60)

    # Step 1: ONNX to TensorFlow SavedModel
    print("\n[1/3] Converting ONNX to TensorFlow SavedModel...")
    try:
        import onnx
        from onnx_tf.backend import prepare

        onnx_model = onnx.load(ONNX_MODEL)
        tf_rep = prepare(onnx_model)

        savedmodel_path = os.path.join(OUTPUT_DIR, "saved_model")
        tf_rep.export_graph(savedmodel_path)
        print(f"  Saved to: {savedmodel_path}")

    except ImportError:
        print("ERROR: Install onnx-tf: pip install onnx-tf")
        print("\nAlternative: Use ONNX directly with ONNX Runtime on i.MX")
        return False
    except Exception as e:
        print(f"ERROR: ONNX to TF conversion failed: {e}")
        print("\nNote: RF-DETR may have ops not supported by onnx-tf.")
        print("Consider using ONNX Runtime directly on i.MX8M Plus.")
        return False

    # Step 2: TensorFlow SavedModel to TFLite FP32
    print("\n[2/3] Converting to TFLite FP32...")
    try:
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
        tflite_fp32 = converter.convert()

        fp32_path = os.path.join(OUTPUT_DIR, "rfdetr_alpr_fp32.tflite")
        with open(fp32_path, 'wb') as f:
            f.write(tflite_fp32)
        print(f"  Saved to: {fp32_path}")
        print(f"  Size: {os.path.getsize(fp32_path) / 1024 / 1024:.1f} MB")

    except Exception as e:
        print(f"ERROR: TFLite FP32 conversion failed: {e}")
        return False

    # Step 3: Quantize to INT8
    print("\n[3/3] Quantizing to INT8...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)

        # Full integer quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = calibration_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_int8 = converter.convert()

        int8_path = os.path.join(OUTPUT_DIR, "rfdetr_alpr_int8.tflite")
        with open(int8_path, 'wb') as f:
            f.write(tflite_int8)
        print(f"  Saved to: {int8_path}")
        print(f"  Size: {os.path.getsize(int8_path) / 1024 / 1024:.1f} MB")

    except Exception as e:
        print(f"ERROR: INT8 quantization failed: {e}")
        print("\nFalling back to dynamic range quantization...")

        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            tflite_dynamic = converter.convert()

            dynamic_path = os.path.join(OUTPUT_DIR, "rfdetr_alpr_dynamic.tflite")
            with open(dynamic_path, 'wb') as f:
                f.write(tflite_dynamic)
            print(f"  Saved to: {dynamic_path}")

        except Exception as e2:
            print(f"ERROR: Dynamic quantization also failed: {e2}")
            return False

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    return True


# =============================================================================
# ALTERNATIVE: ONNX INT8 QUANTIZATION
# =============================================================================

def quantize_onnx_int8():
    """Quantize ONNX model to INT8 using ONNX Runtime quantization.

    This is an alternative if TFLite conversion fails.
    Use with ONNX Runtime on i.MX8M Plus.
    """
    print("=" * 60)
    print("ONNX INT8 Quantization (Alternative)")
    print("=" * 60)

    try:
        from onnxruntime.quantization import quantize_static, CalibrationDataReader
        from onnxruntime.quantization import QuantType, QuantFormat
        import onnx

        class CalibrationReader(CalibrationDataReader):
            def __init__(self, image_dir, num_samples=100):
                self.image_paths = glob.glob(f"{image_dir}/*.jpg")[:num_samples]
                self.index = 0

            def get_next(self):
                if self.index >= len(self.image_paths):
                    return None

                data = preprocess_image(self.image_paths[self.index])
                self.index += 1

                if data is None:
                    return self.get_next()

                # NCHW format for ONNX
                tensor = data.transpose(2, 0, 1)[np.newaxis, ...]
                return {'images': tensor.astype(np.float32)}

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        print("\nQuantizing ONNX model to INT8...")
        calibration_reader = CalibrationReader(CALIBRATION_IMAGES, NUM_CALIBRATION_SAMPLES)

        output_path = os.path.join(OUTPUT_DIR, "rfdetr_alpr_int8.onnx")

        quantize_static(
            model_input=ONNX_MODEL,
            model_output=output_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
        )

        print(f"  Saved to: {output_path}")
        print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

        return True

    except ImportError:
        print("ERROR: Install onnxruntime: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"ERROR: ONNX quantization failed: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ONNX to TFLite INT8")
    parser.add_argument("--onnx-only", action="store_true",
                        help="Only do ONNX INT8 quantization (skip TFLite)")
    args = parser.parse_args()

    if args.onnx_only:
        quantize_onnx_int8()
    else:
        # Try TFLite first, fall back to ONNX
        success = convert_onnx_to_tflite()
        if not success:
            print("\n" + "=" * 60)
            print("Trying ONNX INT8 quantization as fallback...")
            print("=" * 60)
            quantize_onnx_int8()
