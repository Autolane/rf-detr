#!/usr/bin/env python3
"""
Test RF-DETR license plate detection with OCR.

Runs detection on sample images, draws bounding boxes,
and overlays OCR results using fast-plate-ocr.
"""
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from rfdetr import RFDETRMedium
from fast_plate_ocr import LicensePlateRecognizer

# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINT_PATH = "./output/license_plate_detector/checkpoint_best_ema.pth"
IMAGE_DIR = "./merged_license_plates/valid"
OUTPUT_DIR = "./test_output"
NUM_IMAGES = 20
CONFIDENCE_THRESHOLD = 0.5
RESOLUTION = 576

# =============================================================================
# SETUP
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load RF-DETR model
print("Loading RF-DETR model...")
detector = RFDETRMedium(
    resolution=RESOLUTION,
    num_classes=1,
    pretrain_weights=CHECKPOINT_PATH
)

# Load OCR model (CCT for better accuracy on US plates)
# Options: 'cct-xs-v1-global-model' (faster), 'cct-s-v1-global-model' (more accurate)
OCR_MODEL = 'cct-xs-v1-global-model'
print(f"Loading OCR model ({OCR_MODEL})...")
ocr = LicensePlateRecognizer(hub_ocr_model=OCR_MODEL, device='auto')

# Get image files
image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
random.seed(42)
selected_images = random.sample(image_files, min(NUM_IMAGES, len(image_files)))

print(f"Processing {len(selected_images)} images...")

# =============================================================================
# INFERENCE
# =============================================================================

def draw_detection(image, bbox, plate_text, confidence):
    """Draw bounding box and plate text on image."""
    x1, y1, x2, y2 = map(int, bbox)

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Prepare label
    label = f"{plate_text} ({confidence:.2f})"

    # Get text size for background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw background rectangle for text
    cv2.rectangle(
        image,
        (x1, y1 - text_height - 10),
        (x1 + text_width + 10, y1),
        (0, 255, 0),
        -1
    )

    # Draw text
    cv2.putText(
        image,
        label,
        (x1 + 5, y1 - 5),
        font,
        font_scale,
        (0, 0, 0),
        thickness
    )

    return image


results = []

for idx, image_file in enumerate(selected_images):
    image_path = os.path.join(IMAGE_DIR, image_file)

    # Load image
    image = Image.open(image_path).convert('RGB')
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run detection
    detections = detector.predict(image, threshold=CONFIDENCE_THRESHOLD)

    num_detections = len(detections.xyxy) if hasattr(detections, 'xyxy') else 0

    print(f"[{idx+1}/{len(selected_images)}] {image_file}: {num_detections} plates detected")

    # Process each detection
    if num_detections > 0:
        for i, (bbox, conf) in enumerate(zip(detections.xyxy, detections.confidence)):
            x1, y1, x2, y2 = map(int, bbox)

            # Ensure valid crop coordinates
            h, w = image_cv.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop plate region
            plate_crop = image_cv[y1:y2, x1:x2]

            # Run OCR with preprocessing
            try:
                # Convert BGR to RGB for the model
                plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)

                # Apply CLAHE for contrast enhancement (on each channel)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                plate_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

                # Run OCR (model handles resize/color conversion internally)
                plate_text = ocr.run(plate_enhanced)
                if isinstance(plate_text, list):
                    plate_text = plate_text[0] if plate_text else "N/A"
                plate_text = str(plate_text).strip().rstrip('_')
            except Exception as e:
                print(f"    OCR error: {e}")
                plate_text = "OCR_ERROR"

            # Draw on image
            image_cv = draw_detection(image_cv, (x1, y1, x2, y2), plate_text, conf)

            results.append({
                "image": image_file,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf),
                "plate_text": plate_text
            })

    # Save annotated image
    output_path = os.path.join(OUTPUT_DIR, f"result_{idx:02d}_{image_file}")
    cv2.imwrite(output_path, image_cv)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print(f"\nProcessed: {len(selected_images)} images")
print(f"Total detections: {len(results)}")
print(f"Output directory: {OUTPUT_DIR}")

print("\nDetections:")
print("-" * 60)
for r in results:
    print(f"  {r['image'][:30]:30s} | {r['plate_text']:15s} | conf: {r['confidence']:.2f}")

print("\nDone!")
