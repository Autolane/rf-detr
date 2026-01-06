#!/usr/bin/env python3
"""
Merge multiple COCO-format datasets into a unified dataset for RF-DETR training.

Usage:
    1. Download your COCO-format datasets
    2. Update DATASET_DIRS with paths to your datasets
    3. Run: python prepare_dataset.py
"""

import os
import json
import shutil
from pathlib import Path

# =============================================================================
# CONFIGURATION - Update these paths for your datasets
# =============================================================================

DATASET_DIRS = [
    "./dataset1.coco",
    "./dataset2.coco",
]

OUTPUT_DIR = "./merged_dataset"

# Target category (all source categories will be mapped to this)
TARGET_CATEGORY = "license_plate"

# =============================================================================
# DATASET MERGING
# =============================================================================

def merge_coco_datasets(dataset_dirs: list, output_dir: str, target_category: str):
    """Merge multiple COCO datasets into one unified dataset."""

    output_dir = Path(output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ['train', 'valid', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        merged = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": target_category, "supercategory": "none"}]
        }

        image_id_offset = 0
        annotation_id_offset = 0
        file_counter = 0

        for dataset_idx, dataset_dir in enumerate(dataset_dirs):
            dataset_path = Path(dataset_dir) / split

            if not dataset_path.exists():
                print(f"  Skipping {dataset_dir}/{split} (not found)")
                continue

            # Find annotation file
            ann_file = None
            for possible_name in ["_annotations.coco.json", "annotations.json"]:
                if (dataset_path / possible_name).exists():
                    ann_file = dataset_path / possible_name
                    break

            if ann_file is None:
                print(f"  Skipping {dataset_dir}/{split} (no annotations found)")
                continue

            with open(ann_file, 'r') as f:
                data = json.load(f)

            print(f"  Processing {dataset_dir}/{split}: {len(data['images'])} images")

            old_to_new_image_id = {}

            for img in data['images']:
                old_id = img['id']
                new_id = image_id_offset + len(old_to_new_image_id)
                old_to_new_image_id[old_id] = new_id

                old_filename = img['file_name']
                new_filename = f"ds{dataset_idx}_{file_counter}_{old_filename}"
                file_counter += 1

                src = dataset_path / old_filename
                dst = output_dir / split / new_filename

                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    print(f"    Warning: {src} not found")
                    continue

                merged['images'].append({
                    "id": new_id,
                    "file_name": new_filename,
                    "width": img.get("width", 0),
                    "height": img.get("height", 0)
                })

            for ann in data['annotations']:
                if ann['image_id'] not in old_to_new_image_id:
                    continue

                merged['annotations'].append({
                    "id": annotation_id_offset,
                    "image_id": old_to_new_image_id[ann['image_id']],
                    "category_id": 0,
                    "bbox": ann['bbox'],
                    "area": ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                    "iscrowd": ann.get('iscrowd', 0)
                })
                annotation_id_offset += 1

            image_id_offset = len(merged['images'])

        ann_output = output_dir / split / "_annotations.coco.json"
        with open(ann_output, 'w') as f:
            json.dump(merged, f, indent=2)

        print(f"  {split}: {len(merged['images'])} images, {len(merged['annotations'])} annotations")

    print(f"\nMerged dataset saved to: {output_dir}")


def validate_dataset(dataset_dir: str):
    """Validate the merged dataset is ready for RF-DETR training."""

    dataset_dir = Path(dataset_dir)
    print(f"\nValidating dataset: {dataset_dir}")

    total_images = 0
    total_annotations = 0

    for split in ['train', 'valid', 'test']:
        split_dir = dataset_dir / split
        ann_file = split_dir / "_annotations.coco.json"

        if not ann_file.exists():
            print(f"  [MISSING] {split}: Missing annotations file")
            continue

        with open(ann_file, 'r') as f:
            data = json.load(f)

        num_images = len(data['images'])
        num_annotations = len(data['annotations'])

        missing = sum(1 for img in data['images'] if not (split_dir / img['file_name']).exists())

        status = "[OK]" if missing == 0 else "[WARN]"
        print(f"  {status} {split}: {num_images} images, {num_annotations} annotations", end="")
        if missing > 0:
            print(f" ({missing} files missing)")
        else:
            print()

        total_images += num_images
        total_annotations += num_annotations

    print(f"\n  Total: {total_images} images, {total_annotations} annotations")

    return total_images > 0


if __name__ == "__main__":
    print("=" * 60)
    print("Checking source datasets")
    print("=" * 60)
    for d in DATASET_DIRS:
        if Path(d).exists():
            print(f"  [OK] Found: {d}")
        else:
            print(f"  [MISSING] {d}")

    print("\n" + "=" * 60)
    print("Merging datasets")
    print("=" * 60)

    merge_coco_datasets(DATASET_DIRS, OUTPUT_DIR, TARGET_CATEGORY)

    print("\n" + "=" * 60)
    print("Validating merged dataset")
    print("=" * 60)

    if validate_dataset(OUTPUT_DIR):
        print(f"\nDataset ready for training!")
        print(f"Use: dataset_dir='{OUTPUT_DIR}' in your training script")
    else:
        print("\nDataset validation failed - check errors above")
