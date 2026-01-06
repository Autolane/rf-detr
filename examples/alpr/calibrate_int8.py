#!/usr/bin/env python3
"""
INT8 Calibration for RF-DETR TensorRT Engine
Generates calibration cache from training images
"""
import tensorrt as trt
import numpy as np
import os
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit

# Configuration
ONNX_PATH = "./export/rfdetr_alpr_optimized.onnx"
ENGINE_PATH = "./export/license_plate_detector_int8.engine"
CALIBRATION_DIR = "./merged_license_plates/valid"  # Use validation images
CACHE_FILE = "./export/calibration.cache"
INPUT_SHAPE = (3, 576, 576)
BATCH_SIZE = 1
MAX_CALIBRATION_IMAGES = 200  # Use up to 200 images for calibration


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_dir, batch_size=1, input_shape=(3, 576, 576), cache_file="calibration.cache"):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.input_shape = input_shape

        # Get list of calibration images
        self.image_files = []
        for f in os.listdir(calibration_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_files.append(os.path.join(calibration_dir, f))

        # Limit number of calibration images
        self.image_files = self.image_files[:MAX_CALIBRATION_IMAGES]
        self.num_images = len(self.image_files)
        self.current_index = 0

        print(f"Found {self.num_images} calibration images")

        # Allocate device memory for input
        size = batch_size * int(np.prod(input_shape)) * np.dtype(np.float32).itemsize
        self.device_input = cuda.mem_alloc(size)
        self.batch_data = np.zeros((batch_size, *input_shape), dtype=np.float32)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= self.num_images:
            return None

        try:
            # Load and preprocess image
            img_path = self.image_files[self.current_index]
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.input_shape[2], self.input_shape[1]), Image.BILINEAR)
            img = np.array(img).astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)  # HWC -> CHW

            self.batch_data[0] = img

            cuda.memcpy_htod(self.device_input, self.batch_data)
            self.current_index += 1

            if self.current_index % 50 == 0:
                print(f"Calibrated {self.current_index}/{self.num_images} images")

            return [int(self.device_input)]
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            self.current_index += 1
            return self.get_batch(names)

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"Reading calibration cache from {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        print(f"Writing calibration cache to {self.cache_file}")
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_int8_engine():
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # Parse ONNX model
    parser = trt.OnnxParser(network, TRT_LOGGER)
    print(f"Parsing ONNX model: {ONNX_PATH}")

    with open(ONNX_PATH, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX Parse Error: {parser.get_error(i)}")
            return None

    print("ONNX model parsed successfully")

    # Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 * 1024 * 1024)  # 4GB

    # Enable INT8 and FP16
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)

    # Create calibrator
    calibrator = Int8Calibrator(
        calibration_dir=CALIBRATION_DIR,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        cache_file=CACHE_FILE
    )
    config.int8_calibrator = calibrator

    # Set optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "images",
        min=(1, 3, 576, 576),
        opt=(1, 3, 576, 576),
        max=(1, 3, 576, 576)
    )
    config.add_optimization_profile(profile)

    # Build engine
    print("Building INT8 TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        return None

    # Save engine
    print(f"Saving engine to {ENGINE_PATH}")
    with open(ENGINE_PATH, 'wb') as f:
        f.write(serialized_engine)

    print("INT8 engine built successfully!")
    return serialized_engine


if __name__ == "__main__":
    # Check calibration directory
    if not os.path.exists(CALIBRATION_DIR):
        # Try alternate path
        alt_dir = "./merged_license_plates/train"
        if os.path.exists(alt_dir):
            CALIBRATION_DIR = alt_dir
        else:
            print(f"Calibration directory not found: {CALIBRATION_DIR}")
            exit(1)

    build_int8_engine()
