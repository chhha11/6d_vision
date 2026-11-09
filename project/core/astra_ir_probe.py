"""Capture an unmirrored Astra IR burst for RGB-IR calibration."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from openni import openni2

from .paths import CALIBRATION_ROOT, SDK_ROOT

DEFAULT_SDK = SDK_ROOT
DEFAULT_OUTPUT = CALIBRATION_ROOT / "rgb_ir_probe"
BOARD = (9, 6)


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def normalize_ir(ir: np.ndarray) -> np.ndarray:
    valid = ir[ir > 0]
    if not valid.size:
        return np.zeros(ir.shape, dtype=np.uint8)
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((ir.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def find_board(gray: np.ndarray) -> tuple[bool, np.ndarray | None, str]:
    flags = (cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
             | cv2.CALIB_CB_ACCURACY)
    variants = {
        "normalized": gray,
        "inverted": cv2.bitwise_not(gray),
        "clahe": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
    }
    for name, image in variants.items():
        found, corners = cv2.findChessboardCornersSB(image, BOARD, flags)
        if found:
            return True, corners, name
    return False, None, "none"


def capture(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initialized = False
    device = None
    stream = None
    try:
        openni2.initialize(str(args.sdk.resolve()))
        initialized = True
        if not list(openni2.Device.enumerate_uris()):
            raise RuntimeError("No OpenNI2 device found.")
        device = openni2.Device.open_any()
        if not device.has_sensor(openni2.SENSOR_IR):
            raise RuntimeError("The OpenNI2 device does not expose an IR sensor.")
        device_info = device.get_device_info()
        stream = device.create_ir_stream()
        mirroring_default = bool(stream.get_mirroring_enabled())
        stream.set_mirroring_enabled(False)
        mirroring_active = bool(stream.get_mirroring_enabled())
        stream.start()

        for _ in range(args.warmup):
            frame = stream.read_frame()
            frame.close()

        frames = []
        frame_indices = []
        for _ in range(args.frames):
            frame = stream.read_frame()
            ir = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16).copy()
            ir = ir.reshape(frame.height, frame.width)
            frames.append(ir)
            frame_indices.append(int(frame.frameIndex))
            frame.close()

        stack = np.stack(frames)
        median = np.median(stack, axis=0).astype(np.uint16)
        preview = normalize_ir(median)
        found, corners, variant = find_board(preview)
        annotated = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        if found:
            cv2.drawChessboardCorners(annotated, BOARD, corners, True)
        mode = stream.get_video_mode()
        metadata = {
            "captured_at": datetime.now().astimezone().isoformat(),
            "device": {
                "name": device_info.name.decode(errors="replace"),
                "vendor": device_info.vendor.decode(errors="replace"),
                "uri": device_info.uri.decode(errors="replace"),
            },
            "video_mode": {
                "width": int(mode.resolutionX),
                "height": int(mode.resolutionY),
                "fps": int(mode.fps),
                "pixel_format": str(mode.pixelFormat),
            },
            "ir_mirroring": {
                "driver_default": mirroring_default,
                "active": mirroring_active,
            },
            "warmup_frames": int(args.warmup),
            "captured_frames": int(args.frames),
            "device_frame_indices": frame_indices,
            "board_size": list(BOARD),
            "checkerboard_found": bool(found),
            "checkerboard_corners": int(len(corners)) if found else 0,
            "detection_variant": variant,
        }
        np.savez_compressed(output / "ir_burst.npz", ir_frames=stack, median_ir=median)
        write_png(output / "ir_median_uint16.png", median)
        write_png(output / "ir_median_preview.png", preview)
        write_png(output / "ir_checkerboard_preview.png", annotated)
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return output
    finally:
        if stream is not None:
            stream.close()
        if device is not None:
            device.close()
        if initialized:
            openni2.unload()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.frames < 1 or args.frames % 2 == 0:
        parser.error("--frames must be a positive odd number")
    return args


def main() -> None:
    started = time.perf_counter()
    gc.disable()
    try:
        output = capture(parse_args())
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    print(f"Saved IR probe to {output}", flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.2f} s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
