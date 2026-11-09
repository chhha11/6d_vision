"""Capture a short Astra depth burst through the bundled OpenNI2 runtime."""
from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from openni import openni2
except ImportError:  # pragma: no cover - depends on the optional legacy SDK
    class _UnavailableOpenNI2:
        """Keep legacy helpers importable when Linux uses the native backend."""

        IMAGE_REGISTRATION_DEPTH_TO_COLOR = "depth_to_color"

        class Device:
            @staticmethod
            def enumerate_uris():
                raise RuntimeError(
                    "The legacy OpenNI2 Python package is not installed. "
                    "Use the native Orbbec RGB-D backend on Linux."
                )

            @staticmethod
            def open_any(*args, **kwargs):
                raise RuntimeError(
                    "The legacy OpenNI2 Python package is not installed. "
                    "Use the native Orbbec RGB-D backend on Linux."
                )

        @staticmethod
        def initialize(*args, **kwargs):
            raise RuntimeError(
                "The legacy OpenNI2 Python package is not installed. "
                "Use the native Orbbec RGB-D backend on Linux."
            )

        @staticmethod
        def wait_for_any_stream(*args, **kwargs):
            raise RuntimeError(
                "The legacy OpenNI2 Python package is not installed. "
                "Use the native Orbbec RGB-D backend on Linux."
            )

        @staticmethod
        def unload(*args, **kwargs):
            return None

    openni2 = _UnavailableOpenNI2()

from .paths import DATA_ROOT, SDK_ROOT

DEFAULT_SDK = SDK_ROOT
DEFAULT_OUTPUT = DATA_ROOT / "tracking" / "depth_probe_current"


def write_progress(output: Path, stage: str, **details: object) -> None:
    """Persist the last native-driver stage before entering an OpenNI call.

    A legacy OpenNI driver can terminate the worker before Python gets a
    chance to write its exception or timing file.  This small atomic marker is
    therefore deliberately written *before* each native operation.
    """
    payload = {
        "stage": stage,
        "updated_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        **details,
    }
    try:
        output.mkdir(parents=True, exist_ok=True)
        temporary = output / "capture_progress.json.tmp"
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        os.replace(temporary, output / "capture_progress.json")
        with (output / "capture_events.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        # Diagnostics must never turn a camera failure into another failure.
        pass


def read_frame_with_timeout(stream, timeout_s: float):
    """Wait for a frame through OpenNI's bounded wait API before reading it."""
    ready = openni2.wait_for_any_stream([stream], timeout=timeout_s)
    if ready is None:
        raise TimeoutError(f"OpenNI depth frame timeout after {timeout_s:.2f} s")
    return stream.read_frame()


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(path)


def depth_stats(depth_mm: np.ndarray) -> dict[str, object]:
    valid = depth_mm[depth_mm > 0]
    stats: dict[str, object] = {
        "valid_pixels": int(valid.size),
        "total_pixels": int(depth_mm.size),
        "valid_ratio": float(valid.size / depth_mm.size),
    }
    if valid.size:
        values = np.percentile(valid, [1, 5, 25, 50, 75, 95, 99])
        stats["depth_mm_percentiles"] = {
            key: float(value)
            for key, value in zip(("p01", "p05", "p25", "p50", "p75", "p95", "p99"), values)
        }
        stats["depth_mm_min"] = int(valid.min())
        stats["depth_mm_max"] = int(valid.max())
    return stats


def make_preview(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm[depth_mm > 0]
    if not valid.size:
        return np.zeros((*depth_mm.shape, 3), dtype=np.uint8)
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((depth_mm.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    gray = np.round((1.0 - normalized) * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    preview[depth_mm == 0] = 0
    return preview


def estimated_intrinsics(width: int, height: int, horizontal_fov: float,
                         vertical_fov: float) -> dict[str, float]:
    return {
        "fx": float(width / (2.0 * math.tan(horizontal_fov / 2.0))),
        "fy": float(height / (2.0 * math.tan(vertical_fov / 2.0))),
        "cx": float((width - 1) / 2.0),
        "cy": float((height - 1) / 2.0),
        "horizontal_fov_rad": float(horizontal_fov),
        "vertical_fov_rad": float(vertical_fov),
    }


def capture(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    timings = {}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    initialized = False
    device = None
    stream = None

    def close_native_resources():
        nonlocal stream, device, initialized
        if stream is not None:
            write_progress(output, "closing_depth_stream")
            stream.close()
            stream = None
        if device is not None:
            write_progress(output, "closing_device")
            device.close()
            device = None
        if initialized:
            write_progress(output, "unloading_openni")
            openni2.unload()
            initialized = False
        write_progress(output, "native_closed")

    try:
        write_progress(output, "initializing_openni", sdk=str(args.sdk.resolve()))
        openni2.initialize(str(args.sdk.resolve()))
        initialized = True
        write_progress(output, "enumerating_devices")
        uris = list(openni2.Device.enumerate_uris())
        if not uris:
            raise RuntimeError("No OpenNI2 device found.")
        write_progress(output, "opening_device", device_count=len(uris))
        device = openni2.Device.open_any()
        write_progress(output, "reading_device_info")
        device_info = device.get_device_info()
        write_progress(output, "configuring_registration")
        registration_supported = device.is_image_registration_mode_supported(
            openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        if args.register_to_color:
            if not registration_supported:
                raise RuntimeError("Depth-to-color registration is not supported by this device.")
            try:
                device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
            except Exception as exc:
                raise RuntimeError(
                    "The driver advertises depth-to-color registration but rejected the setting."
                ) from exc
        try:
            registration_mode = str(device.get_image_registration_mode())
        except Exception:
            registration_mode = "unavailable"
        write_progress(output, "creating_depth_stream")
        stream = device.create_depth_stream()
        mirroring_default = bool(stream.get_mirroring_enabled())
        stream.set_mirroring_enabled(False)
        mirroring_active = bool(stream.get_mirroring_enabled())
        write_progress(output, "starting_depth_stream")
        stream.start()
        timings['open_stream_s'] = time.perf_counter() - started

        for index in range(args.warmup):
            write_progress(output, "waiting_warmup_frame", index=index,
                           timeout_s=args.frame_timeout_s)
            warmup_frame = read_frame_with_timeout(stream, args.frame_timeout_s)
            write_progress(output, "releasing_warmup_frame", index=index)
            warmup_frame.close()
        timings['warmup_complete_s'] = time.perf_counter() - started

        frames = []
        frame_metadata = []
        for index in range(args.frames):
            write_progress(output, "waiting_depth_frame", index=index,
                           timeout_s=args.frame_timeout_s)
            frame = read_frame_with_timeout(stream, args.frame_timeout_s)
            try:
                write_progress(output, "copying_depth_frame", index=index)
                depth = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16).copy()
                depth = depth.reshape(frame.height, frame.width)
                frames.append(depth)
                frame_metadata.append({
                    "index": index,
                    "device_frame_index": int(frame.frameIndex),
                    "device_timestamp": int(frame.timestamp),
                    **depth_stats(depth),
                })
            finally:
                write_progress(output, "releasing_depth_frame", index=index)
                frame.close()
        timings['read_complete_s'] = time.perf_counter() - started

        write_progress(output, "reading_stream_properties")
        mode = stream.get_video_mode()
        horizontal_fov = float(stream.get_horizontal_fov())
        vertical_fov = float(stream.get_vertical_fov())
        # Do not leave USB/native streaming threads active during median,
        # compression, lazy NumPy imports, or image encoding. Frames above are
        # owned copies, so no later processing needs the driver or frame pool.
        close_native_resources()
        timings['native_closed_s'] = time.perf_counter() - started
        write_progress(output, "computing_depth_median")
        stack = np.stack(frames)
        median = np.ma.median(np.ma.masked_equal(stack, 0), axis=0).filled(0).astype(np.uint16)
        height, width = median.shape
        center = median[height // 4: 3 * height // 4, width // 4: 3 * width // 4]
        metadata = {
            "captured_at": datetime.now().astimezone().isoformat(),
            "sdk_path": str(args.sdk.resolve()),
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
                "depth_unit_mm": 1,
            },
            "registration": {
                "requested_depth_to_color": bool(args.register_to_color),
                "depth_to_color_supported": bool(registration_supported),
                "active_mode": registration_mode,
            },
            "depth_mirroring": {
                "driver_default": mirroring_default,
                "active": mirroring_active,
            },
            "intrinsics_estimate": estimated_intrinsics(
                width, height, horizontal_fov, vertical_fov),
            "warmup_frames": int(args.warmup),
            "captured_frames": int(args.frames),
            "frames": frame_metadata,
            "median_frame": depth_stats(median),
            "median_center_50_percent": depth_stats(center),
        }

        write_progress(output, "writing_depth_archive")
        np.savez_compressed(output / "depth_burst.npz", depth_frames_mm=stack,
                            median_depth_mm=median)
        write_progress(output, "writing_depth_previews")
        write_png(output / "depth_median_uint16.png", median)
        write_png(output / "depth_median_preview.png", make_preview(median))
        write_progress(output, "writing_depth_metadata")
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        timings['files_ready_s'] = time.perf_counter() - started
        write_progress(output, "capture_complete", frames=len(frames))
        return output
    finally:
        if stream is not None or device is not None or initialized:
            close_native_resources()
        timings['total_s'] = time.perf_counter() - started
        try:
            (output / 'capture_timing.json').write_text(
                json.dumps(timings, indent=2), encoding='utf-8')
        except OSError:
            # Optional timing must not replace a capture or cleanup exception.
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=DEFAULT_SDK,
                        help="Directory containing OpenNI2.dll and OpenNI2/Drivers")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--frame-timeout-s", type=float, default=2.0,
                        help="Maximum wait for each OpenNI frame")
    parser.add_argument("--register-to-color", action="store_true",
                        help="Ask OpenNI2 to map depth pixels into the color camera frame")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.frames < 1 or args.frames % 2 == 0:
        parser.error("--frames must be a positive odd number")
    if not math.isfinite(args.frame_timeout_s) or args.frame_timeout_s <= 0:
        parser.error("--frame-timeout-s must be positive")
    return args


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    # Keep native fatal tracebacks in worker.log and suppress Windows modal
    # crash dialogs: the parent owns timeout/recovery of this isolated worker.
    faulthandler.enable()
    faulthandler.dump_traceback_later(5.0, repeat=True)
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    # The legacy Orbbec OpenNI 2.3 Windows driver corrupts its heap during
    # interpreter GC after shutdown. Finish this short-lived worker explicitly.
    gc.disable()
    try:
        output = capture(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    print(f"Saved depth probe to {output}", flush=True)
    print(f"Elapsed: {time.perf_counter() - started:.2f} s", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
