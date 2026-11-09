"""Isolate optional YOLO imports and OpenCV global settings from 3D search.

Only the existing background worker waits on this process. Camera acquisition,
preview and metric pose search never import Ultralytics or wait for inference.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import time

import numpy as np


class DetectorProcess:
    def __init__(self, model_path, timings=None, timeout_s=15., command=None):
        self.timeout_s = timeout_s
        self.process = None
        env = dict(os.environ)
        project = str(Path(__file__).resolve().parents[1])
        env['PYTHONPATH'] = os.pathsep.join([project, env.get('PYTHONPATH', '')])
        try:
            self.process = subprocess.Popen(
                command or [sys.executable, '-m', 'core.detector_process', str(model_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
                env=env, bufsize=0)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            reply = self._receive(time.monotonic() + timeout_s)
            if not reply.get('ready'):
                raise RuntimeError(reply.get('error', 'YOLO worker did not become ready'))
            if timings is not None:
                timings.update(reply.get('timings', {}))
        except BaseException:
            self.close()
            raise

    def _read(self, size, deadline):
        data = bytearray()
        while len(data) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError('YOLO worker response timeout')
            block = os.read(self.process.stdout.fileno(), size - len(data))
            if not block:
                raise RuntimeError('YOLO worker stopped')
            data.extend(block)
        return bytes(data)

    def _receive(self, deadline):
        size, = struct.unpack('<I', self._read(4, deadline))
        if size > 65536:
            raise RuntimeError('Invalid YOLO worker reply size')
        reply = json.loads(self._read(size, deadline))
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply

    def _write(self, payload, deadline):
        view = memoryview(payload)
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                raise TimeoutError('YOLO worker input timeout')
            count = os.write(self.process.stdin.fileno(), view)
            if count <= 0:
                raise RuntimeError('YOLO worker input closed')
            view = view[count:]

    def get_box(self, frame, fallback_rotation=2):
        return self._request(frame, fallback_rotation, multiple=False)

    def get_boxes(self, frame, fallback_rotation=2, *, progressive=False):
        return self._request(frame, fallback_rotation, multiple=True, progressive=progressive)

    def _request(self, frame, fallback_rotation, multiple, progressive=False):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('YOLO worker expects an HxWx3 uint8 BGR frame')
        h, w = frame.shape[:2]
        if not 0 < h <= 4096 or not 0 < w <= 4096:
            raise ValueError('YOLO frame dimensions are out of bounds')
        if fallback_rotation not in (0, 1, 2, 3):
            raise ValueError('YOLO rotation must be 0, 1, 2 or 3')
        deadline = time.monotonic() + self.timeout_s
        try:
            self._write(struct.pack('<III', h, w, int(fallback_rotation) + (8 if progressive and multiple else 4 * multiple)) +
                        np.ascontiguousarray(frame).tobytes(), deadline)
            reply = self._receive(deadline)
            if multiple:
                raw = reply.get('boxes')
                if not isinstance(raw, list):
                    raise RuntimeError('Invalid YOLO worker boxes')
                boxes = np.asarray(raw, np.float32)
                if not raw:
                    return []
                if (boxes.ndim != 2 or boxes.shape[1] != 4
                        or not np.isfinite(boxes).all()
                        or np.any(boxes[:, 2:] <= boxes[:, :2])
                        or np.any(boxes < 0)
                        or np.any(boxes > [w, h, w, h])):
                    raise RuntimeError('Invalid YOLO worker boxes')
                return [box.copy() for box in boxes]
            box = reply.get('box')
            if box is None:
                return None
            box = np.asarray(box, np.float32)
            if box.shape != (4,) or not np.isfinite(box).all():
                raise RuntimeError('Invalid YOLO worker box')
            return box
        except BaseException:
            self.close()
            raise

    def close(self):
        process = self.process
        if process is None:
            return
        self.process = None
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.)
        for pipe in (process.stdin, process.stdout):
            if pipe is not None:
                pipe.close()


def worker(model_path):
    # Retain the binary stdout pipe before redirecting library progress output.
    protocol = sys.stdout.buffer
    sys.stdout = sys.stderr
    def send(reply):
        payload = json.dumps(reply).encode('utf-8')
        protocol.write(struct.pack('<I', len(payload)) + payload)
        protocol.flush()
    def read_exact(size):
        data = bytearray()
        while len(data) < size:
            block = sys.stdin.buffer.read(size - len(data))
            if not block:
                return None
            data.extend(block)
        return data
    try:
        started = time.perf_counter()
        from .live_detector import prepare_detector
        timings = {'detector_import_s': round(time.perf_counter() - started, 4)}
        model = prepare_detector(Path(model_path), timings)
        from .live_depth_6dof import detect_target_box
        from .multi_target_detection import detect_target_boxes
        send(dict(ready=True, timings=timings))
        while True:
            header = read_exact(12)
            if header is None:
                return
            h, w, turn = struct.unpack('<III', header)
            if not 0 < h <= 4096 or not 0 < w <= 4096 or turn > 11:
                raise ValueError('Invalid YOLO frame header')
            data = read_exact(h * w * 3)
            if data is None:
                return
            frame = np.frombuffer(data, np.uint8).reshape(h, w, 3)
            if turn >= 4:
                boxes = detect_target_boxes(model, frame, turn % 4, progressive=turn >= 8)
                send(dict(boxes=[box.tolist() for box in boxes]))
            else:
                box = detect_target_box(model, frame, turn)
                send(dict(box=box.tolist() if box is not None else None))
    except Exception as exc:
        send(dict(error=f'{type(exc).__name__}: {exc}'))


if __name__ == '__main__':
    worker(sys.argv[1])
