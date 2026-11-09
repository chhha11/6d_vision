"""YOLO preparation for the live CPU detection path.

Imported in the preparation thread so importing the live entry point does
not have to load Torch or Ultralytics before showing the camera window.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor


class LiveDetectionPredictor(DetectionPredictor):
    def setup_model(self, model, verbose: bool = True):
        super().setup_model(model, verbose)
        # Ultralytics 8.4 imports all of torchvision in AutoBackend.warmup()
        # even for PyTorch CPU, where it does no model forward pass. Recent
        # Ultralytics already provides TorchNMS for a torchvision-free path.
        # Our explicit dummy predict below warms the *real* preprocess,
        # forward, NMS and result construction without that optional import.
        from ultralytics.utils import nms

        if (self.model.device.type == 'cpu'
                and (getattr(self.model, 'format', None) == 'pt'
                     or getattr(self.model, 'pt', False))
                and hasattr(nms, 'TorchNMS')):
            self.done_warmup = True


def prepare_detector(model_path: Path, timings: dict | None = None):
    started = time.perf_counter()
    detector = YOLO(str(model_path))
    if timings is not None:
        timings['detector_weights_s'] = round(time.perf_counter() - started, 4)
    started = time.perf_counter()
    detector.predict(np.zeros((480, 640, 3), np.uint8), conf=.20, imgsz=640,
                     verbose=False, predictor=LiveDetectionPredictor)
    if timings is not None:
        timings['detector_warmup_s'] = round(time.perf_counter() - started, 4)
    return detector
