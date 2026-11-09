"""Bounded, contiguous camera frames for delayed background results."""
import numpy as np


class FrameTrail:
    def __init__(self, source, max_frames=32):
        self.frames = [source]
        self.max_frames = max_frames
        self.complete = True

    def append(self, frame):
        if not self.complete or np.array_equal(self.frames[-1], frame):
            return
        if frame.shape != self.frames[0].shape or len(self.frames) >= self.max_frames:
            # Never silently skip an intermediate frame or fall back to a big jump.
            self.complete = False
            self.frames = []
            return
        self.frames.append(frame.copy())
