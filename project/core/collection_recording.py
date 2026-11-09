"""Opt-in whole-session RGB recording, including frames before initialization."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2


MARKERS = {ord('v'): 'target_visible', ord('h'): 'target_hidden',
           ord('u'): 'visible_but_unrecognized', ord('f'): 'visible_drift',
           ord('t'): 'motion_stopped', ord('m'): 'motion_started',
           ord('n'): 'next_sequence'}


class SessionRecording:
    def __init__(self, directory: Path, duration_s: float, fps: float = 30.):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.duration_s, self.fps = duration_s, fps or 30.
        self.started = time.perf_counter()
        self.timestamps = []
        self.writer = None
        self.size = None
        self.closed = False
        self.last_marker = 'V visible | H hidden | U unrecognized | F drift'
        self.log = (self.directory / 'events.jsonl').open('w', encoding='utf-8')
        self.frames = (self.directory / 'frames.jsonl').open('w', encoding='utf-8')
        self.event('recording_started')

    def event(self, name, **details):
        if not self.closed:
            payload = dict(event=name, elapsed_s=time.perf_counter() - self.started,
                           video_frame_index=len(self.timestamps) - 1, **details)
            self.log.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n')
            self.log.flush()

    def write(self, frame):
        if self.closed:
            return
        elapsed = time.perf_counter() - self.started
        if self.writer is None:
            self.size = frame.shape[1::-1]
            self.writer = cv2.VideoWriter(str(self.directory / 'rgb.avi'),
                                         cv2.VideoWriter_fourcc(*'MJPG'), self.fps, self.size)
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise OSError(f'Cannot record RGB to {self.directory}')
        if frame.shape[1::-1] != self.size:
            raise ValueError('Collection frame size changed')
        self.writer.write(frame)
        self.frames.write(json.dumps(dict(video_frame_index=len(self.timestamps),
                                          elapsed_s=elapsed)) + '\n')
        self.timestamps.append(elapsed)
        if len(self.timestamps) % 30 == 0:
            self.frames.flush()

    def pose(self, record):
        # Estimated poses are observations, never reference truth.
        self.event('tracker_observation', estimate=record)

    def key(self, key):
        lower = ord(chr(key).lower()) if 0 <= key < 128 else key
        if lower in MARKERS:
            self.last_marker = MARKERS[lower]
            self.event('user_marker', label=self.last_marker)
            print(f'Collection marker: {self.last_marker}', flush=True)
        if time.perf_counter() - self.started >= self.duration_s:
            return ord('q')
        return key

    def draw(self, image):
        remaining = max(0, self.duration_s - (time.perf_counter() - self.started))
        lines = [f'REC {len(self.timestamps)} frames | {remaining:.0f}s left | Q finish',
                 'V visible H hidden U unrecognized F drift T stop M move N next',
                 self.last_marker]
        for index, line in enumerate(lines):
            y = image.shape[0] - 48 + index * 18
            cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .43,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .43,
                        (60, 240, 255), 1, cv2.LINE_AA)
        return image

    def close(self):
        if self.closed:
            return
        self.event('recording_finished')
        if self.writer is not None:
            self.writer.release()
        self.frames.close()
        self.log.close()
        metadata = dict(video='rgb.avi', image_size=self.size,
                        nominal_video_fps=self.fps, recorded_frames=len(self.timestamps),
                        frame_timestamps_s=self.timestamps,
                        timing_note='Seconds since session start; use timestamps, not AVI FPS. '
                                    'RGB-depth switching produces real gaps; warmup reads are not recorded.',
                        poses='events.jsonl', frames='frames.jsonl',
                        pose_note='Tracker estimates and user markers are not ground truth.')
        (self.directory / 'metadata.json').write_text(
            json.dumps(metadata, indent=2), encoding='utf-8')
        self.closed = True


class RecordedCamera:
    def __init__(self, capture, collection):
        self.capture, self.collection = capture, collection

    def read(self):
        ok, frame = self.capture.read()
        if ok:
            self.collection.write(frame)
        return ok, frame

    def release(self):
        self.collection.event('rgb_stream_released')
        return self.capture.release()

    def __getattr__(self, name):
        return getattr(self.capture, name)


def wrap_camera(cap, args):
    recording = getattr(args, 'collection_recording', None)
    if recording is None or isinstance(cap, RecordedCamera):
        return cap
    recording.event('rgb_stream_opened')
    return RecordedCamera(cap, recording)


def collection_key(cap, key):
    recording = getattr(cap, 'collection', None)
    return recording.key(key) if recording is not None else key


def collection_display(cap, image):
    recording = getattr(cap, 'collection', None)
    return recording.draw(image) if recording is not None else image
