"""Bounded, synchronized evidence for live pose errors and recovery failures."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class TrackingDiagnostics:
    """Keep a rolling, bounded diagnostic recording.

    Video is written to short AVI segments so segments that are completely
    outside the rolling window can be deleted while tracking is still running.
    The public output remains ``rgb.avi`` plus the synchronized ``poses.jsonl``
    and ``metadata.json`` files, created when the recorder closes.
    """

    _SEGMENT_DURATION_S = 1.0

    def __init__(self, directory, K, dist, model, intrinsics, *, record=False,
                 duration_s=60., fps=30., segment_duration_s=None):
        self.directory = Path(directory)
        self.calibration = dict(camera_matrix=np.asarray(K).tolist(),
                                distortion=np.asarray(dist).tolist(),
                                model=str(Path(model).resolve()),
                                intrinsics=str(Path(intrinsics).resolve()),
                                units='mm', coordinate_frame='rgb_camera_optical')
        self.enabled = record
        self.duration_s = float(duration_s)
        self.fps = fps if np.isfinite(fps) and fps > 0 else 30.
        self.segment_duration_s = min(
            float(segment_duration_s or self._SEGMENT_DURATION_S), self.duration_s)
        if not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError('Diagnostic duration must be finite and positive')
        if not np.isfinite(self.segment_duration_s) or self.segment_duration_s <= 0:
            raise ValueError('Diagnostic segment duration must be finite and positive')

        self.writer = None
        self.records = None  # Kept for compatibility with callers that inspect it.
        self.timestamps = []
        self.first_elapsed = None
        self.image_size = None
        self.closed = False
        self._last_relative_time = None
        self._next_frame_index = 0
        self._next_segment_index = 0
        self._segments = []
        self._current_segment = None

    def write(self, frame, record):
        """Record one raw frame and its synchronized pose/status record."""
        if not self.enabled or self.closed:
            return
        elapsed = float(record['elapsed_s'])
        if self.first_elapsed is None:
            self.first_elapsed = elapsed
        relative_time = elapsed - self.first_elapsed
        if (self._last_relative_time is not None
                and relative_time <= self._last_relative_time):
            raise ValueError('Diagnostic frame timestamps must increase')
        self._last_relative_time = relative_time

        if self.image_size is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.image_size = frame.shape[1::-1]
        if frame.shape[1::-1] != self.image_size:
            raise ValueError('Diagnostic frame size changed during recording')

        if (self.writer is None or self._current_segment is None
                or relative_time - self._current_segment['start'] >= self.segment_duration_s):
            self._rotate_segment()
            if not self._open_segment(relative_time):
                return

        payload = dict(record, video_frame_index=self._next_frame_index,
                       video_elapsed_s=relative_time)
        self.writer.write(frame)
        self._current_segment['records'].append(payload)
        self._current_segment['end'] = relative_time
        self._next_frame_index += 1
        self._prune_old_segments(relative_time)

    def snapshot(self, frame, record, overlay=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = f"frame_{int(record['frame_index']):06d}"
        raw_name = stem + '_rgb.png'
        if not cv2.imwrite(str(self.directory / raw_name), frame):
            raise OSError('Cannot save raw diagnostic frame')
        payload = dict(self.calibration, **record, raw_image=raw_name,
                       image_size=list(frame.shape[1::-1]))
        if overlay is not None:
            name = stem + '_overlay.png'
            if not cv2.imwrite(str(self.directory / name), overlay):
                raise OSError('Cannot save diagnostic overlay')
            payload['overlay_image'] = name
        path = self.directory / (stem + '.json')
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        return path

    def _open_segment(self, relative_time):
        self.directory.mkdir(parents=True, exist_ok=True)
        index = self._next_segment_index
        self._next_segment_index += 1
        path = self.directory / f'rgb.part-{index:06d}.avi'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'),
                                 self.fps, self.image_size)
        if not writer.isOpened():
            writer.release()
            path.unlink(missing_ok=True)
            self.writer = None
            self.enabled = False
            print(f'Diagnostic recording unavailable: {self.directory}', flush=True)
            return False
        self.writer = writer
        self._current_segment = dict(path=path, start=relative_time,
                                     end=relative_time, records=[])
        return True

    def _rotate_segment(self):
        if self.writer is None or self._current_segment is None:
            return
        self.writer.release()
        self.writer = None
        if self._current_segment['records']:
            self._segments.append(self._current_segment)
        else:
            self._current_segment['path'].unlink(missing_ok=True)
        self._current_segment = None

    def _prune_old_segments(self, latest_time):
        cutoff = latest_time - self.duration_s
        while self._segments and self._segments[0]['end'] < cutoff:
            segment = self._segments.pop(0)
            segment['path'].unlink(missing_ok=True)

    def _selected_records(self):
        segments = list(self._segments)
        if self._current_segment is not None and self._current_segment['records']:
            segments.append(self._current_segment)
        if not segments:
            return [], []
        latest_time = segments[-1]['records'][-1]['video_elapsed_s']
        cutoff = latest_time - self.duration_s
        selected = []
        for segment in segments:
            selected.extend(record for record in segment['records']
                            if record['video_elapsed_s'] >= cutoff)
        return segments, selected

    def _write_final_video(self, segments, selected):
        output = self.directory / 'rgb.avi'
        selected_ids = {id(record) for record in selected}
        all_records = [record for segment in segments for record in segment['records']]
        if len(selected) == len(all_records) and len(segments) == 1:
            segments[0]['path'].replace(output)
            return

        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'MJPG'),
                                 self.fps, self.image_size)
        if not writer.isOpened():
            writer.release()
            raise OSError(f'Cannot create diagnostic video: {output}')
        written = 0
        try:
            for segment in segments:
                capture = cv2.VideoCapture(str(segment['path']))
                if not capture.isOpened():
                    raise OSError(f'Cannot read diagnostic segment: {segment["path"]}')
                try:
                    for record in segment['records']:
                        ok, frame = capture.read()
                        if not ok:
                            raise OSError(f'Cannot read diagnostic frame: {segment["path"]}')
                        if id(record) in selected_ids:
                            writer.write(frame)
                            written += 1
                finally:
                    capture.release()
        finally:
            writer.release()
        if written != len(selected):
            output.unlink(missing_ok=True)
            raise OSError(f'Diagnostic video frame count mismatch: {written} != {len(selected)}')

    def _metadata(self):
        metadata = dict(self.calibration, image_size=list(self.image_size),
                        frame_timestamps_s=self.timestamps,
                        nominal_video_fps=self.fps,
                        timing_note='Use frame_timestamps_s; AVI playback FPS is nominal.',
                        video='rgb.avi', poses='poses.jsonl',
                        recorded_frames=len(self.timestamps),
                        rolling_window_s=self.duration_s)
        (self.directory / 'metadata.json').write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')

    def close(self):
        if self.closed:
            return
        self._rotate_segment()
        segments, selected = self._selected_records()
        if selected:
            self._write_final_video(segments, selected)
            self.timestamps = [record['video_elapsed_s'] for record in selected]
            with (self.directory / 'poses.jsonl').open('w', encoding='utf-8') as records_file:
                for index, record in enumerate(selected):
                    payload = dict(record, video_frame_index=index)
                    records_file.write(json.dumps(payload, allow_nan=False) + '\n')
            self._metadata()
        for segment in segments:
            segment['path'].unlink(missing_ok=True)
        self._segments.clear()
        self._current_segment = None
        self.records = None
        self.closed = True
