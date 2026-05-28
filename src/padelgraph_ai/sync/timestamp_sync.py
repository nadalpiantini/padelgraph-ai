"""Multi-camera timestamp synchronization.

Given N video files (each from a phone at a different angle of the same scene)
plus optional per-camera time offsets, ``MultiCamSync`` yields
:class:`~padelgraph_ai.schemas.SyncedFrameBatch` instances — one per common
timestamp — each containing the corresponding frame from every camera that is
still producing frames at that moment.

Design notes
------------
- **Lazy iteration.** Frames are read on demand using
  :class:`cv2.VideoCapture`; the full videos are never loaded into memory.
- **Seek by timestamp.** Each camera is positioned with
  ``CAP_PROP_POS_MSEC`` per batch — this is simpler than custom frame
  interpolation and accurate enough for MVP (story-003 ``Notes`` section).
- **Per-camera offsets.** ``offsets_seconds[i]`` is added to the common
  timeline before seeking that camera. Use to compensate for known clock
  skew between phones.
- **Ended cameras.** When a camera's video ends (seek returns no frame), its
  key is *omitted* from the ``frames`` dict — never set to ``None``. This
  matches the ``SyncedFrameBatch`` schema contract.
- **Termination.** The iterator stops once every camera has ended.

Apache 2.0.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from padelgraph_ai.schemas import SyncedFrameBatch

__all__ = ["MultiCamSync"]

# F3: how aggressively the sync module retries a transient seek+read
# failure before giving up on the frame. Two attempts (1 initial + 1
# retry) with a small backoff is enough to absorb the codec-buffer hiccup
# we hit in practice without paying noticeable wall-clock cost on the
# happy path (real seek failures stay rare).
_SEEK_RETRY_ATTEMPTS: int = 2
_SEEK_RETRY_BACKOFF_S: float = 0.1
# Above this fraction (per camera, of *attempted* seeks) we surface a
# warning in MultiCamSync.report_seek_stats() — below it we stay silent.
_SEEK_FAILURE_WARN_THRESHOLD: float = 0.05


class MultiCamSync:
    """Align N video streams onto a common timestamp grid.

    Parameters
    ----------
    video_paths
        Ordered list of video file paths. Cameras are keyed ``cam1``,
        ``cam2``, ... by their index in this list.
    offsets_seconds
        Optional per-camera offset added to the common timestamp before
        seeking. Length must match ``video_paths``. Defaults to ``[0.0, ...]``.
    target_fps
        Optional sampling rate for the output batches. Defaults to the
        minimum FPS across input videos.

    Raises
    ------
    FileNotFoundError
        If a video path does not exist.
    ValueError
        If ``offsets_seconds`` length does not match ``video_paths``, if
        no videos are provided, or if a video cannot be opened.
    """

    def __init__(
        self,
        video_paths: list[Path],
        offsets_seconds: list[float] | None = None,
        target_fps: float | None = None,
    ) -> None:
        if not video_paths:
            raise ValueError("video_paths must contain at least one entry")

        self._video_paths: list[Path] = [Path(p) for p in video_paths]
        for p in self._video_paths:
            if not p.exists():
                raise FileNotFoundError(f"Video not found: {p}")

        if offsets_seconds is None:
            offsets_seconds = [0.0] * len(self._video_paths)
        if len(offsets_seconds) != len(self._video_paths):
            raise ValueError(
                f"offsets_seconds length ({len(offsets_seconds)}) must match "
                f"video_paths length ({len(self._video_paths)})"
            )
        self._offsets_seconds: list[float] = list(offsets_seconds)

        # Probe each video for FPS and duration so we can decide the target
        # sampling rate and termination condition without opening every file
        # for a full read.
        self._fps_per_cam: list[float] = []
        self._duration_per_cam: list[float] = []
        for path in self._video_paths:
            cap = cv2.VideoCapture(str(path))
            try:
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video: {path}")
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                if fps <= 0:
                    raise ValueError(f"Invalid FPS reported for {path}: {fps}")
                self._fps_per_cam.append(fps)
                self._duration_per_cam.append(frame_count / fps if frame_count > 0 else 0.0)
            finally:
                cap.release()

        if target_fps is not None and target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        self._target_fps: float = (
            float(target_fps) if target_fps is not None else min(self._fps_per_cam)
        )

        self._cam_ids: list[str] = [f"cam{i + 1}" for i in range(len(self._video_paths))]

        # F3: per-camera seek bookkeeping. ``_seek_attempts`` counts every
        # call to ``cap.set + cap.read`` (including retries). ``_seek_failures``
        # counts the calls that still failed after all retries — i.e. the
        # ones the caller actually sees as a missing frame. The ratio of
        # the two is what ``report_seek_stats`` decides on.
        self._seek_attempts: dict[str, int] = {cam_id: 0 for cam_id in self._cam_ids}
        self._seek_failures: dict[str, int] = {cam_id: 0 for cam_id in self._cam_ids}

    @property
    def cam_ids(self) -> list[str]:
        """Camera identifiers (``cam1``, ``cam2``, ...) in input order."""
        return list(self._cam_ids)

    @property
    def target_fps(self) -> float:
        """Effective sampling rate of the aligned output."""
        return self._target_fps

    @property
    def seek_stats(self) -> dict[str, tuple[int, int]]:
        """Return ``{cam_id: (failures, attempts)}`` accumulated by ``align``.

        ``attempts`` counts every ``cap.set + cap.read`` call (including
        retries). ``failures`` counts those that still failed after all
        retries — i.e. frames the caller saw as missing.
        """
        return {
            cam_id: (self._seek_failures[cam_id], self._seek_attempts[cam_id])
            for cam_id in self._cam_ids
        }

    def align(self) -> Iterator[SyncedFrameBatch]:
        """Yield :class:`SyncedFrameBatch` instances at ``1/target_fps`` spacing.

        Each batch is keyed by camera id. Cameras whose video has ended are
        absent from the ``frames`` dict (never present with a ``None`` value).
        Iteration stops when every camera has ended.

        Per-camera seek failures are retried up to
        ``_SEEK_RETRY_ATTEMPTS`` times with a short backoff to absorb
        codec-buffer hiccups; persistent failures are recorded in
        :pyattr:`seek_stats` and surfaced by :py:meth:`report_seek_stats`.
        """
        captures: list[cv2.VideoCapture] = [cv2.VideoCapture(str(p)) for p in self._video_paths]
        try:
            for cap, path in zip(captures, self._video_paths, strict=True):
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video: {path}")

            ended: list[bool] = [False] * len(captures)
            dt = 1.0 / self._target_fps
            frame_index = 0
            ts = 0.0

            while not all(ended):
                frames: dict[str, np.ndarray] = {}
                for i, cap in enumerate(captures):
                    if ended[i]:
                        continue

                    target_ts = ts + self._offsets_seconds[i]
                    if target_ts < 0:
                        # Camera offset puts us before its timeline start.
                        # Skip this cam for this batch but keep it alive.
                        continue
                    if self._duration_per_cam[i] > 0 and target_ts >= self._duration_per_cam[i]:
                        ended[i] = True
                        continue

                    frame = self._seek_and_read(cap, self._cam_ids[i], target_ts)
                    if frame is None:
                        # Persistent seek failure after retries — treat as
                        # end-of-stream for this camera. The counters
                        # were already updated by ``_seek_and_read``.
                        ended[i] = True
                        continue
                    frames[self._cam_ids[i]] = frame

                if frames:
                    yield SyncedFrameBatch(
                        ts=ts,
                        frame_index=frame_index,
                        frames=frames,
                    )

                frame_index += 1
                ts += dt
        finally:
            for cap in captures:
                cap.release()

    def _seek_and_read(
        self,
        cap: cv2.VideoCapture,
        cam_id: str,
        target_ts: float,
    ) -> np.ndarray | None:
        """Seek ``cap`` to ``target_ts`` seconds and return the frame.

        Retries up to ``_SEEK_RETRY_ATTEMPTS`` times with a short backoff
        between attempts. Every attempt is reflected in
        :pyattr:`seek_stats`, with the final outcome (success or
        persistent failure) recorded in the per-camera counters.
        """
        last_frame: np.ndarray | None = None
        for attempt in range(_SEEK_RETRY_ATTEMPTS):
            self._seek_attempts[cam_id] += 1
            cap.set(cv2.CAP_PROP_POS_MSEC, target_ts * 1000.0)
            ok, frame = cap.read()
            if ok and frame is not None:
                last_frame = frame
                break
            if attempt + 1 < _SEEK_RETRY_ATTEMPTS:
                time.sleep(_SEEK_RETRY_BACKOFF_S)

        if last_frame is None:
            self._seek_failures[cam_id] += 1
        return last_frame

    def report_seek_stats(self) -> list[str]:
        """Return a list of human-readable warnings for cameras over threshold.

        A camera is reported when more than
        ``_SEEK_FAILURE_WARN_THRESHOLD`` of its seek+read attempts
        ultimately failed (after retries). Cameras under threshold are
        silent so the orchestrator only logs when something is actually
        wrong. Returns an empty list when every camera is healthy.
        """
        messages: list[str] = []
        for cam_id in self._cam_ids:
            attempts = self._seek_attempts[cam_id]
            failures = self._seek_failures[cam_id]
            if attempts == 0:
                continue
            failure_rate = failures / attempts
            if failure_rate > _SEEK_FAILURE_WARN_THRESHOLD:
                messages.append(
                    f"WARNING: {cam_id} had {failures}/{attempts} seek failures "
                    f"({failure_rate * 100:.1f}%) after retries — output for this "
                    "camera is incomplete; check codec / file integrity."
                )
        return messages
