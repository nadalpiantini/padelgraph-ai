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

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from padelgraph_ai.schemas import SyncedFrameBatch

__all__ = ["MultiCamSync"]


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

    @property
    def cam_ids(self) -> list[str]:
        """Camera identifiers (``cam1``, ``cam2``, ...) in input order."""
        return list(self._cam_ids)

    @property
    def target_fps(self) -> float:
        """Effective sampling rate of the aligned output."""
        return self._target_fps

    def align(self) -> Iterator[SyncedFrameBatch]:
        """Yield :class:`SyncedFrameBatch` instances at ``1/target_fps`` spacing.

        Each batch is keyed by camera id. Cameras whose video has ended are
        absent from the ``frames`` dict (never present with a ``None`` value).
        Iteration stops when every camera has ended.
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

                    cap.set(cv2.CAP_PROP_POS_MSEC, target_ts * 1000.0)
                    ok, frame = cap.read()
                    if not ok or frame is None:
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
