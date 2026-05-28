"""Shared Pydantic schemas for padelgraph-ai pipeline.

This is the contract between all modules (detection, pose, court, sync,
fusion, pipeline). Modules should import from here rather than redefining
their own shapes.

Apache 2.0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class Detection(BaseModel):
    """A single object detection from one frame of one camera."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    class_id: int = Field(description="COCO class id (0=person, 32=sports ball)")
    class_name: str = Field(description="Human-readable class name")
    bbox: tuple[float, float, float, float] = Field(
        description="Bounding box in pixel coords: (x, y, w, h) with (x, y) top-left"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Detector confidence")
    track_id: int | None = Field(default=None, description="Optional tracker id (epic 2)")


class FrameDetections(BaseModel):
    """All detections in a single frame from a single camera."""

    frame_id: int = Field(ge=0, description="Sequential frame index in the video")
    ts: float = Field(ge=0.0, description="Timestamp in seconds from video start")
    detections: list[Detection] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Multi-camera sync
# ---------------------------------------------------------------------------


class SyncedFrameBatch(BaseModel):
    """Frames from multiple cameras synchronized to one timestamp.

    `frames` is intentionally a dict so cameras can be identified by name
    (cam1, cam2, ...) and a missing camera (one stream ended early) is
    represented by an absent key — not a None value.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ts: float = Field(ge=0.0, description="Common timestamp in seconds")
    frame_index: int = Field(ge=0, description="Logical batch index (not raw frame number)")
    frames: dict[str, np.ndarray] = Field(description="cam_id -> frame as np.ndarray (BGR uint8)")


# ---------------------------------------------------------------------------
# Court detection + homography
# ---------------------------------------------------------------------------


class CourtKeypoints(BaseModel):
    """The 4 court corners in pixel coordinates of one camera frame.

    Order: top_left, top_right, bottom_right, bottom_left (as the player sees
    the court from this camera's angle).
    """

    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]
    bottom_left: tuple[float, float]

    def as_array(self) -> np.ndarray:
        """Return as a (4, 2) numpy float32 array suitable for cv2.getPerspectiveTransform."""
        return np.asarray(
            [self.top_left, self.top_right, self.bottom_right, self.bottom_left],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Camera calibration
# ---------------------------------------------------------------------------


class Calibration(BaseModel):
    """Camera calibration for one camera.

    intrinsics: 3x3 camera matrix (fx, 0, cx; 0, fy, cy; 0, 0, 1)
    extrinsics: 4x4 world-to-camera matrix (rotation + translation)
    distortion_coeffs: optional, OpenCV-style [k1, k2, p1, p2, k3]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cam_id: str
    intrinsics: list[list[float]] = Field(description="3x3 K matrix as nested list")
    extrinsics: list[list[float]] = Field(description="4x4 [R|t] matrix as nested list")
    distortion_coeffs: list[float] | None = Field(default=None, description="OpenCV format")
    image_size: tuple[int, int] = Field(description="(width, height) in pixels")

    def K(self) -> np.ndarray:
        return np.asarray(self.intrinsics, dtype=np.float64)

    def Rt(self) -> np.ndarray:
        return np.asarray(self.extrinsics, dtype=np.float64)


# ---------------------------------------------------------------------------
# 3D output
# ---------------------------------------------------------------------------


class Point3D(BaseModel):
    """A point in court 3D coordinates (meters)."""

    x: float
    y: float
    z: float
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Match-level analysis (final pipeline output)
# ---------------------------------------------------------------------------


class BallObservation(BaseModel):
    """Ball observation in one frame across all cameras."""

    position_2d_per_cam: dict[str, tuple[float, float]] = Field(default_factory=dict)
    position_3d: Point3D | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FrameAnalysis(BaseModel):
    """One frame of analysis after fusion across cameras."""

    frame_id: int = Field(ge=0)
    ts: float = Field(ge=0.0)
    players_per_cam: dict[str, list[Detection]] = Field(default_factory=dict)
    ball: BallObservation = Field(default_factory=BallObservation)


class MatchMeta(BaseModel):
    """Match-level metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    match_id: str
    cameras: list[Calibration]
    fps: float = Field(gt=0.0)
    court_dim_meters: tuple[float, float] = Field(default=(20.0, 10.0))
    court_keypoints_2d_per_cam: dict[str, CourtKeypoints] = Field(default_factory=dict)
    notes: dict[str, Any] = Field(default_factory=dict)


class MatchAnalysis(BaseModel):
    """Top-level pipeline output."""

    meta: MatchMeta
    frames: list[FrameAnalysis] = Field(default_factory=list)


__all__ = [
    "Detection",
    "FrameDetections",
    "SyncedFrameBatch",
    "CourtKeypoints",
    "Calibration",
    "Point3D",
    "BallObservation",
    "FrameAnalysis",
    "MatchMeta",
    "MatchAnalysis",
]
