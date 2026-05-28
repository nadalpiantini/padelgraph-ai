"""Court keypoint detection (manual seed) and 2D homography for padel courts.

For MVP (Epic 1) the court keypoints are picked manually once per camera
position and cached to a JSON sidecar. Auto-detection is explicit non-goal
of the MVP and tracked in Epic 2.

Apache 2.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from padelgraph_ai.schemas import CourtKeypoints

# Default doubles padel court: 20m long × 10m wide.
DEFAULT_COURT_DIM_METERS: tuple[float, float] = (20.0, 10.0)

# Label shown to the user on each click, in the canonical corner order.
_CORNER_LABELS: tuple[str, str, str, str] = (
    "top_left",
    "top_right",
    "bottom_right",
    "bottom_left",
)


class CourtDetector:
    """Detect (or recall) the 4 court corners in a camera frame.

    For MVP the detector is interactive only: it opens an OpenCV window and
    asks the user to click the 4 corners in the canonical order
    (top_left, top_right, bottom_right, bottom_left). The resulting
    keypoints can then be cached to JSON and reloaded on subsequent runs
    via :py:meth:`load_from_cache` / :py:meth:`save_to_cache`.

    Auto-detection from a single frame (no manual seed) is Epic 2 scope and
    intentionally not implemented here.
    """

    def __init__(self, window_name: str = "padelgraph-ai: click 4 corners") -> None:
        self._window_name = window_name

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #

    def detect_keypoints(
        self,
        frame: np.ndarray,
        interactive: bool = False,
    ) -> CourtKeypoints | None:
        """Return the 4 court corners for ``frame``.

        Parameters
        ----------
        frame:
            BGR uint8 frame as produced by ``cv2.VideoCapture.read``.
        interactive:
            When ``True`` open an OpenCV window and let the user click the 4
            corners in canonical order (top_left → top_right → bottom_right
            → bottom_left). When ``False`` (the default in tests/CI) the
            detector has no auto path yet, so it returns ``None``.

        Returns
        -------
        ``CourtKeypoints`` populated with the 4 clicked corners when
        interactive mode is used, otherwise ``None``.
        """

        if not interactive:
            # Auto-detect is Epic 2 scope. Returning None keeps the API stable
            # and lets callers fall back to a cached file.
            return None

        return self._pick_corners_interactive(frame)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_from_cache(cache_path: Path) -> CourtKeypoints:
        """Load previously-picked keypoints from a JSON sidecar."""
        path = Path(cache_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CourtKeypoints.model_validate(payload)

    @staticmethod
    def save_to_cache(keypoints: CourtKeypoints, cache_path: Path) -> None:
        """Persist keypoints to a JSON sidecar (parents are created)."""
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(keypoints.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _pick_corners_interactive(self, frame: np.ndarray) -> CourtKeypoints | None:
        clicks: list[tuple[float, float]] = []
        display = frame.copy()

        def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
            del flags, param  # unused
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if len(clicks) >= 4:
                return
            label = _CORNER_LABELS[len(clicks)]
            clicks.append((float(x), float(y)))
            cv2.circle(display, (x, y), 6, (0, 255, 0), thickness=-1)
            cv2.putText(
                display,
                label,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.imshow(self._window_name, display)

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self._window_name, display)
        cv2.setMouseCallback(self._window_name, _on_mouse)

        try:
            while len(clicks) < 4:
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q")):
                    return None
        finally:
            cv2.destroyWindow(self._window_name)

        return CourtKeypoints(
            top_left=clicks[0],
            top_right=clicks[1],
            bottom_right=clicks[2],
            bottom_left=clicks[3],
        )


class Homography:
    """A 2D perspective transform between pixel and court coordinates.

    Court coordinates are expressed in meters with the origin at the
    top-left corner of the court, the X axis growing toward
    ``top_right`` and the Y axis growing toward ``bottom_left``. Default
    dimensions are doubles padel: 20m long × 10m wide.

    Construct via :py:meth:`from_keypoints` rather than the bare initializer.
    """

    def __init__(self, matrix: np.ndarray, court_dim_meters: tuple[float, float]) -> None:
        if matrix.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got {matrix.shape}")
        self._matrix = np.asarray(matrix, dtype=np.float64)
        self._inverse = np.linalg.inv(self._matrix)
        self._court_dim_meters = court_dim_meters

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_keypoints(
        cls,
        keypoints: CourtKeypoints,
        court_dim_meters: tuple[float, float] = DEFAULT_COURT_DIM_METERS,
    ) -> Homography:
        """Build a homography from 4 court-corner pixel coords.

        The destination quadrilateral is the metric court rectangle ordered
        ``(0, 0) → (W, 0) → (W, H) → (0, H)`` to match the canonical corner
        order ``top_left, top_right, bottom_right, bottom_left``.
        """

        width_m, height_m = court_dim_meters
        src = keypoints.as_array()
        dst = np.asarray(
            [
                (0.0, 0.0),
                (width_m, 0.0),
                (width_m, height_m),
                (0.0, height_m),
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cls(matrix=matrix, court_dim_meters=court_dim_meters)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 perspective transform mapping pixels → court meters."""
        return self._matrix

    @property
    def court_dim_meters(self) -> tuple[float, float]:
        return self._court_dim_meters

    def warp_point(self, pixel_xy: tuple[float, float]) -> tuple[float, float]:
        """Map a pixel coordinate to court coordinates (meters)."""
        return self._apply(self._matrix, pixel_xy)

    def inverse_warp(self, court_xy: tuple[float, float]) -> tuple[float, float]:
        """Map a court coordinate (meters) back to a pixel coordinate."""
        return self._apply(self._inverse, court_xy)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply(
        matrix: np.ndarray,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        x, y = point
        vec = np.array([float(x), float(y), 1.0], dtype=np.float64)
        out = matrix @ vec
        w = out[2]
        if w == 0:
            raise ValueError(
                f"Degenerate perspective division (w=0) for point {point}; "
                "the input is at the homography's vanishing line."
            )
        return (float(out[0] / w), float(out[1] / w))


__all__ = ["CourtDetector", "Homography", "DEFAULT_COURT_DIM_METERS"]
