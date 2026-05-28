"""Unit tests for :mod:`padelgraph_ai.court.homography`.

Interactive corner picking (cv2 window) is intentionally *not* exercised in
CI; we test the cache round-trip and the math instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from padelgraph_ai.court import CourtDetector, Homography
from padelgraph_ai.schemas import CourtKeypoints

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Synthetic keypoints: a perfectly axis-aligned 1000x500 px "court" that
# corresponds 1:1 with the canonical metric court order.
_SYNTHETIC_KEYPOINTS = CourtKeypoints(
    top_left=(0.0, 0.0),
    top_right=(1000.0, 0.0),
    bottom_right=(1000.0, 500.0),
    bottom_left=(0.0, 500.0),
)

# A mildly skewed quadrilateral so the homography actually does something
# non-trivial in the round-trip test.
_SKEWED_KEYPOINTS = CourtKeypoints(
    top_left=(120.0, 80.0),
    top_right=(1180.0, 95.0),
    bottom_right=(1300.0, 690.0),
    bottom_left=(40.0, 710.0),
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_corner_mapping() -> None:
    """The 4 picked corners must map exactly onto the metric court corners."""

    homography = Homography.from_keypoints(_SYNTHETIC_KEYPOINTS)

    assert homography.warp_point((0.0, 0.0)) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert homography.warp_point((1000.0, 0.0)) == pytest.approx((20.0, 0.0), abs=1e-6)
    assert homography.warp_point((1000.0, 500.0)) == pytest.approx((20.0, 10.0), abs=1e-6)
    assert homography.warp_point((0.0, 500.0)) == pytest.approx((0.0, 10.0), abs=1e-6)


def test_homography_round_trip() -> None:
    """warp(p) followed by inverse_warp must return the original pixel within ε."""

    homography = Homography.from_keypoints(_SKEWED_KEYPOINTS)

    # A handful of interior probe points spread across the frame.
    probes: list[tuple[float, float]] = [
        (500.0, 360.0),
        (250.0, 200.0),
        (1100.0, 600.0),
        (75.0, 420.0),
        (980.0, 110.0),
    ]

    for pixel in probes:
        court = homography.warp_point(pixel)
        recovered = homography.inverse_warp(court)
        # ε = 0.5 px per spec.
        assert recovered == pytest.approx(pixel, abs=0.5)


def test_save_and_load_cache(tmp_path: Path) -> None:
    """Save keypoints to JSON, reload them, expect identical Pydantic model."""

    cache_path = tmp_path / "court_keypoints_cam1.json"

    CourtDetector.save_to_cache(_SKEWED_KEYPOINTS, cache_path)

    # Sanity check: the file actually exists and contains valid JSON with the
    # expected canonical keys (Pydantic .model_dump output).
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"top_left", "top_right", "bottom_right", "bottom_left"}

    loaded = CourtDetector.load_from_cache(cache_path)
    assert loaded == _SKEWED_KEYPOINTS

    # And the matrix derived from a fresh load must match the original.
    original_matrix = Homography.from_keypoints(_SKEWED_KEYPOINTS).matrix
    reloaded_matrix = Homography.from_keypoints(loaded).matrix
    np.testing.assert_allclose(reloaded_matrix, original_matrix, atol=1e-10)


def test_detect_keypoints_non_interactive_returns_none() -> None:
    """Without interactive=True and no auto-detector, MVP returns None."""

    detector = CourtDetector()
    random_frame = np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)

    assert detector.detect_keypoints(random_frame, interactive=False) is None


# ---------------------------------------------------------------------------
# Edge case tests (F6 audit)
# ---------------------------------------------------------------------------


def test_homography_singular_matrix() -> None:
    """Degenerate keypoints (all collinear) must surface a clear error.

    When all 4 ``CourtKeypoints`` lie on the same line the perspective
    transform is singular: ``cv2.getPerspectiveTransform`` either
    returns a non-invertible matrix or raises. Either way the user
    deserves a recognizable exception (``ValueError`` or ``cv2.error``
    wrapped via ``np.linalg.LinAlgError`` for the inverse) — never a
    silent garbage transform — and the error message must make the
    degeneracy obvious.
    """

    # Four points on a single horizontal line — no perspective info.
    collinear = CourtKeypoints(
        top_left=(0.0, 100.0),
        top_right=(500.0, 100.0),
        bottom_right=(1000.0, 100.0),
        bottom_left=(1500.0, 100.0),
    )

    with pytest.raises((np.linalg.LinAlgError, ValueError, RuntimeError)) as excinfo:
        Homography.from_keypoints(collinear)

    # Whatever flavor of exception the underlying stack throws, the
    # message must hint at the singular / degenerate / non-invertible
    # cause so the user can map it back to the keypoint geometry.
    msg = str(excinfo.value).lower()
    assert any(
        marker in msg
        for marker in ("singular", "degenerate", "non-invertible", "linalg", "transform")
    ), f"unexpected error message: {msg!r}"
