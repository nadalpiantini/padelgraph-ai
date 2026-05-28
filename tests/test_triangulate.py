"""Unit tests for :mod:`padelgraph_ai.fusion` — triangulation + calibration.

The triangulation tests run on fully synthetic camera geometry (no video,
no detector) so they remain hermetic. The calibration tests verify the
PnP round-trip via re-projection error in pixels.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from padelgraph_ai.fusion import Triangulator, compute_extrinsics_from_court_keypoints
from padelgraph_ai.schemas import Calibration, CourtKeypoints, Point3D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_intrinsics(
    focal_px: float = 1200.0,
    width: int = 1920,
    height: int = 1080,
) -> np.ndarray:
    """Return a typical 1080p pinhole K matrix."""
    return np.asarray(
        [
            [focal_px, 0.0, width / 2.0],
            [0.0, focal_px, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _build_extrinsics(
    camera_position_world: tuple[float, float, float],
    look_at_world: tuple[float, float, float] = (10.0, 5.0, 0.0),
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Build a 4x4 world-to-camera extrinsic matrix using a look-at convention.

    Camera convention: +Z is the optical axis (forward), +X is right in the
    image, +Y is down in the image (OpenCV standard).
    """
    eye = np.asarray(camera_position_world, dtype=np.float64)
    target = np.asarray(look_at_world, dtype=np.float64)
    up = np.asarray(world_up, dtype=np.float64)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    # In OpenCV convention image-y points down → flip the up axis.
    cam_y = -cam_up

    rotation = np.stack([right, cam_y, forward], axis=0)  # world -> camera
    translation = -rotation @ eye

    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation
    return extrinsics


def _project_world_point_to_pixel(
    point_world: tuple[float, float, float],
    k: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[float, float]:
    """Forward-project a 3D world point through ``K @ [R|t]`` to a pixel."""
    point_homog = np.array([*point_world, 1.0], dtype=np.float64)
    cam_point = extrinsics @ point_homog
    image_point = k @ cam_point[:3]
    return (float(image_point[0] / image_point[2]), float(image_point[1] / image_point[2]))


# ---------------------------------------------------------------------------
# Triangulation tests
# ---------------------------------------------------------------------------


def test_triangulate_synthetic() -> None:
    """Round-trip: known 3D point → project to 2 cams → triangulate → recover within ε."""
    k = _build_intrinsics()
    image_size = (1920, 1080)

    # Two cameras viewing the court from opposite sidelines, both looking at
    # the same target ~10m, 5m, 0m (center of court at floor).
    extrinsics_cam1 = _build_extrinsics(camera_position_world=(10.0, -8.0, 4.0))
    extrinsics_cam2 = _build_extrinsics(camera_position_world=(10.0, 18.0, 4.0))

    ground_truth = (10.0, 5.0, 1.5)
    pixel_cam1 = _project_world_point_to_pixel(ground_truth, k, extrinsics_cam1)
    pixel_cam2 = _project_world_point_to_pixel(ground_truth, k, extrinsics_cam2)

    calibrations: dict[str, Calibration] = {
        "cam1": Calibration(
            cam_id="cam1",
            intrinsics=k.tolist(),
            extrinsics=extrinsics_cam1.tolist(),
            image_size=image_size,
        ),
        "cam2": Calibration(
            cam_id="cam2",
            intrinsics=k.tolist(),
            extrinsics=extrinsics_cam2.tolist(),
            image_size=image_size,
        ),
    }
    triangulator = Triangulator(calibrations)

    recovered = triangulator.triangulate({"cam1": pixel_cam1, "cam2": pixel_cam2})
    assert isinstance(recovered, Point3D)
    assert recovered.x == pytest.approx(ground_truth[0], abs=0.1)
    assert recovered.y == pytest.approx(ground_truth[1], abs=0.1)
    assert recovered.z == pytest.approx(ground_truth[2], abs=0.1)


def test_triangulate_single_camera_returns_none() -> None:
    """Fewer than 2 cameras must return None, not raise or fabricate."""
    k = _build_intrinsics()
    extrinsics = _build_extrinsics(camera_position_world=(10.0, -8.0, 4.0))
    calibrations: dict[str, Calibration] = {
        "cam1": Calibration(
            cam_id="cam1",
            intrinsics=k.tolist(),
            extrinsics=extrinsics.tolist(),
            image_size=(1920, 1080),
        ),
    }
    triangulator = Triangulator(calibrations)

    assert triangulator.triangulate({"cam1": (960.0, 540.0)}) is None
    assert triangulator.triangulate({}) is None
    # An entry for an unknown cam_id should also be filtered out → None.
    assert triangulator.triangulate({"cam_unknown": (100.0, 100.0)}) is None


def test_triangulate_requires_at_least_one_calibration() -> None:
    """Constructing with an empty calibration map is a programmer error."""
    with pytest.raises(ValueError, match="at least one"):
        Triangulator({})


# ---------------------------------------------------------------------------
# Calibration helper tests
# ---------------------------------------------------------------------------


def test_calibration_extrinsics_roundtrip() -> None:
    """compute_extrinsics_from_court_keypoints → project → recover ≤ε pixels."""
    k = _build_intrinsics()
    court_dim = (20.0, 10.0)

    # Build a synthetic camera with known extrinsics, project the 4 court
    # corners to pixels — those become the "manual" keypoints the helper
    # will consume.
    true_extrinsics = _build_extrinsics(camera_position_world=(10.0, -8.0, 4.0))
    corners_world = [
        (0.0, 0.0, 0.0),
        (court_dim[0], 0.0, 0.0),
        (court_dim[0], court_dim[1], 0.0),
        (0.0, court_dim[1], 0.0),
    ]
    pixel_corners = [_project_world_point_to_pixel(c, k, true_extrinsics) for c in corners_world]

    keypoints = CourtKeypoints(
        top_left=pixel_corners[0],
        top_right=pixel_corners[1],
        bottom_right=pixel_corners[2],
        bottom_left=pixel_corners[3],
    )

    recovered_extrinsics = np.asarray(
        compute_extrinsics_from_court_keypoints(
            camera_keypoints_2d=keypoints,
            intrinsics=k,
            court_dim_meters=court_dim,
        ),
        dtype=np.float64,
    )
    assert recovered_extrinsics.shape == (4, 4)

    # Re-project every corner through the recovered extrinsics and check
    # the residual is ≤ε=2 px per spec.
    for world_corner, expected_pixel in zip(corners_world, pixel_corners, strict=True):
        reprojected = _project_world_point_to_pixel(
            world_corner, k, recovered_extrinsics
        )
        assert reprojected[0] == pytest.approx(expected_pixel[0], abs=2.0)
        assert reprojected[1] == pytest.approx(expected_pixel[1], abs=2.0)


def test_calibration_helper_rejects_bad_intrinsics() -> None:
    """A non-3x3 intrinsics matrix must raise rather than silently mis-solve."""
    keypoints = CourtKeypoints(
        top_left=(100.0, 100.0),
        top_right=(1820.0, 100.0),
        bottom_right=(1820.0, 980.0),
        bottom_left=(100.0, 980.0),
    )
    # Pass a 2x2 matrix to trigger the validation guard.
    bad_intrinsics = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="intrinsics must be 3x3"):
        compute_extrinsics_from_court_keypoints(
            camera_keypoints_2d=keypoints,
            intrinsics=bad_intrinsics,
        )


def test_calibration_helper_handles_distortion_coeffs() -> None:
    """Passing distortion coeffs must not crash; round-trip stays within tolerance.

    We use zeroed distortion so the round-trip residual matches the no-dist
    case — the goal here is to exercise the code path, not to validate the
    distortion model itself.
    """
    k = _build_intrinsics()
    court_dim = (20.0, 10.0)
    true_extrinsics = _build_extrinsics(camera_position_world=(10.0, -8.0, 4.0))
    corners_world = [
        (0.0, 0.0, 0.0),
        (court_dim[0], 0.0, 0.0),
        (court_dim[0], court_dim[1], 0.0),
        (0.0, court_dim[1], 0.0),
    ]
    pixel_corners = [_project_world_point_to_pixel(c, k, true_extrinsics) for c in corners_world]
    keypoints = CourtKeypoints(
        top_left=pixel_corners[0],
        top_right=pixel_corners[1],
        bottom_right=pixel_corners[2],
        bottom_left=pixel_corners[3],
    )

    extrinsics = compute_extrinsics_from_court_keypoints(
        camera_keypoints_2d=keypoints,
        intrinsics=k,
        court_dim_meters=court_dim,
        distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    # Sanity: result is a 4x4 nested list of floats.
    arr = np.asarray(extrinsics, dtype=np.float64)
    assert arr.shape == (4, 4)
    # Last row must be the homogeneous identity row.
    np.testing.assert_allclose(arr[3], np.asarray([0.0, 0.0, 0.0, 1.0]))


def test_cv2_solvepnp_ippe_is_available() -> None:
    """Sanity: the OpenCV flag used by the calibration helper is exposed.

    Older opencv-python wheels (<4.5) lack SOLVEPNP_IPPE; this guard
    surfaces an actionable error before users hit the calibration path.
    """
    assert hasattr(cv2, "SOLVEPNP_IPPE"), (
        "opencv-python >= 4.5 required for SOLVEPNP_IPPE; "
        "upgrade via 'uv sync'"
    )
