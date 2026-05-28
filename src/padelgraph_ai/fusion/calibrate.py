"""Derive camera extrinsics from manually-picked court keypoints.

Story-005 — MVP calibration helper.

The MVP avoids a separate camera-calibration step (checkerboard pattern,
etc.) by reusing the same 4 court-corner pixel picks that
:mod:`padelgraph_ai.court.homography` already requires for the 2D
warp. We pair those 2D picks with the known 3D positions of the court
corners (z=0 by definition) and solve the Perspective-n-Point (PnP)
problem via :func:`cv2.solvePnP` to recover the rotation + translation
of each camera in the court world frame.

Court frame convention (matches :class:`padelgraph_ai.court.homography.Homography`):

- Origin at ``top_left`` corner
- +X axis runs toward ``top_right`` (long side, default 20 m)
- +Y axis runs toward ``bottom_left`` (short side, default 10 m)
- +Z axis points up out of the court floor

Returns a 4x4 homogeneous ``[R | t; 0 0 0 1]`` matrix as a nested Python
list so it can be dropped straight into the
:class:`padelgraph_ai.schemas.Calibration` ``extrinsics`` field.

Apache 2.0.
"""

from __future__ import annotations

import cv2
import numpy as np

from padelgraph_ai.court.homography import DEFAULT_COURT_DIM_METERS
from padelgraph_ai.schemas import CourtKeypoints

__all__ = ["compute_extrinsics_from_court_keypoints"]


def compute_extrinsics_from_court_keypoints(
    camera_keypoints_2d: CourtKeypoints,
    intrinsics: np.ndarray | list[list[float]],
    court_dim_meters: tuple[float, float] = DEFAULT_COURT_DIM_METERS,
    assumed_z: float = 0.0,
    distortion_coeffs: np.ndarray | list[float] | None = None,
) -> list[list[float]]:
    """Compute a 4x4 ``[R|t]`` extrinsic matrix from 4 court keypoints + intrinsics.

    Parameters
    ----------
    camera_keypoints_2d
        Pixel coordinates of the 4 court corners in canonical order
        (``top_left``, ``top_right``, ``bottom_right``, ``bottom_left``).
    intrinsics
        3x3 camera matrix ``K`` (fx, fy, cx, cy).
    court_dim_meters
        ``(width, height)`` of the court in meters. Defaults to a
        standard doubles padel court (20 m × 10 m).
    assumed_z
        Z coordinate (height) of the court corners. Defaults to 0 — the
        court floor is the world plane.
    distortion_coeffs
        Optional OpenCV-format distortion coefficients ``[k1, k2, p1,
        p2, k3]``. Pass ``None`` for an undistorted pinhole assumption.

    Returns
    -------
    A 4x4 homogeneous transform mapping world points (court frame) to
    camera coordinates, as a nested Python list ready to slot into
    :class:`padelgraph_ai.schemas.Calibration` ``extrinsics``.
    """
    width_m, height_m = court_dim_meters

    # 3D positions of the 4 court corners in the world (court) frame, in
    # the same canonical order as CourtKeypoints.
    object_points = np.asarray(
        [
            (0.0, 0.0, assumed_z),
            (width_m, 0.0, assumed_z),
            (width_m, height_m, assumed_z),
            (0.0, height_m, assumed_z),
        ],
        dtype=np.float64,
    )

    image_points = camera_keypoints_2d.as_array().astype(np.float64)

    k_matrix = np.asarray(intrinsics, dtype=np.float64)
    if k_matrix.shape != (3, 3):
        raise ValueError(f"intrinsics must be 3x3, got shape {k_matrix.shape}")

    if distortion_coeffs is None:
        dist = np.zeros((5, 1), dtype=np.float64)
    else:
        dist = np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1, 1)

    # SOLVEPNP_IPPE is the Infinitesimal Plane-based Pose Estimation
    # solver — purpose-built for planar targets like our flat court
    # floor, and (unlike SOLVEPNP_IPPE_SQUARE) imposes no winding-order
    # or symmetric-centering requirements on the 4 object points.
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        k_matrix,
        dist,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success:
        raise RuntimeError(
            "cv2.solvePnP failed to recover camera pose from court keypoints"
        )

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = rotation_matrix
    extrinsics[:3, 3] = tvec.reshape(3)
    return extrinsics.tolist()
