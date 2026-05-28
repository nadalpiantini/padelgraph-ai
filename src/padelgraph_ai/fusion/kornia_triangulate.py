"""Kornia-based 3D triangulation across multiple synchronized cameras.

Story-005 — Foundation Epic.

Given a 2D observation of the same physical point in 2+ synchronized
camera frames, recover its 3D position in world coordinates using the DLT
formulation provided by :func:`kornia.geometry.epipolar.triangulate_points`.

Design notes
------------
- **2-camera MVP.** When more than 2 cameras provide a valid observation,
  we triangulate every pair and average the resulting 3D points. This is
  intentionally simpler than a single N-view DLT solve; bundle adjustment
  is Epic 2 scope per ``OFF_LIMITS`` in the story spec.
- **None when underconstrained.** Triangulation requires at least 2
  cameras with a valid 2D point — anything less returns ``None`` so
  callers can keep iterating without crashing.
- **Schema contract.** :class:`Calibration` lives in
  :mod:`padelgraph_ai.schemas` (unchanged by this story). The returned
  :class:`Point3D` mirrors that schema. Coordinates are in whatever world
  frame the calibration extrinsics were computed in — for MVP that's the
  court frame (origin at top-left corner, +X toward top-right, +Y toward
  bottom-left, +Z up) per :mod:`padelgraph_ai.fusion.calibrate`.

Apache 2.0.
"""

from __future__ import annotations

import numpy as np
import torch
from kornia.geometry.epipolar import triangulate_points

from padelgraph_ai.schemas import Calibration, Point3D

__all__ = ["Triangulator"]


class Triangulator:
    """Multi-camera 3D triangulator backed by Kornia's DLT solver.

    Parameters
    ----------
    calibrations
        Mapping from ``cam_id`` to its :class:`Calibration`. Cameras
        referenced in :meth:`triangulate` that are absent from this
        mapping are silently skipped — the typical case is a camera
        that detected the ball in a frame but was never calibrated.
    """

    def __init__(self, calibrations: dict[str, Calibration]) -> None:
        if not calibrations:
            raise ValueError("Triangulator requires at least one Calibration")
        self._calibrations: dict[str, Calibration] = dict(calibrations)
        # Pre-compute 3x4 projection matrices P = K @ [R|t] once per
        # camera so repeat triangulations don't re-do the matmul.
        self._projections: dict[str, np.ndarray] = {
            cam_id: self._projection_matrix(calib)
            for cam_id, calib in self._calibrations.items()
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def triangulate(
        self,
        points_2d_per_cam: dict[str, tuple[float, float]],
    ) -> Point3D | None:
        """Triangulate a single 3D point from 2+ camera observations.

        Parameters
        ----------
        points_2d_per_cam
            Mapping ``cam_id -> (x, y)`` of the 2D pixel coordinate of the
            same physical point seen in each camera frame.

        Returns
        -------
        A :class:`Point3D` in world coordinates (meters), or ``None`` if
        fewer than 2 known cameras supplied a point.
        """
        # Filter to cameras we actually have calibration for. Anything
        # else is silently dropped — see class docstring.
        usable: list[tuple[str, tuple[float, float]]] = [
            (cam_id, xy)
            for cam_id, xy in points_2d_per_cam.items()
            if cam_id in self._projections
        ]
        if len(usable) < 2:
            return None

        # For MVP: triangulate every pair, average. This degrades
        # gracefully to the single-pair case when only 2 cams are
        # available and avoids an N-view solver this story explicitly
        # excludes.
        results: list[np.ndarray] = []
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                cam_a, point_a = usable[i]
                cam_b, point_b = usable[j]
                results.append(
                    self._triangulate_pair(
                        self._projections[cam_a],
                        self._projections[cam_b],
                        point_a,
                        point_b,
                    )
                )

        if not results:  # defensive — len(usable) >= 2 guarantees this never trips
            return None

        averaged = np.mean(np.stack(results, axis=0), axis=0)
        return Point3D(
            x=float(averaged[0]),
            y=float(averaged[1]),
            z=float(averaged[2]),
            confidence=1.0,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    @staticmethod
    def _projection_matrix(calib: Calibration) -> np.ndarray:
        """Return the 3x4 projection matrix ``K @ [R|t]`` for ``calib``."""
        k = calib.K()
        if k.shape != (3, 3):
            raise ValueError(
                f"Calibration {calib.cam_id} has K shape {k.shape}, expected (3, 3)"
            )
        rt = calib.Rt()
        if rt.shape == (4, 4):
            rt_3x4 = rt[:3, :4]
        elif rt.shape == (3, 4):
            rt_3x4 = rt
        else:
            raise ValueError(
                f"Calibration {calib.cam_id} has extrinsics shape {rt.shape}, "
                "expected (4, 4) or (3, 4)"
            )
        return k @ rt_3x4

    @staticmethod
    def _triangulate_pair(
        projection_a: np.ndarray,
        projection_b: np.ndarray,
        point_a: tuple[float, float],
        point_b: tuple[float, float],
    ) -> np.ndarray:
        """Run Kornia's DLT on a single (cam_a, cam_b) pair, return ``[x, y, z]``."""
        # Kornia expects float32 tensors with batched shapes:
        #   P: (B, 3, 4),  points: (B, N, 2). Use B=1, N=1.
        p1 = torch.from_numpy(projection_a.astype(np.float32)).unsqueeze(0)
        p2 = torch.from_numpy(projection_b.astype(np.float32)).unsqueeze(0)
        pts1 = torch.tensor([[list(point_a)]], dtype=torch.float32)
        pts2 = torch.tensor([[list(point_b)]], dtype=torch.float32)

        # Use the SVD solver: this is a single-point call, so the speed
        # penalty is negligible and the extra numerical stability is worth
        # the ~10x cost when N=1.
        point_3d = triangulate_points(p1, p2, pts1, pts2, solver="svd")
        return point_3d.detach().cpu().numpy().reshape(3)
