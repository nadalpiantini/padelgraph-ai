"""3D fusion across multiple cameras (story-005).

Public API:

- :class:`Triangulator` — triangulate a single 3D point from 2D observations
  in 2+ synchronized cameras using Kornia's DLT solver.
- :func:`compute_extrinsics_from_court_keypoints` — derive a per-camera
  ``[R|t]`` extrinsic matrix from the same 4 court keypoints used by
  :mod:`padelgraph_ai.court.homography` plus the camera intrinsics.

Apache 2.0.
"""

from padelgraph_ai.fusion.calibrate import compute_extrinsics_from_court_keypoints
from padelgraph_ai.fusion.kornia_triangulate import Triangulator

__all__ = ["Triangulator", "compute_extrinsics_from_court_keypoints"]
