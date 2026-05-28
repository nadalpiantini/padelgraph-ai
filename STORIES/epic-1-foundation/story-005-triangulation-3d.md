# Story 005 — Kornia 3D triangulation (ball position)

**Epic:** 1 (Foundation, MVP semana 1) — **stretch goal, may slip to mes 2 if triangulation proves harder than expected**
**Status:** pending
**Estimate:** ~8h
**Target completion:** día 4–5 (2026-05-31 / 2026-06-01)
**Blocked by:** story-002 (ball detection per camera), story-003 (synced batches), story-004 (homography per camera)

---

## Goal

Given the 2D ball position detected in 2 synchronized camera frames, plus camera intrinsics and extrinsics, use Kornia to triangulate the ball's 3D position in world coordinates (court frame of reference).

## WHAT

1. **Create `src/padelgraph_ai/fusion/__init__.py`** exporting `Triangulator`
2. **Create `src/padelgraph_ai/fusion/kornia_triangulate.py`** with:
   - `class Calibration` (Pydantic): per-camera `intrinsics` (3×3 matrix), `extrinsics` (4×4 matrix), `distortion_coeffs` (optional)
   - `class Triangulator`:
     - `__init__(calibrations: dict[str, Calibration])`
     - `triangulate(points_2d_per_cam: dict[str, tuple[float, float]]) -> Point3D | None`
     - Internally calls `kornia.geometry.epipolar.triangulate_points` with projection matrices
     - Returns `None` if fewer than 2 cameras have a valid point
3. **Create a simple calibration helper script** `src/padelgraph_ai/fusion/calibrate.py`:
   - `compute_extrinsics_from_court_keypoints(camera_keypoints_2d: CourtKeypoints, court_dim_meters=(20, 10), z=0) -> 4×4 matrix`
   - Lets us derive extrinsics from the same manual court keypoint pick that homography uses — no separate camera calibration step in MVP
4. **Add `Calibration`, `Point3D`, `MatchAnalysis` to `schemas.py`**
5. **Create `tests/test_triangulate.py`** with:
   - Test 1: synthetic — 2 cameras with known extrinsics, a known 3D point → projects to 2D → triangulate back → ≤ε from original
   - Test 2: real Alan-provided 1-second pair with known ball position (Alan annotates frame manually) → triangulated 3D should be within reasonable court bounds
6. **CLI entry point**: `src/padelgraph_ai/pipeline/infer.py` (Click-based) that orchestrates story 002 + 003 + 004 + 005 end-to-end and writes the final JSON

## WHERE

- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/fusion/__init__.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/fusion/kornia_triangulate.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/fusion/calibrate.py`
- Modify: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/schemas.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/pipeline/__init__.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/pipeline/infer.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/tests/test_triangulate.py`
- Test calibration JSON: `data/inputs/calib_2cam.json`

## VERIFY

```bash
cd ~/Dev/padelgraph-ai
uv run pytest tests/test_triangulate.py -v

# End-to-end: run the full pipeline CLI
uv run padelgraph-ai-infer \
  --video data/inputs/test_cam1.mp4 \
  --video data/inputs/test_cam2.mp4 \
  --calib data/inputs/calib_2cam.json \
  --out data/outputs/run-001.json

# Verify JSON shape per ARCHITECTURE.md
jq '.frames | length' data/outputs/run-001.json    # > 0
jq '.frames[10].ball.position_3d' data/outputs/run-001.json   # [x, y, z]
jq '.frames[10].players_per_cam.cam1' data/outputs/run-001.json   # [...]
```

## OFF_LIMITS

- NO 4-camera triangulation (MVP is 2 cameras only)
- NO bundle adjustment (Epic 2 — assume calibration is good enough for MVP)
- NO sub-pixel refinement of ball center (Epic 2)
- NO temporal smoothing of 3D trajectory (Epic 2 — likely Kalman filter)
- NO inventing "accuracy %" numbers (D26 — report what tests actually show)
- NO modifying upstream Kornia / OpenCV code

## Acceptance criteria

- Synthetic test passes (round-trip 3D → 2D × 2 → 3D within ε=0.05m for points on a 20m × 10m × 3m court)
- Real-video test produces 3D ball positions that, when manually inspected, are plausible (inside the court bounds, height between 0 and ~5m)
- End-to-end CLI run produces a JSON file matching the schema in ARCHITECTURE.md
- `tests/test_triangulate.py` passes
- Code passes `ruff check`
- README updated with a working command example using actual file paths

## Dependencies

- kornia (already pinned)
- torch (already pinned)
- All prior stories' modules
- Calibration JSON (derived from court keypoint manual pick + camera intrinsics estimate — see notes)

## Notes

This is the most uncertain story of MVP semana 1. If 3D triangulation proves too finicky (calibration sensitivity, kornia API gotchas), the **fallback is to keep ball positions only as 2D-per-camera + homography-warped 2D-court-coords**. This still produces a usable JSON output, just without the 3D height information. Document the fallback decision in `.decision-log.md` if exercised.

Camera intrinsics estimate for MVP: assume a typical iPhone main rear camera (focal length ~28mm equivalent → in pixels ~1200 for a 1080p frame). Refinement is Epic 2 (proper checkerboard calibration).

This story closes Epic 1. After this, the MVP semana 1 acceptance criteria from PRD.md should all pass.
