# Story 004 — Court detection + homography 2D

**Epic:** 1 (Foundation, MVP semana 1)
**Status:** pending
**Estimate:** ~4h
**Target completion:** día 3 (2026-05-30)

---

## Goal

Given a single padel match frame, detect the 4 court corners (manual seed for MVP — see Epic 2 for auto), then compute a homography matrix that maps any pixel coordinate to court-coordinate space (court is a known rectangle: 20m × 10m for doubles padel).

## WHAT

1. **Create `src/padelgraph_ai/court/__init__.py`** exporting `CourtDetector`, `Homography`
2. **Create `src/padelgraph_ai/court/homography.py`** with:
   - `class CourtKeypoints` (Pydantic) — 4 corners in pixel coords: `top_left, top_right, bottom_right, bottom_left`
   - `class CourtDetector`:
     - `detect_keypoints(frame: np.ndarray) -> CourtKeypoints | None` — MVP uses manual seed via interactive click picker (`cv2.setMouseCallback`) the first time, then caches to a sidecar JSON
     - `load_from_cache(cache_path: Path) -> CourtKeypoints` — load previously-picked keypoints
   - `class Homography`:
     - `from_keypoints(keypoints: CourtKeypoints, court_dim_meters: tuple = (20.0, 10.0)) -> Homography`
     - `warp_point(pixel_xy: tuple[float, float]) -> tuple[float, float]` — returns court coords in meters
     - `inverse_warp(court_xy: tuple[float, float]) -> tuple[float, float]` — court → pixel
3. **Add `CourtKeypoints` to `schemas.py`** (or keep in court module — small Pydantic class)
4. **Create `tests/test_homography.py`** with:
   - Test 1: round-trip — `warp` then `inverse_warp` should return original pixel within ε
   - Test 2: known court corners → warped baseline should equal `(0, 0)`, far corner should equal `(20, 10)`
   - Test 3: load real Alan-provided test frame → manual pick 4 corners → save cache JSON → load it → verify match

## WHERE

- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/court/__init__.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/court/homography.py`
- Modify (optional): `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/schemas.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/tests/test_homography.py`
- Test artifact: `/Users/nadalpiantini/Dev/padelgraph-ai/data/inputs/test_court_keypoints_cam1.json` (after manual pick)

## VERIFY

```bash
cd ~/Dev/padelgraph-ai
uv run pytest tests/test_homography.py -v

# Sanity: pick keypoints interactively for a real frame
uv run python -c "
from padelgraph_ai.court import CourtDetector, Homography
import cv2
detector = CourtDetector()
frame = cv2.imread('data/inputs/test_court_frame.jpg')  # Alan-provided
kp = detector.detect_keypoints(frame)  # opens window, user clicks 4 corners
kp.save('data/inputs/test_court_keypoints_cam1.json')
h = Homography.from_keypoints(kp)
print('Service line at pixel ~middle should map to court (10m, 5m) ish')
print(h.warp_point(((frame.shape[1] // 2, frame.shape[0] // 2))))
"
```

## OFF_LIMITS

- NO auto-detection of court corners (Epic 2 — explicit non-goal for MVP)
- NO 3D court mesh / single-court vs glass-court differentiation (MVP assumes flat ground)
- NO multiple-court types (assume standard doubles padel: 20m × 10m)
- NO per-camera handling of tilted vs eye-level shots (will need refinement in Epic 2)

## Acceptance criteria

- `Homography` round-trip preserves pixel coords within ε=1 pixel
- Manual keypoint picker works (click 4 corners → JSON file written)
- Cached keypoints load and reproduce the same homography matrix
- `tests/test_homography.py` passes
- Code passes `ruff check`

## Dependencies

- opencv-python (already pinned)
- numpy (already pinned)
- One real Alan-provided test frame (extracted from one of the test videos via OpenCV or ffmpeg)

## Notes

For MVP, court keypoints are picked once per camera position and cached. This is acceptable because Alan's setup with phone mounts on the glass is static — court doesn't move relative to the cameras.

For Epic 2, the auto-court-detection story will likely use a custom-trained keypoint regression model. Reference repos to study at that time: `gchlebus/tennis-court-detection` (65★, manual but with feature extraction tricks) and PadelVic dataset for training data.
