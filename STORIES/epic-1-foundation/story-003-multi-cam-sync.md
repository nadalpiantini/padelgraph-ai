# Story 003 — Multi-camera sync (2 cameras)

**Epic:** 1 (Foundation, MVP semana 1)
**Status:** pending
**Estimate:** ~6h
**Target completion:** día 3–4 (2026-05-30 / 2026-05-31)
**Blocked by:** story-002 (Detector class shape stable)

---

## Goal

Given 2 video files (each from a phone at a different angle of the same padel match) plus a known time offset between them, produce a synchronized iterator yielding `SyncedFrameBatch` per timestamp — each batch contains the corresponding frame from each camera.

## WHAT

1. **Create `src/padelgraph_ai/sync/__init__.py`** exporting `MultiCamSync`
2. **Create `src/padelgraph_ai/sync/timestamp_sync.py`** with:
   - `class MultiCamSync` with `__init__(video_paths: list[Path], offsets_seconds: list[float])`
   - `align() -> Iterator[SyncedFrameBatch]` yielding `SyncedFrameBatch(ts: float, frames: dict[cam_id → np.ndarray])`
   - Reads videos lazily (do not load all frames into memory)
   - Handles different FPS by interpolating to a common FPS (target: lowest FPS of inputs, or 30 if all match)
   - When one video ends before others, emits remaining batches with `None` for the ended camera until all end
3. **Add `SyncedFrameBatch` to `schemas.py`**
4. **Create `tests/test_sync.py`** with:
   - Test 1: 2 synthetic videos (10 seconds, same FPS, no offset) → 300 batches at 30fps
   - Test 2: 2 synthetic videos (10 seconds, 0.5s offset) → batches start aligned
   - Test 3: Real Alan-provided pair → end-to-end smoke test (iterates without crash)

## WHERE

- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/sync/__init__.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/sync/timestamp_sync.py`
- Modify: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/schemas.py` (add `SyncedFrameBatch`)
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/tests/test_sync.py`
- Test videos: `data/inputs/test_cam1.mp4` + `data/inputs/test_cam2.mp4` (Alan-provided, gitignored)

## VERIFY

```bash
cd ~/Dev/padelgraph-ai
uv run pytest tests/test_sync.py -v

# Sanity: walk 50 batches of real video pair
uv run python -c "
from padelgraph_ai.sync import MultiCamSync
sync = MultiCamSync(['data/inputs/test_cam1.mp4', 'data/inputs/test_cam2.mp4'], offsets_seconds=[0.0, 0.3])
for i, batch in enumerate(sync.align()):
    if i >= 50: break
    cam_count = sum(1 for f in batch.frames.values() if f is not None)
    print(f'Batch {i} t={batch.ts:.2f}s — {cam_count}/2 cams')
"
```

## OFF_LIMITS

- NO 4-camera support (Epic 2)
- NO automatic offset detection from audio (Epic 2 — assume user provides offset manually for MVP)
- NO realtime streaming sync (MVP is post-process only)
- NO calibration in this story (story-004 handles court homography; story-005 handles triangulation)

## Acceptance criteria

- `MultiCamSync` aligns 2 synthetic videos with known offset within ±1 frame
- Memory usage stays bounded (lazy iteration; should handle 30-minute videos without OOM on a 16GB Mac)
- `tests/test_sync.py` passes
- Code passes `ruff check`

## Dependencies

- opencv-python (already pinned)
- numpy (already pinned)
- 2 real Alan-provided test videos with known offset

## Notes

Frame interpolation when FPS differs: use OpenCV `cv2.VideoCapture.set(cv2.CAP_PROP_POS_MSEC, ts*1000)` for each camera at each target timestamp. This is simpler than implementing custom interpolation and accurate enough for MVP.

For audio-based auto-sync (Epic 2), the typical approach is cross-correlation of audio waveforms. Out of scope here.
