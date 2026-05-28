# Story 002 — Single-cam YOLOX detection (ball + player bbox)

**Epic:** 1 (Foundation, MVP semana 1)
**Status:** pending
**Estimate:** ~6h
**Target completion:** día 2 (2026-05-29)

---

## Goal

Wrap YOLOX inference in `src/padelgraph_ai/detection/yolox_runner.py` such that passing a single video file produces per-frame ball and player bounding boxes. Smoke test with one real Alan-provided padel video.

## WHAT

1. **Install YOLOX** as an extra dependency (instructions in README — `uv pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git`)
2. **Download or symlink** a YOLOX pre-trained checkpoint (YOLOX-s on COCO works as starting point; better fine-tuning happens in Epic 2)
3. **Create `src/padelgraph_ai/detection/__init__.py`** exporting the `Detector` class
4. **Create `src/padelgraph_ai/detection/yolox_runner.py`** with:
   - `class Detector` with `__init__(checkpoint_path: str, device: str = "auto")` (auto picks MPS on Apple Silicon, CPU fallback)
   - `infer(frame: np.ndarray) -> list[Detection]` returning bbox + class + confidence
   - `infer_video(video_path: str | Path) -> Iterator[FrameDetections]` yielding per-frame results
   - Filter classes to relevant: `sports ball` (32) + `person` (0) from COCO
5. **Create `src/padelgraph_ai/schemas.py`** with Pydantic models: `Detection`, `FrameDetections`
6. **Create `tests/test_detection.py`** — smoke test with a 30-frame test clip (Alan provides) verifying:
   - `Detector` instantiates without error
   - Returns non-empty detections per frame for >50% of frames
   - All detections have valid bbox coordinates within frame bounds

## WHERE

- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/detection/__init__.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/detection/yolox_runner.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/src/padelgraph_ai/schemas.py`
- New file: `/Users/nadalpiantini/Dev/padelgraph-ai/tests/test_detection.py`
- Test video: `/Users/nadalpiantini/Dev/padelgraph-ai/data/inputs/test_single_cam.mp4` (Alan-provided, gitignored)

## VERIFY

```bash
cd ~/Dev/padelgraph-ai
uv sync --extra dev
uv pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git

# Acceptance: smoke test passes
uv run pytest tests/test_detection.py -v

# Sanity: run on real video and inspect a few detections
uv run python -c "
from padelgraph_ai.detection import Detector
det = Detector(checkpoint_path='/path/to/yolox_s.pth')
for i, frame_dets in enumerate(det.infer_video('data/inputs/test_single_cam.mp4')):
    if i >= 3: break
    print(f'Frame {i}: {len(frame_dets.detections)} detections')
"
```

## OFF_LIMITS

- NO multi-camera support in this story (story-003 handles sync)
- NO court detection in this story (story-004)
- NO pose tracking (mes 2)
- NO fine-tuning YOLOX checkpoint (Epic 2)
- NO inventing accuracy numbers (D26 Verificar-antes-de-Afirmar — report what the test actually produces)

## Acceptance criteria

- `Detector` class loads YOLOX checkpoint without errors on Apple Silicon (MPS)
- `infer(frame)` returns within ≤200ms per frame at 480p
- Smoke test detects players in ≥50% of test frames (low bar; tighter accuracy is Epic 2)
- Smoke test detects ball in ≥30% of test frames (ball is harder; tighter is Epic 2)
- `tests/test_detection.py` passes
- Code passes `ruff check`

## Dependencies

- YOLOX checkpoint file (download separately, gitignored)
- One real Alan-provided test video at `data/inputs/test_single_cam.mp4`

## Notes

This story is the riskiest of week 1 because YOLOX install on Apple Silicon may have quirks (torch + CUDA assumptions in YOLOX repo). If install fails, fallback is `detectron2` with similar wrapper interface — but `Detector` class signature stays the same.
