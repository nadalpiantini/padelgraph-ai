# Architecture — padelgraph-ai

**Status:** v0.1 (MVP semana 1)
**Date:** 2026-05-28
**Pattern:** BMAD-inspired structure, Sephirot-native workflow (no BMAD install)

---

## High-level diagram

```
  ┌──────────────────┐        ┌─────────────────────┐
  │  Video sources   │        │  Camera calibration │
  │  (1–4 phones)    │        │  (intrinsics +      │
  │  via LiveKit or  │        │   extrinsics JSON)  │
  │  filesystem      │        └──────────┬──────────┘
  └────────┬─────────┘                   │
           │                             │
           ▼                             ▼
  ┌────────────────────────────────────────────────┐
  │           padelgraph_ai.pipeline.infer         │
  │                  (CLI entry)                   │
  └────────┬───────────────────────────────────────┘
           │
           ├──▶ sync          (timestamp + offset; 2 cams MVP)
           ├──▶ detection     (YOLOX → ball + player bbox per cam)
           ├──▶ court         (OpenCV → 4 court corners + homography 2D)
           ├──▶ fusion        (Kornia → ball 3D triangulation)
           ├──▶ pose          (MediaPipe BlazePose — mes 2)
           │
           ▼
  ┌────────────────────────────────────────────────┐
  │   schemas.MatchAnalysis  (Pydantic JSON)       │
  │   { meta, frames[]: { ball, players, court } } │
  └────────┬───────────────────────────────────────┘
           │
           ├──▶ JSON file
           └──▶ Overlay video (OpenCV → mp4)
```

## Module boundaries (`src/padelgraph_ai/`)

Each module has a single responsibility and exports a clear interface. Modules communicate via Pydantic schemas (`schemas.py`), never via shared global state.

| Module | Public interface (planned) | Depends on |
|--------|---------------------------|------------|
| `detection/` | `Detector.infer(frame: np.ndarray) -> list[Detection]` | YOLOX, supervision, torch |
| `pose/` (mes 2) | `PoseEstimator.infer(frame, bbox) -> Skeleton` | mediapipe |
| `court/` | `CourtDetector.detect_keypoints(frame) -> CourtKeypoints` + `Homography.warp(point_2d) -> court_coord` | opencv-python |
| `sync/` | `MultiCamSync.align(video_paths: list[str]) -> Iterator[SyncedFrameBatch]` | opencv-python, numpy |
| `fusion/` | `Triangulator.triangulate(points_2d_per_cam: dict, calib: Calibration) -> Point3D` | kornia, torch |
| `pipeline/` | `infer.main()` (Click CLI) — orchestrates all modules | all of the above + click, pydantic, tqdm |
| `schemas.py` | Pydantic models: `Detection`, `Skeleton`, `CourtKeypoints`, `Point3D`, `MatchAnalysis`, `Calibration` | pydantic |

### Why isolation matters

- Each module testable independent (`tests/test_detection.py`, etc.)
- `pipeline/infer.py` is thin orchestration — easy to swap detectors, fusion algorithms, or output formats
- When `padelgraph-app` consumes this as a library (mes 9), it imports `padelgraph_ai.pipeline.infer` or individual modules — never reaches into internals
- Module boundaries make ZAI worker delegation possible: one story per module, one worker per story

## Data flow (MVP semana 1, 2 cameras)

```
1. Parse CLI args → load videos + calibration JSON
2. MultiCamSync.align(video_paths) yields SyncedFrameBatch{ ts, frames: {cam_id → np.ndarray} }
3. For each batch:
   a. For each camera frame:
      - Detector.infer(frame) → ball + player bboxes
      - (mes 2) PoseEstimator.infer(frame, player_bbox) → skeleton
   b. CourtDetector.detect_keypoints(frame_per_cam) → 4 corners per cam (cached if static)
   c. Triangulator.triangulate(ball_2d_per_cam, calib) → ball position_3d
4. Aggregate per-frame results → MatchAnalysis (Pydantic)
5. Write JSON to --out path
6. (Optional) Render overlay video → --overlay path
```

## Stack rationale (per `.decision-log.md`)

| Choice | Alternative considered | Why chosen |
|--------|------------------------|------------|
| YOLOX (Megvii) | YOLOv8 (Ultralytics) | YOLOv8 is AGPL-3.0 which contaminates any SaaS built on top. YOLOX is Apache 2.0 — verified via Context7 |
| Kornia | PyTorch3D, custom OpenCV | Kornia is purpose-built for differentiable computer vision; triangulation primitives + camera calibration helpers reduce custom code by ~80% (10K★ Apache, OpenCV authors) |
| MediaPipe BlazePose | OpenPose, RTMPose | BlazePose runs on Apple Silicon natively; OpenPose is stale (1.5+ years) and has NOASSERTION license; RTMPose requires MMPose framework overhead |
| supervision (Roboflow) | Hand-rolled annotation utilities | Saves 200+ lines of boilerplate for bbox tracking, video I/O, detection annotation |
| uv | poetry, pip-tools, hatch | uv is the 2026 standard; reproducible installs, faster than poetry, native pyproject.toml support |
| Python 3.12 (not 3.13) | Python 3.13 | MediaPipe compatibility verified via Context7; 3.13 not yet fully supported in the CV ecosystem |
| Apache 2.0 (repo license) | MIT, BSD-3, MPL-2.0 | Apache 2.0 explicitly handles patent grants; MIT is too minimal for a CV library where patent disputes are real risk |
| BMAD-hybrid (pattern only) | Full BMAD install, no scaffold | 95% of BMAD's 44 skills are planning-focused, irrelevant for CV pipeline. Apropia the discipline (PRD → Epics → Stories → decision-log) without overhead |

## Sephirot integration

This repo is **standalone** per D28 (Nunca Ligar Proyectos). It does NOT live inside `padelgraph-app/`, `sephirot-cc/`, or any other project.

- **`.sephirot/skills/`** (created mes 2–3 when patterns stabilize): 5 CV-specific Sephirot skills — `cv-data-prep`, `model-train`, `inference`, `metrics-gen`, `viz-dashboard`. NOT included in MVP semana 1 to avoid premature abstraction.
- **Workflow**: Opus (Sephirot main) handles architecture + PRD/EPICS/STORIES writing + code review + decision-log curation. Sonnet/ZAI workers handle implementation per story using caveman briefs (WHAT/WHERE/VERIFY/OFF_LIMITS).
- **Knowledge graph**: decisions captured in `.decision-log.md` and mirrored to MemPalace drawers (wing=sephirot, room=decisions) for cross-session continuity.

## Performance budget (MVP semana 1)

| Operation | Target | Hardware |
|-----------|--------|----------|
| Single-cam YOLOX inference | ≤200ms per frame (480p) | Apple Silicon MPS |
| Multi-cam sync alignment | ≤50ms per batch | CPU |
| OpenCV homography compute | ≤10ms per frame | CPU |
| Kornia 3D triangulation | ≤30ms per ball position | Apple Silicon MPS or CPU |
| End-to-end pipeline (2 cams, 1 minute @ 30fps) | ≤10 minutes wall clock | M2 Air |

Targets are best-effort, not SLOs. If exceeded, MVP still ships; optimization is mes 2+.

## Future expansion (mes 2+)

- **Auto court calibration** — replace manual seed with court-aware model (likely custom-trained on PadelVic + augmented synthetic data)
- **Pose tracking** — MediaPipe BlazePose at player bbox crops, exported as 13-DoF skeleton sequences
- **Shot classifier** — MMAction2 or TorchVision video models, trained on ≥500 labeled padel match clips
- **4-camera support** — extend `MultiCamSync` and `Triangulator` to 4 cameras; UI mockup for cross-angle review
- **Streaming mode** — RTSP/NDI ingest, near-real-time inference, hand-off to `padelgraph-app` LiveKit pipeline
- **Federated dataset** — opt-in upload of anonymized match clips from clubs → grows shot classifier accuracy
- **Federated model registry** — pre-trained checkpoints versioned and downloadable via `padelgraph-ai-download`

## Out of architectural scope (explicit)

- Real-time live AI overlay during play (post-process only)
- Refereeing / line-call decisions
- Broadcast video composition (Veo's domain)
- Hardware camera tower design (separate validation track)
- Mobile-app inference (server-side only in MVP; on-device inference may come post mes 12)

## References

- `.decision-log.md` — full decision history with rationale
- `EPICS.md` — milestone breakdown
- `STORIES/epic-1-foundation/` — per-story specs (MVP semana 1)
- Sephirot deck `~/Dev/sephirot-cc/presentations/veo-padelgraph-case-study-20260528.html`
