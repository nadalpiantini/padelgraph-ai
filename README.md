# padelgraph-ai

> Open-source AI analytics pipeline for padel — ball + player + court detection, multi-camera 3D fusion, shot classification.

**Status:** pre-alpha · MVP semana 1 in progress · Apache 2.0

## What it does (planned)

Process video from 1–4 cameras filming a padel match (including just phones) and produce structured analytics:

- **Ball detection + 3D trajectory** via YOLOX + Kornia multi-camera triangulation
- **Player detection + skeleton** via YOLOX + MediaPipe BlazePose
- **Court keypoint detection** via OpenCV homography
- **Shot classification** (volea / derecha / revés / smash / bandeja / víbora / globo) — roadmap, mes 6+
- **Per-player feedback report** — roadmap, mes 6+

## Status

| Capability | Status | Target |
|------------|--------|--------|
| Single-cam ball + player detection | 🚧 in progress | MVP semana 1 |
| Court homography (manual 4 corners) | 🚧 in progress | MVP semana 1 |
| Multi-cam sync (2 cameras) | 🚧 in progress | MVP semana 1 |
| Kornia 3D triangulation | 🚧 in progress | MVP semana 1 (stretch) |
| JSON unified output | 🚧 in progress | MVP semana 1 |
| Auto court calibration | ⏳ planned | mes 2 |
| BlazePose pose tracking | ⏳ planned | mes 2 |
| Shot classifier custom | ⏳ planned | mes 6 |
| Integration with padelgraph-app | ⏳ planned | mes 9 |

## Known Limitations (Epic 1 MVP)

The current release is the MVP scaffold. Some surfaces work end-to-end against
synthetic data, but the following are intentionally deferred — every item
below is documented here so the CLI never *silently* misbehaves:

- **YOLOX checkpoint loader is an Epic 2 stub.** Passing `--checkpoint <path>`
  is parsed and the file existence is validated, but if the optional `yolox`
  package is not installed the detector silently falls back to **stub mode**
  and returns 0 detections per frame. A `UserWarning` (and a stderr `WARNING:`
  line from the CLI) is emitted whenever this fallback kicks in so the empty
  output cannot go unnoticed. Install YOLOX or omit `--checkpoint` to silence
  the warning.
- **Court keypoints are picked manually.** `CourtDetector.detect_keypoints`
  opens an interactive `cv2` window for the user to click the four corners.
  There is no auto-detection — that lands in Epic 2.
- **Multi-camera fusion is limited to 2 cameras.** The triangulator and CLI
  both run with `N >= 1`, but the MVP is only validated against 2 cameras;
  4-camera support and confidence-weighted N-cam fusion are Epic 2.
- **No pose tracking.** BlazePose integration is mes 2.
- **Single-ball assumption.** The pipeline keeps the highest-confidence
  `sports ball` detection per frame; doubles-ball edge cases (e.g. a ball
  bouncing back into play during a rally) are not modeled.
- **Long-video memory profile is untested.** Videos longer than ~30 minutes
  may degrade performance — the MVP smoke suite only exercises ~1s synthetic
  clips.
- **No real-time inference.** Everything is post-process only; streaming /
  on-the-fly inference is out of scope for Epic 1.

## Install

Requires Python 3.12, [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/nadalpiantini/padelgraph-ai.git
cd padelgraph-ai
uv sync
```

Optional extras:

```bash
uv sync --extra pose      # MediaPipe BlazePose
uv sync --extra ui        # Streamlit dev dashboard
uv sync --extra dev       # pytest + ruff
```

**YOLOX installation** (separate from `uv sync` for now — story-002):

```bash
uv pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git
```

## Usage (planned MVP CLI)

```bash
padelgraph-ai-infer \
  --video data/inputs/match_cam1.mp4 \
  --video data/inputs/match_cam2.mp4 \
  --calib data/inputs/calib_2cam.json \
  --out data/outputs/run-001.json \
  --overlay data/outputs/run-001-overlay.mp4
```

## JSON output schema

```json
{
  "meta": {
    "match_id": "match-001",
    "cameras": [
      {"id": "cam1", "intrinsics": {...}, "extrinsics": {...}},
      {"id": "cam2", "intrinsics": {...}, "extrinsics": {...}}
    ],
    "fps": 30,
    "court_keypoints_2d_per_cam": {"cam1": [...], "cam2": [...]}
  },
  "frames": [
    {
      "frame_id": 0,
      "ts": 0.0,
      "players_per_cam": {
        "cam1": [{"id": 1, "bbox": [x, y, w, h], "confidence": 0.95}],
        "cam2": [{"id": 1, "bbox": [x, y, w, h], "confidence": 0.92}]
      },
      "ball": {
        "position_2d_per_cam": {"cam1": [x, y], "cam2": [x, y]},
        "position_3d": [x, y, z],
        "confidence": 0.87
      }
    }
  ]
}
```

## Stack

| Layer | Library | License |
|-------|---------|---------|
| Detection | [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Megvii) | Apache 2.0 |
| Pose | [MediaPipe BlazePose](https://github.com/google-ai-edge/mediapipe) | Apache 2.0 |
| Court / image | [OpenCV](https://github.com/opencv/opencv-python) | Apache 2.0 |
| Multi-cam 3D | [Kornia](https://github.com/kornia/kornia) | Apache 2.0 |
| Utils | [supervision](https://github.com/roboflow/supervision) (Roboflow) | MIT |
| Action recognition (roadmap) | [TorchVision video models](https://pytorch.org/vision/) | BSD-3 |
| Package mgmt | [uv](https://docs.astral.sh/uv/) | Apache 2.0 |
| Dev UI | [Streamlit](https://streamlit.io/) | Apache 2.0 |

**No AGPL dependencies.** Designed for commercial reuse from day 1.

## Project structure

```
padelgraph-ai/
├── PRD.md                          Product requirements
├── ARCHITECTURE.md                 Tech decisions
├── EPICS.md                        Milestone breakdown
├── STORIES/                        Per-epic tickets
│   └── epic-1-foundation/          MVP semana 1
├── .decision-log.md                Bitemporal decision history
├── src/padelgraph_ai/
│   ├── detection/                  YOLOX wrapper
│   ├── pose/                       BlazePose (mes 2)
│   ├── court/                      OpenCV homography
│   ├── sync/                       Multi-video sync
│   ├── fusion/                     Kornia 3D triangulation
│   └── pipeline/                   CLI entry point
├── src/padelgraph_ai_ui/           Streamlit dashboard (crudo)
├── tests/
├── data/{inputs,outputs}/          gitignored
└── notebooks/                      Jupyter exploratory
```

## Why this exists

There is no Apache/MIT padel CV pipeline maintained with >300 stars as of 2026-05-28. The closest reference, [Joao-M-Silva/padel_analytics](https://github.com/Joao-M-Silva/padel_analytics), is licensed CC BY-NC-SA which blocks commercial reuse. **padelgraph-ai aims to fill that gap.**

See [PRD.md](./PRD.md) for full context.

## Contributing

This is pre-alpha; APIs will change. PRs welcome once MVP semana 1 lands. See [ARCHITECTURE.md](./ARCHITECTURE.md) for module boundaries and [.decision-log.md](./.decision-log.md) for context.

## License

Apache License 2.0 — see [LICENSE](./LICENSE).
