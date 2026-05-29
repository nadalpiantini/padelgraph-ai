# Handoff: padelgraph-ai post-audit — ready for Epic 2 + bridge to padelgraph.com

**Generated:** 2026-05-29 (RD/AST)
**Branch:** main
**Commit:** 8737e1a
**Status:** Ready for next phase (Epic 2 hardening OR Epic 5 bridge to padelgraph.com)
**Node:** M2 Air (orchestrator) · all work pushed to GitHub origin/main

---

## Goal

Build padelgraph-ai: Apache 2.0 Python CV pipeline for padel match analysis (ball + player + court detection, Kornia multi-camera 3D triangulation, JSON output). Per Sephirot business panel verdict (D12): reframe as "schema + dataset + reputation play", not a product.

Long-term: integrate with **padelgraph.com / padelgraph-app** (Next.js + Supabase + LiveKit) so club operators can click "Generate AI Analysis" on a match → asynchronous Python worker processes video pair from R2 → MatchAnalysis JSON appears in frontend tab.

---

## Completed

- [x] Repo scaffolded standalone at `~/Dev/padelgraph-ai/` (D28 isolation respected)
- [x] GitHub public Apache 2.0: https://github.com/nadalpiantini/padelgraph-ai
- [x] 4 commits on main, **CI green** (latest run #26607923223)
- [x] Epic 1 MVP foundation complete: 5 stories (detection, sync, court, fusion, pipeline)
- [x] Stack pinned: YOLOX (stub mode), MediaPipe, OpenCV, Kornia, supervision, uv, Streamlit, Python 3.12
- [x] Shared `schemas.py` Pydantic contract (7 models: Detection / SyncedFrameBatch / CourtKeypoints / Calibration / Point3D / BallObservation / FrameAnalysis / MatchMeta / MatchAnalysis)
- [x] **46 tests pass** (smoke + 7 detection + 8 sync + 4 homography + 7 triangulate + 6 pipeline + 7 scenarios + 2 init)
- [x] ruff check + format clean across 25 files
- [x] `uv build --wheel` succeeds → `dist/padelgraph_ai-0.1.0-py3-none-any.whl`
- [x] CLI `padelgraph-ai-infer` end-to-end with --video (multi) --calib --out --overlay --checkpoint --match-id --max-frames --verbose --quiet
- [x] Audit P0/P1 fixes shipped (6 fixes F1-F6 + defensive singular-matrix guard in Homography)
- [x] Scenario simulator `tests/test_scenarios.py` (710 LoC: ScenarioBuilder + RallyGenerator + HumanErrorInjector + VariableSweep)
- [x] Business panel debate captured: `BUSINESS_PANEL.md` (Christensen / Porter / Kim·Mauborgne / Taleb / Drucker, debate mode)
- [x] `/padelgraph-stress-test` Sephirot skill installed at `~/.claude/skills/padelgraph-stress-test/SKILL.md` with Ollama Cloud devstral-2 parallel orchestration + cost gate D24
- [x] 13 decisions logged in `.decision-log.md` (D1-D13, bitemporal D17 pattern)
- [x] Veo case study deck shipped at `~/Dev/sephirot-cc/presentations/veo-padelgraph-case-study-20260528.html`

---

## Not Yet Done

### Epic 2 backlog (audit surfaced — by priority)

- [ ] **EP2-R6 (HIGHEST):** Real YOLOX checkpoint loader. Currently stub mode. Production blocker. **Two paths:** (a) fix `onnx-simplifier` Python 3.12 incompat by pinning `onnx-simplifier<=0.4` in install command, or (b) migrate to `detectron2` (Apache 2.0, no install issues on Py 3.12). Try (a) first — 30 min spike.
- [ ] **EP2-R1 (HIGH):** Calibration `cam_id` ≠ `--video` positional order = silent footgun (discovered by Track 2 simulator `test_pipeline_handles_wrong_cam_labels`). Add Pydantic validator warning. Tied to EP2-R5.
- [ ] **EP2-R3:** Per-camera CLI `--offset cam_id seconds` flag — `HumanErrorInjector.sync_drift_seconds` records intent but no way to actually exercise sync module's drift handling without this. Add to `pipeline/infer.py`.
- [ ] **EP2-R2:** Expand `VariableSweep` to full 12 configs (3 cam_counts × 2 resolutions × 2 durations) under `@pytest.mark.slow` for nightly CI. Unparametrize `[:1]` slice in `tests/test_scenarios.py:704`.
- [ ] **EP2-R4:** `tests/conftest.py` shared fixtures (`_write_synthetic_video`, `_calibration_dict`) — currently duplicated ~40 LoC between `test_pipeline.py` and `test_scenarios.py`.
- [ ] Court auto-detection — replace manual keypoint picker (Epic 2 mes 2). Train custom keypoint regression model with PadelVic dataset + augmented synthetic.
- [ ] BlazePose pose tracking integration (Epic 2 mes 2-3) — `src/padelgraph_ai/pose/` stub already exists.
- [ ] Shot classifier custom (Epic 3-4 mes 6) — MMAction2 or TorchVision video models trained on >500 Alan-labeled clips.
- [ ] 4-camera support + bundle adjustment for N-view triangulation (currently pairwise + average — Epic 2).
- [ ] Real video integration smoke test — requires Alan to grabar 2 videos reales (clave día 5-6).
- [ ] `MultiCamSync` with very large negative offsets spins long before terminating (perf issue, not crash — flagged by Track 1 worker as low-priority).

### Epic 5 — Bridge to padelgraph.com (per Alan question 2026-05-28)

Architectural answer given in chat last turn. Concrete next steps:

- [ ] **Audit `~/Dev/padelgraph-app/` repo** via Explore agent — map pg-boss queue layer, Supabase schema (matches, recordings tables), LiveKit recording → R2 flow, authentication. **Sephirot has NOT read this code recently** (only memory entries + Track 3 explorer touched tournament schema).
- [ ] Write Epic 5 stories in `STORIES/epic-5-integration/` based on audit:
  - story-01-fastapi-wrapper.md
  - story-02-pgboss-worker.md
  - story-03-supabase-match-analyses-migration.md
  - story-04-calibration-ui-wizard.md
  - story-05-analysis-frontend-tab.md
- [ ] Decide hosting path A/B/C (M1 self-host vs Railway vs Modal/Replicate). Recommendation: A first (cero costo + dataset control), migrate to B after ≥10 clubs.
- [ ] First bridge end-to-end with 1 real match: video pair upload → job queue → Python worker → MatchAnalysis JSON → frontend tab render.

### Business panel verdict actions (D12, 60-day window)

- [ ] Update `PRD.md` success criteria — add criterion #9: "≥1 LATAM federation or club signed as design partner contributing calibration profile" mes 6.
- [ ] Position MatchAnalysis JSON as open schema for padel data (publish standalone spec doc).
- [ ] Sign 3-5 LATAM clubs/federations as design partners (data contribution agreement, separately licensed CC-BY-NC or proprietary).
- [ ] Cross-position with Filmatron (sports cinema angle) — Alan as "LATAM padel data infrastructure person".

---

## Failed Approaches (Don't Repeat These)

### 1. YOLOX install on Python 3.12 (Track 1 worker, 2 attempts)

**Attempted:**
```bash
uv pip install "yolox @ git+https://github.com/Megvii-BaseDetection/YOLOX.git"
# → AssertionError: torch is required for pre-compiling ops (isolated build env has no torch)

uv pip install --no-build-isolation "yolox @ git+https://github.com/Megvii-BaseDetection/YOLOX.git"
# → packaging.version.InvalidVersion: Invalid version: 'unknown' (onnx-simplifier transitive dep)
```

**Why it failed:** modern setuptools (Python 3.12+) rejects legacy version strings used by `onnx-simplifier`. YOLOX has it as transitive dep.

**Why current approach is better:** Detector ships with stub mode — returns `[]` from `infer()` when YOLOX absent or `checkpoint_path is None`. Pipeline scaffolding works end-to-end without YOLOX. CLI emits `YOLOXStubModeWarning` via stderr + Python warnings when `--checkpoint` passed but stub active (F1 audit fix).

**For next session:** try `onnx-simplifier<=0.4` pin first. If still fails, switch to detectron2 (Apache 2.0).

### 2. First CI run #26604663559 — ruff I001 + format check

**Attempted:** workers ran `ruff check --fix` locally → clean. Did NOT run `ruff format`. CI also runs `ruff format --check src/ tests/` which failed on 9 worker-written files + `schemas.py` had I001 import sort issue (blank line between stdlib `typing` and third-party `numpy`).

**Why it failed:** worker briefs missed the `ruff format` step. Workers also didn't audit Opus-written files (off-limits per their scope).

**Why current approach is better:** Opus ran full local `ruff format src/ tests/` + `ruff check` post-worker before commit. Second CI run #26604835295 green.

**For next session:** **worker briefs must mirror CI checks exactly** (ruff check + ruff format + pytest). Add to caveman brief template.

### 3. Track 2 worker mid-run AttributeError

**Attempted:** Track 2 worker hit `AttributeError: 'MultiCamSync' object has no attribute 'report_seek_stats'` from `pipeline/infer.py:411` during initial run. Track 1 was in flight implementing F3.

**Why it happened:** Track 1 F3 brief asked for `report_seek_stats()` method; Track 1 finalized the method before Track 2's final run, but there was a window where Track 2 imported broken state.

**Why current approach is better:** Track 1 worker finalized F3 before reporting done; Track 2 final run was green (race window closed). For future parallel work touching cross-track interfaces, dispatch in **dependency-ordered batches** rather than full parallel.

### 4. Business panel skill assumed missing (Track 3 audit Phase 1)

**Attempted:** Phase 1 explore agent claimed `business-panel` skill "NOT EXISTS in sephirot-cc" based on grep of `config/skills/`.

**Why it failed:** explore agent scoped search to `sephirot-cc` only. `business-panel-experts` agent **DOES exist** in global Sephirot agent catalog (visible in `/agents` list with Christensen/Porter/Drucker/Godin/Kim·Mauborgne/Collins/Taleb/Meadows/Doumont).

**Why current approach is better:** invoked agent directly via Agent tool `subagent_type=business-panel-experts` in Track 3 dispatch — worked first try, returned 1500-word debate captured verbatim in `BUSINESS_PANEL.md`.

**For next session:** when checking if a skill/agent exists, search **global Sephirot catalog**, not just project-local `config/skills/`.

---

## Key Decisions

| Decision | Rationale | Doctrine |
|----------|-----------|----------|
| Path B from scratch (no `padel_analytics` code) | CC BY-NC-SA viral license blocks SaaS | D1 |
| BMAD-Hybrid pattern (no framework install) | 95% of BMAD skills are planning, irrelevant for CV pipeline | D2 |
| MVP B multi-cam day 1 (not single-cam) | Validates multi-cam architectural bet early; fallback to single-cam if Kornia 3D fails | D3 |
| YOLOX (Megvii) NOT YOLOv8 (Ultralytics) | YOLOv8 AGPL-3.0 contaminates SaaS | D4 |
| Kornia for multi-cam triangulation | Purpose-built diff CV, 10K★ Apache, OpenCV authors | D5 |
| Apache 2.0 license | Permissive, no viral, allows external contributors | D8 |
| Standalone repo `~/Dev/padelgraph-ai/` | D28 isolation — never inside `padelgraph-app/` or `sephirot-cc/` | D9, D28 |
| 5 Sephirot skills `.sephirot/skills/` deferred to mes 2-3 | Anti-premature-abstraction: build after 3 repetitive uses | D10 |
| D24 12mo retrospect: 8 criteria, retire if <5/8 | Force concrete success definition | D11 |
| **Reframe to "schema+dataset+reputation play"** (60-day window) | Business panel consensus: code depreciates, data appreciates | **D12** |
| Audit fixes + scenario sim + Ollama skill shipped in 1 commit | 4 parallel workers + 1 business-panel agent, D3 Tiferet pattern N=5 validated | D13 |

---

## Current State

**Working:**
- CLI `padelgraph-ai-infer` end-to-end against synthetic videos
- All 5 modules: detection (stub mode), sync (2-cam timestamp+offset), court (homography 2D), fusion (Kornia 3D triangulation), pipeline (orchestration)
- 46 tests pass, ruff clean, wheel builds
- CI green on every push
- `/padelgraph-stress-test` skill registered globally — visible in skill list

**Broken / Limited:**
- YOLOX real inference (stub mode — returns 0 detections). See Failed Approach #1.
- Court auto-detection (manual interactive picker only — Epic 2).
- No real video tested (only synthetic 1s clips). Alan to grabar 2 padel videos día 5-6.
- N-view triangulation (3+ cams) doesn't weight by confidence (Epic 2).
- Calibration JSON `cam_id` field silently re-keyed by positional order — wrong labels = silently wrong 3D (EP2-R1).

**Uncommitted Changes:**
- `HANDOFF.md` (this file) — to be committed and pushed in close.

---

## Files to Know

| File | Why It Matters |
|------|----------------|
| `PRD.md` | Product requirements + 8 D24 12mo criteria. Update if adopting business panel reframe (add criterion #9). |
| `ARCHITECTURE.md` | Tech decisions + module boundaries + data flow diagram |
| `EPICS.md` | 6 epics + bonus mapped to 12mo criteria |
| `STORIES/epic-1-foundation/` | 5 stories Epic 1 (all complete) |
| `.decision-log.md` | 13 decisions bitemporal D17 pattern — read this FIRST when resuming |
| `BUSINESS_PANEL.md` | 5 expert debate, verdict, 3 weakest assumptions, 1 missing path |
| `src/padelgraph_ai/schemas.py` | Pydantic contract (DO NOT MODIFY casually — affects all modules) |
| `src/padelgraph_ai/pipeline/infer.py` | CLI entry point, Click-based orchestration |
| `src/padelgraph_ai/detection/yolox_runner.py:175-200` | Stub mode hot zone — Epic 2 real loader goes here |
| `tests/test_scenarios.py` | Synthetic tournament + error injection simulator (7 tests under `scenarios` marker) |
| `scripts/stress_test.sh` + `_stress_worker.py` | Ollama Cloud parallel orchestrator — use `--dry-run` for safe smoke |
| `~/.claude/skills/padelgraph-stress-test/SKILL.md` | Sephirot global skill, invocable via `/padelgraph-stress-test` |
| `~/.claude/plans/snoopy-gathering-pudding.md` | Current plan file (audit+stress+biz panel — completed) |

---

## Code Context

**CLI signature:**
```bash
padelgraph-ai-infer \
  --video PATH (multiple, required, expect 2) \
  --calib PATH (required, JSON list of Calibration) \
  --out PATH (required, MatchAnalysis JSON output) \
  --overlay PATH (optional, mp4 with bbox + ball trail) \
  --checkpoint PATH (optional YOLOX; stub if absent) \
  --match-id STR (optional, defaults to out.stem) \
  --max-frames INT (optional cap for smoke) \
  --verbose / --quiet (logging level)
```

**MatchAnalysis JSON shape:**
```json
{
  "meta": {
    "match_id": "...",
    "cameras": [{"cam_id": "cam1", "intrinsics": [[..3x3..]], "extrinsics": [[..4x4..]], "image_size": [w, h]}, ...],
    "fps": 30,
    "court_dim_meters": [20.0, 10.0],
    "court_keypoints_2d_per_cam": {"cam1": {"top_left": [x,y], ...}, "cam2": {...}}
  },
  "frames": [
    {
      "frame_id": 0,
      "ts": 0.0,
      "players_per_cam": {"cam1": [Detection, ...], "cam2": [...]},
      "ball": {
        "position_2d_per_cam": {"cam1": [x, y], "cam2": [x, y]},
        "position_3d": {"x": ..., "y": ..., "z": ..., "confidence": ...} | null,
        "confidence": 0.87
      }
    }
  ]
}
```

**Calibration helper (story-005):**
```python
from padelgraph_ai.fusion.calibrate import compute_extrinsics_from_court_keypoints
# Uses cv2.solvePnP SOLVEPNP_IPPE (NOT IPPE_SQUARE — winding/centering constraints incompatible
# with our canonical CourtKeypoints order: TL, TR, BR, BL clockwise from origin).
```

**Stub mode trigger (DO NOT silently bypass):**
```python
# src/padelgraph_ai/detection/yolox_runner.py
# When YOLOX uninstalled OR checkpoint_path is None:
warnings.warn(
    f"STUB MODE active: checkpoint {checkpoint_path} provided but YOLOX not installed. "
    "Returning 0 detections per frame. Epic 2 implements real loader.",
    YOLOXStubModeWarning,
)
```

---

## Resume Instructions

### Scenario A — Continue padelgraph-ai Epic 2 (YOLOX real loader + court auto + pose)

1. `cd ~/Dev/padelgraph-ai && git pull origin main` (M2 just pushed, but safe check)
2. `uv sync --extra dev` — verify dependencies still resolve
3. `uv run pytest tests/ -v` — should be 46 passed
4. **First Epic 2 task: try fix YOLOX install**
   ```bash
   uv pip install "onnx-simplifier<=0.4"
   uv pip install --no-build-isolation "yolox @ git+https://github.com/Megvii-BaseDetection/YOLOX.git"
   ```
   - Expected: install succeeds, `python -c "import yolox"` works
   - If fails: switch to detectron2 — `uv pip install detectron2` + rewrite `Detector` to use Detectron2 API (preserve same public interface)
5. Implement real loader in `src/padelgraph_ai/detection/yolox_runner.py:175-200`. Wire `Exp.get_model()` + `load_state_dict()`.
6. Add test `test_yolox_loads_real_checkpoint` behind `@pytest.mark.gpu` skip if model file absent.
7. Update `.decision-log.md` with D14 entry capturing the fix.

### Scenario B — Build bridge to padelgraph.com (Epic 5 acceleration)

1. **Dispatch Explore agent on padelgraph-app first** (D29 — read code before designing bridge):
   ```
   "Audit ~/Dev/padelgraph-app/ for: (a) pg-boss/queue infrastructure, (b) Supabase tables related to matches + recordings, (c) LiveKit recording → R2 flow, (d) authentication between Next.js and any background workers. Output: 1-page map of integration points where padelgraph-ai Python worker would plug in."
   ```
2. After audit returns: invoke `superpowers:brainstorming` skill + create Epic 5 spec
3. Write 5 stories in `STORIES/epic-5-integration/`
4. Decide hosting path A/B/C (A = M1 self-host, B = Railway, C = Modal) — recommend A first
5. Implement first slice: FastAPI wrapper around `padelgraph-ai-infer` CLI → deploy → test with 1 real match end-to-end

### Scenario C — Adopt business panel verdict (60-day window starts 2026-05-28)

1. Update `PRD.md` Success Criteria — add criterion #9 (≥1 LATAM design partner contributing calibration profile by mes 6).
2. Draft outreach playbook for 5 LATAM padel clubs/federations (Dominican Republic + Puerto Rico first).
3. Publish standalone MatchAnalysis JSON schema spec doc (e.g. `SCHEMA_SPEC.md` in repo root, also consider OpenAPI YAML).
4. Position cross-link with Filmatron — Alan brand portfolio "LATAM AI sports + cinema infrastructure".

### Scenario D — Real stress test (low cost validation)

1. `cd ~/Dev/padelgraph-ai`
2. Verify Ollama Cloud key: `echo $OLLAMA_API_KEY | head -c 20` (should be non-empty)
3. Run real stress test N=20: `bash scripts/stress_test.sh 20 8 devstral-2:cloud`
4. Review `artifacts/stress-test-<timestamp>.md` for anomalies
5. Surface top 5 findings → file as Epic 2 backlog issues

---

## Setup Required

- **Python 3.12** (NOT 3.13 — MediaPipe incompat per Context7 verified)
- **uv 0.4+** (`brew install uv` or `pipx install uv`)
- Local clone: `git clone https://github.com/nadalpiantini/padelgraph-ai.git ~/Dev/padelgraph-ai`
- Dependencies: `cd ~/Dev/padelgraph-ai && uv sync --extra dev`
- For Ollama stress-test: `OLLAMA_API_KEY` env var (sourced from `~/.freejack-credentials.env`)
- For real video test (día 5-6): 2 padel match videos (any resolution, h264 mp4) + 1 frame extracted as JPG per cam for court keypoint picking
- **NO YOLOX checkpoint required** for current functionality — stub mode is the contract until Epic 2

---

## Mesh / Sync Notes

- All work shipped from **M2 Air** (orchestrator), pushed to GitHub origin/main
- No M1 or M1ni state changes this session
- `/padelgraph-stress-test` skill at `~/.claude/skills/` — needs to propagate to M1/M1ni via standard mesh sync if those nodes will invoke it (currently M2-only)
- Sephirot doctrines updated NONE this session (D12-D13 are project-specific, not system-level — no D30+ candidates emerged)
- `~/.openclaw/policies/ollama-cloud.json` defines devstral-2 rate limit 120 RPM — shared across nodes, no per-node config needed

---

## Edge Cases & Error Handling

- **Pipeline doesn't crash with corrupted video** — caught by `test_pipeline_handles_corrupt_video`, emits friendly error
- **1-camera input** — pipeline runs, `ball.position_3d = null` for every frame (Triangulator needs ≥2 observations)
- **Missing calibration field** — CalibrationParseError with explicit camera + field name (F4)
- **Sync seek failure** — retry 2x with 100ms delay, count per-camera, warn at end if >5% batches partial (F3)
- **Wrong calib `cam_id` order** — pipeline silently re-keys by `--video` positional order. **THIS IS A FOOTGUN** — see EP2-R1.
- **Negative offset large** — sync skips that cam until `target_ts >= |offset|`. Doesn't crash but may spin long iterations on small video + huge negative offset (perf, not correctness).
- **YOLOX checkpoint specified but stub mode active** — Python warning + stderr CLI warning emitted (F1). User cannot miss it.

---

## Warnings

- **`schemas.py` is the contract — modify with extreme caution.** All 5 modules depend on it. Adding a new field is OK; renaming or removing breaks every module.
- **`Detection.class_id` validator** has no `Literal[0, 32]` constraint despite tests asserting `class_id in {0, 32}`. Malformed external JSON with class_id=99 would pass Pydantic but later violate. EP2-R5 territory.
- **`SOLVEPNP_IPPE` not `SOLVEPNP_IPPE_SQUARE`** — the SQUARE variant requires centered-square + counter-clockwise winding which the canonical CourtKeypoints clockwise order violates. Plain IPPE is correct here.
- **`MultiCamSync.target_fps` defaults to `min()` of per-camera FPS** — if cameras differ wildly (30fps vs 60fps), sync downsamples to 30. Document in any user-facing tutorial.
- **`uv build --wheel` includes `src/padelgraph_ai_ui/`** even though it's currently just `__init__.py`. When Streamlit dashboard lands (Epic 5+), pyproject.toml `packages` entry needs no change.
- **`/padelgraph-stress-test` skill defaults to dry-run via flag** — real run requires explicit invocation. Cost gate D24 mandates Opus shows estimate + waits for yes/no. Do NOT shortcut.

---

## Referenced Artifacts

- **Plan file:** `~/.claude/plans/snoopy-gathering-pudding.md` (current plan, audit+stress+biz panel — completed)
- **Veo deck:** `~/Dev/sephirot-cc/presentations/veo-padelgraph-case-study-20260528.html` (strategic case study, Working doc internal)
- **Sephirot doctrines:** `~/Dev/sephirot-cc/.claude/rules/sephirot-doctrines.md` + `sephirot-doctrines-2.md` + `sephirot-doctrines-3.md` (D1-D29)
- **MemPalace drawers:** `wing=sephirot, room=decisions` — multiple drawers from 2026-05-28 sessions capture decisions verbatim
- **GitHub repo:** https://github.com/nadalpiantini/padelgraph-ai
- **CI run reference (latest green):** https://github.com/nadalpiantini/padelgraph-ai/actions/runs/26607923223
- **Joao-M-Silva/padel_analytics (rejected reference):** https://github.com/Joao-M-Silva/padel_analytics — CC BY-NC-SA, contact `jsilvawasd@hotmail.com` looking for collaborators if Path C ever revisited
- **BLAUBECK phone mount (hardware track):** https://blaubeck.com/products/padel-phone-holder-for-glass-courts-magsafe-compatible-with-double-suction-cup-aluminum-arm
- **Ollama Cloud policy:** `~/.openclaw/policies/ollama-cloud.json` — rate limits per model

---

## Open Question Pending (Alan asked, answered architecturally, not implemented)

**Pregunta Alan 2026-05-28 (cierre):** *"como vamos a acceder a esto via padelgraph.com? app?"*

**Response given (chat):** architectural answer with 3 hosting paths (A=M1 self-host, B=Railway, C=Modal) + bridge pattern (pg-boss queue + Python worker + Supabase table + frontend tab) + Epic 5 timeline.

**Pending action:** Alan did NOT yet OK starting Epic 5 audit. Two natural next sessions:
1. Audit padelgraph-app first → spec Epic 5 → build bridge
2. Or fix Epic 2 YOLOX first (production blocker per audit), THEN bridge

Recommendation per business panel D12: **build bridge sooner than plan original mes 9** because without user-facing exposure the code is paper. Alan's call.
