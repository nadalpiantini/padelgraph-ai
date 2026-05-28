# Epics — padelgraph-ai

**Status:** v0.1
**Date:** 2026-05-28
**Pattern:** Epics mapped 1-to-1 to D24 12-month retrospect criteria from PRD

---

## Epic 1 — Foundation (MVP semana 1)

**Goal:** End-to-end pipeline that processes 2 phone-recorded padel videos → produces unified JSON with ball 3D trajectory + player bbox per camera + court keypoints. Validates the multi-camera architectural bet early.

**Target completion:** 2026-06-03 (semana 1)

**Stories:** see `STORIES/epic-1-foundation/`
1. `story-001-repo-bootstrap.md` — Apache LICENSE, README, pyproject.toml, .gitignore, CI workflow, GitHub public repo
2. `story-002-single-cam-detection.md` — YOLOX wrapper → bbox players + ball from 1 video
3. `story-003-multi-cam-sync.md` — Sync 2 video streams via timestamp + manual offset + frame interpolation
4. `story-004-court-homography.md` — OpenCV detect 4 court corners + homography 2D
5. `story-005-triangulation-3d.md` — Kornia triangulate ball 3D position from 2 calibrated cameras

**D24 criterion satisfied:** #1 — MVP multi-camera POC functional with 2 real videos

**Risk:** triangulation 3D non-trivial; if EOW day 5 fails, single-cam stays as foundation (no semana 2 restart).

---

## Epic 2 — Detection robustness + auto court calibration (mes 1–2)

**Goal:** Push ball + player detection accuracy above 70% on Alan's real match library. Eliminate manual court keypoint picking — replace with auto-detection.

**Target completion:** 2026-07-31

**Planned stories:**
- Fine-tune YOLOX on padel-specific dataset (PadelVic + augmented)
- Implement court keypoint regression model (custom or off-the-shelf)
- Add player tracking across frames (ByteTrack or DeepSORT integration via supervision)
- Add 4-camera support to `MultiCamSync` and `Triangulator`
- Benchmarks: accuracy report on 10 Alan matches

**D24 criteria satisfied:** #2 (accuracy >70%), #3 (auto court calibration)

---

## Epic 3 — Pose tracking (mes 2–3)

**Goal:** Integrate MediaPipe BlazePose to produce 13-DoF player skeletons per frame. Enable downstream shot classification training.

**Target completion:** 2026-08-31

**Planned stories:**
- `pose/blazepose.py` — BlazePose wrapper consuming player bbox crops
- `schemas.Skeleton` Pydantic model + JSON output extension
- Visualization overlay for skeletons in the optional output video
- Smoke test on 3 padel matches (rally, serve, smash)

**D24 criterion satisfied:** #4 — pose tracking integrated

---

## Epic 4 — Shot classifier custom (mes 3–6)

**Goal:** Train a custom shot classifier on Alan's labeled match library (target ≥500 labeled clips) reaching >75% accuracy. Classifies: volea / derecha / revés / smash / bandeja / víbora / globo.

**Target completion:** 2026-11-30

**Planned stories:**
- Data collection pipeline (extract clips from match videos by rally boundaries)
- Labeling tool (Streamlit-based, exports JSON)
- Training pipeline (MMAction2 or TorchVision video models)
- Evaluation harness + accuracy/confusion matrix
- Public dataset release (anonymized, Apache-licensed) if Alan agrees

**D24 criterion satisfied:** #5 — shot classifier >75%

**Open question:** synthetic data augmentation from Premier Padel highlights (CC content) vs Alan's library only vs both? Decide at start of epic.

---

## Epic 5 — `padelgraph-app` integration (mes 6–9)

**Goal:** `padelgraph-ai` consumed by `padelgraph-app` as a Python library (pip-installable or git submodule). Live matches recorded via LiveKit → exported as 2-camera video pairs → processed via `padelgraph-ai-infer` → results surfaced in `padelgraph-app` player profile UI.

**Target completion:** 2027-02-28

**Planned stories:**
- Publish `padelgraph-ai` to PyPI (or GitHub package registry)
- Async worker in `padelgraph-app` (`apps/padelgraph-app/src/jobs/analytics-process.ts`) that calls Python pipeline via subprocess or microservice
- Player profile UI extension: shot accuracy charts, court heatmap, weak-pattern callouts
- Privacy: explicit opt-in flow per player before any AI feedback is generated
- Webhook from `padelgraph-app` → `padelgraph-ai` to trigger processing on match-end

**D24 criterion satisfied:** #6 — integrated as optional live feature

**Cross-project boundary:** `padelgraph-app` is in TypeScript / Next.js, `padelgraph-ai` is in Python. Communication via subprocess (simple) or FastAPI microservice (scalable). Decide at start of epic per the data volume expected by then.

---

## Epic 6 — Real club deployment (mes 9–12)

**Goal:** ≥5 clubs in RD/PR actively using the pro feedback feature for real player development.

**Target completion:** 2027-05-31

**Planned stories:**
- Pilot program design (free tier for 3 clubs, paid for 2)
- Onboarding playbook (camera setup, account, coach training)
- Feedback iteration loop (weekly call with each pilot club, capture feature requests)
- Pricing model decision (per-club subscription, per-match credit, hybrid)
- Outreach to padel federation RD for endorsement / co-marketing

**D24 criterion satisfied:** #7 — ≥5 clubs using feedback

**Cross-track dependency:** parallel hardware validation track (BLAUBECK distribution, see Sephirot deck S18) should be running so clubs can buy a turnkey solution.

---

## Bonus Epic — OSS adoption (continuous, mes 1–12)

**Goal:** Reach >300 GitHub stars by mes 12, becoming the de-facto open-source padel CV pipeline.

**Tactics:**
- Public release per epic milestone (semana 1, mes 1, mes 3, mes 6, mes 9, mes 12)
- Blog post per release (Sephirot publishes on `~/Dev/sephirot-cc/` website or Alan's personal channels)
- Submit to "Awesome Computer Vision" and "Awesome Sports Analytics" lists
- Conference talk at PadelTech RD or Padel World Tour developer track (if exists)
- HuggingFace model card for shot classifier (when ready)

**D24 criterion satisfied:** #8 (bonus) — repo >300★

---

## Retrospective gate (2027-05-28)

Per `.decision-log.md` decision #11 (D24 12-month filter), end-of-year retrospective evaluates:

- ≥5/8 criteria met → continue, plan year 2
- 3–4/8 met → pivot scope (drop bottom epics, double down on what worked)
- <3/8 met → retire and write public post-mortem
