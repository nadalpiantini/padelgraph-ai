# Product Requirements — padelgraph-ai

**Status:** v0.1 — pre-alpha
**Date:** 2026-05-28
**Owner:** Alan Nadal Piantini
**Pattern:** BMAD-inspired (PRD → Epics → Stories) without installing BMAD framework

---

## Problem

Padel coaches and players have effectively no open-source AI analytics. Existing commercial options (PlaySight, Pixellot, Veo) either don't support padel at all or have no LATAM presence + pricing accessibility. The closest open-source padel pipeline (`Joao-M-Silva/padel_analytics`, 248★) is licensed CC BY-NC-SA — which prohibits commercial reuse and contaminates any SaaS built on top.

There is **no Apache/MIT padel CV pipeline maintained with >300★** as of 2026-05-28. Verified via GitHub topic search and the deck audit at `presentations/veo-padelgraph-case-study-20260528.html` in the Sephirot repo.

## Why now

- Padel global market: €2B (2023 Deloitte), 25M players, 90+ countries, 84K courts projected by 2026
- USA explosion: 180 courts (2022) → 1000+ (Jan 2026), projection 30K by 2030
- LATAM dominates competitive ranking (Argentina/Brasil/Uruguay/Colombia/Paraguay)
- `padelgraph-app` already has live multi-camera LiveKit infrastructure (4-camera grid + R2 recording) — needs hardware ingest + AI analytics on top
- Apache-licensed OSS components are mature in 2026 (YOLOX, Kornia, MediaPipe, supervision) — building from scratch is now faster than 2 years ago

## Users

| Persona | Use case | Frequency |
|---------|----------|-----------|
| **Padel coach** at academy | Reviews match video post-game → gets per-player feedback (heatmap, weak patterns, shot accuracy) | Weekly per student |
| **Pro padel player** | Reviews own matches → identifies tactical patterns | Daily during prep weeks |
| **Padel club owner** (LATAM) | Broadcasts club matches with overlay analytics → differentiates from competing clubs | Per tournament/league |
| **Padelgraph-app integration** | Consumes the pipeline as a service to enrich existing match data with AI analytics | Per recorded match |
| **OSS contributor** (computer vision researcher) | Forks repo, contributes shot classifier improvements, uses padel as application domain | Per release |

## Success criteria

### MVP (semana 1, ending 2026-06-03)

- `padelgraph-ai-infer` CLI processes a pair of phone-recorded padel videos and produces a unified JSON with ball 3D trajectory + player bbox per camera + court keypoints. See [STORIES/epic-1-foundation/](./STORIES/epic-1-foundation/) for the 5 stories.

### 12-month criteria (2027-05-28)

Per `.decision-log.md` decision #11, the 12-month retrospect filter applies. To say "this was worth doing":

1. MVP multi-camera POC functional with 2 real videos (semana 1)
2. Ball + player detection accuracy >70% (mes 1) — YOLOX out-of-box, no fine-tune
3. Court auto-calibration without manual keypoints (mes 2)
4. Pose tracking (BlazePose) integrated (mes 2–3)
5. Custom shot classifier >75% accuracy (mes 6) — MMAction2 or TorchVision video, trained on ≥500 labeled videos
6. Integrated into `padelgraph-app` as optional live feature (mes 9)
7. ≥5 clubs in RD/PR actually using the pro feedback (mes 12)
8. **Bonus:** public GitHub repo with >300★ (mes 12) — fills the verified OSS gap

If fewer than 5 of 8 criteria are met by 2027-05-28, retire or pivot.

## Scope — MVP semana 1

| In | Out (anti-creep, explicit) |
|----|---------------------------|
| Single-camera ball + player detection (YOLOX) | 4-camera fusion (limit to 2 cameras) |
| Court detection (4 corners) + homography 2D | Shot classification (deferred to mes 3–6) |
| Multi-camera sync (timestamp + manual offset, 2 cameras) | Auto court calibration without manual seed (mes 2) |
| Kornia triangulation 3D for ball trajectory | Stats analytics (rally length, heatmaps, fatigue) — mes 3 |
| Unified JSON output | Pose tracking (BlazePose) — mes 2 |
| CLI entry point (`padelgraph-ai-infer`) | Polished UI (Streamlit is dev-only crudo) |
| Visual overlay video (optional) | `padelgraph-app` integration — mes 9 |
| Smoke test with 2 real Alan-provided videos | Hardware tower (separate validation track) |
| README + ARCHITECTURE + EPICS + STORIES + .decision-log | BMAD official install (using hybrid pattern only) |

## Constraints

- **License:** repo must be Apache 2.0; no AGPL dependencies (excludes YOLOv8/Ultralytics; YOLOX is the substitute)
- **Python:** 3.12 only (MediaPipe compatibility verified via Context7)
- **Hardware:** must run on Apple Silicon (M1/M2/M1ni mesh) for inference; GPU only for training (vast.ai/Scaleway on-demand)
- **No data leakage:** test videos stored in `data/inputs/` (gitignored), backed up to R2
- **Privacy:** any feedback report containing identifiable players requires explicit opt-in (planned for mes 9 integration)
- **Tone:** developer-facing technical content stays neutral; player-facing feedback follows D4 collaborative voice ("tu mejor tiro fue X" not "you did Y wrong")

## Non-goals

- Real-time live AI overlay during play (post-process only in MVP)
- Refereeing decisions / line calls (not the use case)
- Replacing human padel coaches (augmentation only)
- Selling hardware mounts (separate validation track — see deck S18)
- Replicating Veo's broadcast camera feature set (different value prop: analytics, not broadcast)

## Open questions

- Court keypoint auto-detection: train custom model (mes 2) or rely on manual seed indefinitely? — decision deferred to mes 2 retro
- Shot classification dataset bootstrap: synthetic augmentation (Premier Padel highlights) vs Alan's match library vs both? — decision deferred to mes 3
- Pricing model if SaaS-ified: per-club subscription, per-match credit, open-source-and-services? — decision deferred to mes 9

## References

- Sephirot deck `presentations/veo-padelgraph-case-study-20260528.html` — full market + competitor + landscape audit
- Sephirot plan `~/.claude/plans/snoopy-gathering-pudding.md` — current implementation plan
- Sephirot doctrine `D24 12-Month Retrospect Filter` — applied to this project
- Sephirot doctrine `D28 Nunca Ligar Proyectos` — padelgraph-ai lives standalone, NOT inside padelgraph-app or sephirot-cc
- `padelgraph-app` repo at `~/Dev/padelgraph-app/` — downstream consumer (integration mes 9)
