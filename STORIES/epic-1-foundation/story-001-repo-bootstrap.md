# Story 001 — Repo bootstrap

**Epic:** 1 (Foundation, MVP semana 1)
**Status:** in progress (the scaffold is being created in this session)
**Owner:** Sephirot + Alan
**Estimate:** ~2h
**Target completion:** día 1 (2026-05-28)

---

## Goal

Stand up the `padelgraph-ai` repo with all foundation files in place: Apache LICENSE, README, pyproject.toml, .gitignore, GitHub Actions CI, public GitHub repo, first commit pushed.

## WHAT

Create and populate the following files (mostly already in flight as of this story's writing):

- `LICENSE` — Apache 2.0 standard text
- `.gitignore` — Python + data + IDE patterns
- `README.md` — install + run + output schema + roadmap
- `pyproject.toml` — uv-managed, Python 3.12, dependencies pinned (kornia, opencv-python, supervision, pydantic, click, torch, torchvision, numpy, pyyaml, tqdm). Optional extras: `pose` (mediapipe), `ui` (streamlit), `dev` (pytest, ruff)
- `PRD.md`, `ARCHITECTURE.md`, `EPICS.md`, `.decision-log.md` — planning artifacts
- `STORIES/epic-1-foundation/story-{001..005}.md` — this story + 4 implementation stories
- `src/padelgraph_ai/__init__.py` — package marker with `__version__`
- `tests/__init__.py` — package marker
- `.github/workflows/ci.yml` — GitHub Actions running `uv sync --extra dev`, `ruff check`, `pytest`

## WHERE

Repo root: `/Users/nadalpiantini/Dev/padelgraph-ai/`

GitHub: `https://github.com/nadalpiantini/padelgraph-ai` (PUBLIC, to be created)

## VERIFY

```bash
cd ~/Dev/padelgraph-ai
test -f LICENSE && echo "LICENSE present"
test -f pyproject.toml && echo "pyproject present"
test -f README.md && echo "README present"
test -f .decision-log.md && echo "decision-log present"
ls STORIES/epic-1-foundation/*.md | wc -l   # should be ≥5

# Validate pyproject parses
uv pip compile pyproject.toml --quiet 2>&1 | head -5
# (or: uv sync should succeed — that's the real verify)

# CI smoke
git init && git add . && git commit -m "scaffold"
gh repo create nadalpiantini/padelgraph-ai --public --source=. --push --description "..."

# Wait ~30s then verify
gh run list --repo nadalpiantini/padelgraph-ai --limit 1
```

## OFF_LIMITS

- NO publishing to PyPI in this story (manual release later, story will be in epic 5)
- NO including model weights in the repo (those go to R2 / HuggingFace later)
- NO real video files in `data/inputs/` (gitignored — Alan provides locally)
- NO touching `padelgraph-app` repo
- NO BMAD framework install (only the pattern is borrowed — see `.decision-log.md` D2)

## Acceptance criteria

- All foundation files exist at the paths above
- `git status` shows clean working tree after commit
- GitHub repo URL is live and public
- First CI run is green (or yellow if optional deps fail; main pipeline must succeed)
- README renders correctly on github.com

## Notes

This story is unusual because Sephirot orchestrator + ZAI workers create the files during the same session as the brainstorming. Future stories (002–005) follow the more typical pattern: spec → dispatch ZAI worker with caveman brief → review.
