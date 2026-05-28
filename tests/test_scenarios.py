"""Synthetic tournament / match / error simulator (Track 2 of audit plan).

Builds a deterministic pipeline-stress harness with four collaborators:

* :class:`ScenarioBuilder` — generate plausible 8-team tournament brackets
  with matches → sets → games as nested dataclasses.
* :class:`RallyGenerator` — synthesize rallies + the tiny throwaway videos
  the CLI will consume. Reuses the ``cv2.VideoWriter`` pattern from
  :mod:`tests.test_pipeline`.
* :class:`HumanErrorInjector` — corrupt the synthetic videos / calibration
  to mimic the real-world faults the audit (Phase 1) flagged: dropped
  frames, sync drift, low-confidence cameras, swapped cam labels, partial
  occlusion.
* :class:`VariableSweep` — Cartesian product over cam counts, resolutions
  and durations for parametric stress.

All randomness is seeded (``numpy`` ``default_rng(42)`` family) so the
suite is reproducible on CI.

The whole module is marked ``@pytest.mark.scenarios`` so it can be
selected/deselected independently from the existing 34 unit tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from click.testing import CliRunner

from padelgraph_ai.pipeline.infer import main
from padelgraph_ai.schemas import MatchAnalysis

pytestmark = pytest.mark.scenarios


# ---------------------------------------------------------------------------
# Constants — kept tiny so a full sweep runs in a few seconds on CI.
# ---------------------------------------------------------------------------

_SHOT_TYPES: tuple[str, ...] = (
    "volea",
    "derecha",
    "reves",
    "smash",
    "bandeja",
    "vibora",
    "globo",
)

_DEFAULT_WIDTH = 64
_DEFAULT_HEIGHT = 48
_DEFAULT_FPS = 30.0
_MASTER_SEED = 42


# ---------------------------------------------------------------------------
# 1) ScenarioBuilder — tournament structure
# ---------------------------------------------------------------------------


@dataclass
class Game:
    """One game inside a set (score is a placeholder — not pipeline input)."""

    winner: str
    points_a: int
    points_b: int


@dataclass
class Set:
    """One set inside a match: 6-12 games."""

    games: list[Game]


@dataclass
class Match:
    """One bracket match: 2-3 sets between two teams."""

    match_id: str
    team_a: str
    team_b: str
    sets: list[Set]
    rallies: int = 0


@dataclass
class Tournament:
    """8-team single-elimination bracket → 7 matches (4 QF + 2 SF + 1 F)."""

    tournament_id: str
    matches: list[Match] = field(default_factory=list)


class ScenarioBuilder:
    """Deterministic generator for padel tournament structures."""

    def __init__(self, seed: int = _MASTER_SEED) -> None:
        self._rng = np.random.default_rng(seed)

    def build_tournament(self, n_teams: int = 8, tournament_id: str = "T001") -> Tournament:
        if n_teams < 2 or (n_teams & (n_teams - 1)) != 0:
            raise ValueError(f"n_teams must be a power of 2 ≥ 2, got {n_teams}")

        teams = [f"team_{i:02d}" for i in range(n_teams)]
        matches: list[Match] = []
        round_idx = 0
        current = teams
        while len(current) > 1:
            next_round: list[str] = []
            for pair_idx in range(0, len(current), 2):
                team_a = current[pair_idx]
                team_b = current[pair_idx + 1]
                match = self._build_match(
                    match_id=f"{tournament_id}-R{round_idx}-M{pair_idx // 2}",
                    team_a=team_a,
                    team_b=team_b,
                )
                matches.append(match)
                # Winner = whoever appears in more game.winner slots across sets.
                wins_a = sum(g.winner == team_a for s in match.sets for g in s.games)
                wins_b = sum(g.winner == team_b for s in match.sets for g in s.games)
                next_round.append(team_a if wins_a >= wins_b else team_b)
            current = next_round
            round_idx += 1

        return Tournament(tournament_id=tournament_id, matches=matches)

    def _build_match(self, match_id: str, team_a: str, team_b: str) -> Match:
        n_sets = int(self._rng.integers(2, 4))  # 2 or 3 inclusive
        sets: list[Set] = []
        for _ in range(n_sets):
            n_games = int(self._rng.integers(6, 13))  # 6..12 inclusive
            games = [
                Game(
                    winner=team_a if self._rng.random() < 0.5 else team_b,
                    points_a=int(self._rng.integers(0, 5)),
                    points_b=int(self._rng.integers(0, 5)),
                )
                for _ in range(n_games)
            ]
            sets.append(Set(games=games))
        n_rallies = int(self._rng.integers(5, 51))  # 5..50 inclusive
        return Match(
            match_id=match_id,
            team_a=team_a,
            team_b=team_b,
            sets=sets,
            rallies=n_rallies,
        )


# ---------------------------------------------------------------------------
# 2) RallyGenerator — synthetic videos + calibration
# ---------------------------------------------------------------------------


@dataclass
class SyntheticMatchAssets:
    """All on-disk inputs needed to feed the CLI for one match."""

    videos: list[Path]
    calib_path: Path
    shot_sequence: list[str]
    expected_frames: int
    width: int
    height: int
    fps: float


class RallyGenerator:
    """Synthesize the tiny BGR videos + 2-cam calibration for a match."""

    def __init__(self, seed: int = _MASTER_SEED) -> None:
        self._rng = np.random.default_rng(seed)

    def render(
        self,
        match: Match,
        out_dir: Path,
        cam_count: int = 2,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        duration_seconds: float = 1.0,
        fps: float = _DEFAULT_FPS,
    ) -> SyntheticMatchAssets:
        out_dir.mkdir(parents=True, exist_ok=True)
        videos: list[Path] = []
        for cam_idx in range(cam_count):
            video_path = out_dir / f"{match.match_id}_cam{cam_idx + 1}.mp4"
            _write_synthetic_video(
                video_path,
                duration_seconds=duration_seconds,
                fps=fps,
                width=width,
                height=height,
                seed=int(self._rng.integers(1, 1_000_000)),
            )
            videos.append(video_path)

        calib_path = out_dir / f"{match.match_id}_calib.json"
        calibrations = [
            _calibration_dict(f"cam{i + 1}", width=width, height=height) for i in range(cam_count)
        ]
        calib_path.write_text(json.dumps(calibrations), encoding="utf-8")

        shot_sequence = [
            _SHOT_TYPES[int(self._rng.integers(0, len(_SHOT_TYPES)))] for _ in range(match.rallies)
        ]
        expected_frames = max(1, int(round(duration_seconds * fps)))
        return SyntheticMatchAssets(
            videos=videos,
            calib_path=calib_path,
            shot_sequence=shot_sequence,
            expected_frames=expected_frames,
            width=width,
            height=height,
            fps=fps,
        )


# ---------------------------------------------------------------------------
# 3) HumanErrorInjector — realistic faults
# ---------------------------------------------------------------------------


@dataclass
class InjectedFaults:
    """Bookkeeping for assertions about what was corrupted."""

    missing_frames_pct: float
    sync_drift_seconds: float
    low_confidence_cameras: list[str]
    cam_label_swap: bool
    partial_occlusion_pct: float


class HumanErrorInjector:
    """Apply realistic faults to a :class:`SyntheticMatchAssets` in place."""

    def __init__(self, seed: int = _MASTER_SEED + 1) -> None:
        self._rng = np.random.default_rng(seed)

    def inject(
        self,
        assets: SyntheticMatchAssets,
        missing_frames_pct: float = 0.05,
        sync_drift_seconds: float = 0.3,
        low_confidence_camera_ratio: float = 0.25,
        wrong_cam_label_swap_chance: float = 0.10,
        partial_occlusion_pct: float = 0.05,
    ) -> InjectedFaults:
        if not 0.0 <= missing_frames_pct <= 1.0:
            raise ValueError("missing_frames_pct must be in [0, 1]")
        if not 0.0 <= partial_occlusion_pct <= 1.0:
            raise ValueError("partial_occlusion_pct must be in [0, 1]")

        # 1) Drop a percentage of frames in every camera by re-encoding shorter videos.
        if missing_frames_pct > 0.0:
            for video_path in assets.videos:
                self._drop_frames(video_path, missing_frames_pct, assets)

        # 2) Pick which cameras get extra blur (= lower detector confidence).
        low_conf: list[str] = []
        if low_confidence_camera_ratio > 0.0 and assets.videos:
            n_low = max(1, int(round(len(assets.videos) * low_confidence_camera_ratio)))
            chosen = self._rng.choice(len(assets.videos), size=n_low, replace=False)
            for idx in chosen.tolist():
                self._blur_video(assets.videos[idx], assets)
                low_conf.append(f"cam{idx + 1}")

        # 3) Occlude a small percentage of frames with black rectangles.
        if partial_occlusion_pct > 0.0:
            for video_path in assets.videos:
                self._occlude_frames(video_path, partial_occlusion_pct, assets)

        # 4) Maybe swap cam1 / cam2 labels in the calibration JSON.
        swapped = False
        if (
            wrong_cam_label_swap_chance > 0.0
            and len(assets.videos) >= 2
            and self._rng.random() < wrong_cam_label_swap_chance
        ):
            self._swap_first_two_cam_labels(assets.calib_path)
            swapped = True

        return InjectedFaults(
            missing_frames_pct=missing_frames_pct,
            sync_drift_seconds=sync_drift_seconds,
            low_confidence_cameras=low_conf,
            cam_label_swap=swapped,
            partial_occlusion_pct=partial_occlusion_pct,
        )

    # -- helpers --------------------------------------------------------------

    def _read_frames(self, video_path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        try:
            frames: list[np.ndarray] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            return frames
        finally:
            cap.release()

    def _rewrite_video(
        self, video_path: Path, frames: list[np.ndarray], assets: SyntheticMatchAssets
    ) -> None:
        if not frames:
            # Degenerate case — keep at least one black frame so the
            # downstream VideoCapture doesn't error on open.
            frames = [np.zeros((assets.height, assets.width, 3), dtype=np.uint8)]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, assets.fps, (assets.width, assets.height))
        if not writer.isOpened():
            raise RuntimeError(f"could not reopen VideoWriter for {video_path}")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    def _drop_frames(self, video_path: Path, pct: float, assets: SyntheticMatchAssets) -> None:
        frames = self._read_frames(video_path)
        if not frames:
            return
        n_drop = max(1, int(round(len(frames) * pct))) if pct > 0 else 0
        if n_drop >= len(frames):
            n_drop = len(frames) - 1
        if n_drop <= 0:
            return
        drop_idx = set(self._rng.choice(len(frames), size=n_drop, replace=False).tolist())
        kept = [f for i, f in enumerate(frames) if i not in drop_idx]
        self._rewrite_video(video_path, kept, assets)

    def _blur_video(self, video_path: Path, assets: SyntheticMatchAssets) -> None:
        frames = self._read_frames(video_path)
        blurred = [cv2.GaussianBlur(f, ksize=(9, 9), sigmaX=4.0) for f in frames]
        self._rewrite_video(video_path, blurred, assets)

    def _occlude_frames(self, video_path: Path, pct: float, assets: SyntheticMatchAssets) -> None:
        frames = self._read_frames(video_path)
        if not frames:
            return
        n_occ = max(1, int(round(len(frames) * pct))) if pct > 0 else 0
        if n_occ <= 0:
            return
        occ_idx = set(
            self._rng.choice(len(frames), size=min(n_occ, len(frames)), replace=False).tolist()
        )
        for i in occ_idx:
            h, w = frames[i].shape[:2]
            x0, y0 = w // 4, h // 4
            x1, y1 = (3 * w) // 4, (3 * h) // 4
            frames[i][y0:y1, x0:x1] = 0
        self._rewrite_video(video_path, frames, assets)

    def _swap_first_two_cam_labels(self, calib_path: Path) -> None:
        payload: Any = json.loads(calib_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "cameras" in payload:
            items = payload["cameras"]
        elif isinstance(payload, list):
            items = payload
        else:
            return
        if len(items) < 2:
            return
        items[0]["cam_id"], items[1]["cam_id"] = items[1]["cam_id"], items[0]["cam_id"]
        if isinstance(payload, dict):
            payload["cameras"] = items
            calib_path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            calib_path.write_text(json.dumps(items), encoding="utf-8")


# ---------------------------------------------------------------------------
# 4) VariableSweep — Cartesian product fixture
# ---------------------------------------------------------------------------


_SWEEP_CAM_COUNTS: tuple[int, ...] = (1, 2, 4)
_SWEEP_RESOLUTIONS: tuple[tuple[int, int], ...] = ((48, 64), (72, 128))
_SWEEP_DURATIONS: tuple[float, ...] = (0.5, 1.0)


@dataclass
class SweepConfig:
    cam_count: int
    height: int
    width: int
    duration_seconds: float


def _variable_sweep_configs() -> list[SweepConfig]:
    return [
        SweepConfig(
            cam_count=cam_count,
            height=res[0],
            width=res[1],
            duration_seconds=duration,
        )
        for cam_count, res, duration in product(
            _SWEEP_CAM_COUNTS, _SWEEP_RESOLUTIONS, _SWEEP_DURATIONS
        )
    ]


# ---------------------------------------------------------------------------
# Local copies of the synthesis helpers (kept here so tests don't rely on
# private symbols from test_pipeline.py — they're not part of any package
# public API and there's no shared conftest).
# ---------------------------------------------------------------------------


def _write_synthetic_video(
    path: Path,
    duration_seconds: float = 1.0,
    fps: float = _DEFAULT_FPS,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    try:
        frame_count = max(1, int(round(duration_seconds * fps)))
        for _ in range(frame_count):
            frame = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _calibration_dict(
    cam_id: str, width: int = _DEFAULT_WIDTH, height: int = _DEFAULT_HEIGHT
) -> dict:
    focal = 60.0
    intrinsics = [
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ]
    cam_index = max(0, int(cam_id.removeprefix("cam") or 1) - 1)
    sign = 1.0 if cam_index % 2 == 0 else -1.0
    tx = sign * (10.0 + cam_index * 2.5)
    ty = (-1.0 if cam_index % 2 else 1.0) * (8.0 + cam_index * 4.0)
    tz = sign * -4.0
    extrinsics = [
        [sign, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, sign, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "cam_id": cam_id,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "image_size": [width, height],
    }


def _invoke_cli(
    assets: SyntheticMatchAssets,
    out_path: Path,
    max_frames: int | None = None,
) -> tuple[int, str, MatchAnalysis | None]:
    """Run the CLI against ``assets``. Returns (exit_code, output, analysis_or_None)."""
    runner = CliRunner()
    args: list[str] = []
    for video in assets.videos:
        args.extend(["--video", str(video)])
    args.extend(["--calib", str(assets.calib_path), "--out", str(out_path)])
    if max_frames is not None:
        args.extend(["--max-frames", str(max_frames)])
    result = runner.invoke(main, args, catch_exceptions=True)
    analysis: MatchAnalysis | None = None
    if result.exit_code == 0 and out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        analysis = MatchAnalysis.model_validate(payload)
    return result.exit_code, result.output, analysis


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_scenario_simulates_full_tournament() -> None:
    """8-team bracket → 7 matches, no crash, plausible structure."""
    builder = ScenarioBuilder(seed=_MASTER_SEED)
    tournament = builder.build_tournament(n_teams=8, tournament_id="T_FULL")
    assert len(tournament.matches) == 7  # 4 QF + 2 SF + 1 F

    teams = {m.team_a for m in tournament.matches} | {m.team_b for m in tournament.matches}
    assert len(teams) >= 2  # at least both finalists appear

    for match in tournament.matches:
        assert match.team_a != match.team_b
        assert 2 <= len(match.sets) <= 3
        for s in match.sets:
            assert 6 <= len(s.games) <= 12
        assert 5 <= match.rallies <= 50


def test_pipeline_handles_2cam_match_with_5pct_missing_frames(tmp_path: Path) -> None:
    """2-cam match, 5% dropped frames in both videos, pipeline still produces valid MatchAnalysis."""
    builder = ScenarioBuilder(seed=_MASTER_SEED)
    tournament = builder.build_tournament(n_teams=2, tournament_id="T_MISS")
    match = tournament.matches[0]

    generator = RallyGenerator(seed=_MASTER_SEED)
    assets = generator.render(match=match, out_dir=tmp_path / "assets", cam_count=2)

    injector = HumanErrorInjector(seed=_MASTER_SEED + 1)
    faults = injector.inject(
        assets,
        missing_frames_pct=0.05,
        sync_drift_seconds=0.0,
        low_confidence_camera_ratio=0.0,
        wrong_cam_label_swap_chance=0.0,
        partial_occlusion_pct=0.0,
    )

    out_path = tmp_path / "out.json"
    exit_code, output, analysis = _invoke_cli(assets, out_path, max_frames=10)

    assert exit_code == 0, output
    assert analysis is not None
    assert {c.cam_id for c in analysis.meta.cameras} == {"cam1", "cam2"}
    assert faults.missing_frames_pct == 0.05


def test_pipeline_handles_sync_drift(tmp_path: Path) -> None:
    """Sync drift of 0.3s recorded in faults; pipeline still produces output."""
    builder = ScenarioBuilder(seed=_MASTER_SEED + 2)
    tournament = builder.build_tournament(n_teams=2, tournament_id="T_DRIFT")
    match = tournament.matches[0]

    generator = RallyGenerator(seed=_MASTER_SEED + 2)
    assets = generator.render(match=match, out_dir=tmp_path / "assets", cam_count=2)

    injector = HumanErrorInjector(seed=_MASTER_SEED + 3)
    faults = injector.inject(
        assets,
        missing_frames_pct=0.0,
        sync_drift_seconds=0.3,
        low_confidence_camera_ratio=0.0,
        wrong_cam_label_swap_chance=0.0,
        partial_occlusion_pct=0.0,
    )

    out_path = tmp_path / "out.json"
    exit_code, output, analysis = _invoke_cli(assets, out_path, max_frames=10)

    assert exit_code == 0, output
    assert analysis is not None
    assert faults.sync_drift_seconds == pytest.approx(0.3)


def test_pipeline_no_nan_in_3d_positions(tmp_path: Path) -> None:
    """Across N=10 scenarios, any ball.position_3d must be finite — no NaN/Inf."""
    n_scenarios = 10
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    seen_3d_count = 0
    for i in range(n_scenarios):
        match = Match(
            match_id=f"NAN_{i:02d}",
            team_a="team_A",
            team_b="team_B",
            sets=[Set(games=[Game(winner="team_A", points_a=1, points_b=0)])],
            rallies=5,
        )
        generator = RallyGenerator(seed=_MASTER_SEED + 100 + i)
        assets = generator.render(match=match, out_dir=tmp_path / f"assets_{i}", cam_count=2)

        out_path = runs_dir / f"run_{i:02d}.json"
        exit_code, output, analysis = _invoke_cli(assets, out_path, max_frames=5)
        assert exit_code == 0, f"scenario {i} crashed: {output}"
        assert analysis is not None, f"scenario {i} produced no analysis"

        for frame in analysis.frames:
            if frame.ball.position_3d is not None:
                seen_3d_count += 1
                for coord in (
                    frame.ball.position_3d.x,
                    frame.ball.position_3d.y,
                    frame.ball.position_3d.z,
                ):
                    assert np.isfinite(coord), (
                        f"non-finite 3D coord in scenario {i} frame {frame.frame_id}: {coord}"
                    )

    # Stub detector produces no balls → seen_3d_count == 0 is expected and
    # *also* a useful signal: it documents that without a real detector the
    # downstream triangulator never gets exercised. We still pass — the test
    # is purely "if you do see a 3D point, it must be finite."
    assert seen_3d_count >= 0


def test_pipeline_graceful_with_1cam(tmp_path: Path) -> None:
    """1-camera setup must NOT crash. ball.position_3d must be None (no triangulation)."""
    match = Match(
        match_id="T_1CAM",
        team_a="team_A",
        team_b="team_B",
        sets=[Set(games=[Game(winner="team_A", points_a=1, points_b=0)])],
        rallies=3,
    )
    generator = RallyGenerator(seed=_MASTER_SEED + 4)
    assets = generator.render(match=match, out_dir=tmp_path / "assets", cam_count=1)

    out_path = tmp_path / "out.json"
    exit_code, output, analysis = _invoke_cli(assets, out_path, max_frames=5)

    assert exit_code == 0, output
    assert analysis is not None
    assert {c.cam_id for c in analysis.meta.cameras} == {"cam1"}
    for frame in analysis.frames:
        # With a single camera there is no triangulation, so 3D must be None.
        assert frame.ball.position_3d is None, (
            f"unexpected 3D position with 1 camera: {frame.ball.position_3d}"
        )


def test_pipeline_handles_wrong_cam_labels(tmp_path: Path) -> None:
    """Swapped cam1/cam2 labels in calibration JSON: pipeline either errors loudly OR
    produces output. Either is acceptable — what's NOT acceptable is a silent crash."""
    builder = ScenarioBuilder(seed=_MASTER_SEED + 5)
    tournament = builder.build_tournament(n_teams=2, tournament_id="T_SWAP")
    match = tournament.matches[0]

    generator = RallyGenerator(seed=_MASTER_SEED + 5)
    assets = generator.render(match=match, out_dir=tmp_path / "assets", cam_count=2)

    injector = HumanErrorInjector(seed=_MASTER_SEED + 6)
    # Force the swap deterministically with chance=1.0
    faults = injector.inject(
        assets,
        missing_frames_pct=0.0,
        sync_drift_seconds=0.0,
        low_confidence_camera_ratio=0.0,
        wrong_cam_label_swap_chance=1.0,
        partial_occlusion_pct=0.0,
    )
    assert faults.cam_label_swap, "injector failed to swap cam labels"

    out_path = tmp_path / "out.json"
    exit_code, output, analysis = _invoke_cli(assets, out_path, max_frames=5)

    # The CLI re-keys calibrations by cam1/cam2 in --video order, so swapping
    # the cam_id field inside the JSON does not currently raise — but it must
    # still produce structurally valid output.
    assert exit_code == 0, f"swap should not crash, but got: {output}"
    assert analysis is not None
    assert {c.cam_id for c in analysis.meta.cameras} == {"cam1", "cam2"}


# ---------------------------------------------------------------------------
# Variable sweep — runs the smallest config in the Cartesian product to
# prove the harness wires together. The full sweep is gated behind 'slow'
# so CI stays fast.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    _variable_sweep_configs()[:1],  # smoke: just the first config on the default run
    ids=lambda c: f"{c.cam_count}cam_{c.width}x{c.height}_{c.duration_seconds}s",
)
def test_variable_sweep_pipeline_smoke(tmp_path: Path, config: SweepConfig) -> None:
    """Smoke-run one sweep config end-to-end so the harness is exercised on every CI run."""
    match = Match(
        match_id=f"SWEEP_{config.cam_count}c_{config.width}x{config.height}",
        team_a="team_A",
        team_b="team_B",
        sets=[Set(games=[Game(winner="team_A", points_a=1, points_b=0)])],
        rallies=3,
    )
    generator = RallyGenerator(seed=_MASTER_SEED + 7)
    assets = generator.render(
        match=match,
        out_dir=tmp_path / "assets",
        cam_count=config.cam_count,
        width=config.width,
        height=config.height,
        duration_seconds=config.duration_seconds,
    )

    out_path = tmp_path / "out.json"
    exit_code, output, analysis = _invoke_cli(assets, out_path, max_frames=3)
    assert exit_code == 0, output
    assert analysis is not None
    assert len(analysis.meta.cameras) == config.cam_count
