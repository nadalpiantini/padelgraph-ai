"""Internal helper for stress_test.sh — JSON manipulation per worker.

Encapsulates every JSON read/write the bash orchestrator needs, so the shell
script never has to embed multi-line Python or struggle with quote escaping.

Subcommands:
    split-seeds         — explode seeds.jsonl into per-seed JSON files
    read-seed-id        — print the seed_id of one seed file to stdout
    build-prompt        — print the Ollama expansion prompt to stdout
    fake-expand         — synthesize a fake scenario JSON (dry-run mode)
    wrap-ollama-output  — wrap raw Ollama stdout into a validated sim file
    write-error-marker  — write a sim file marking an upstream Ollama failure
    fake-pipeline-run   — synthesize a fake MatchAnalysis-shaped run output

Apache 2.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

# Single source of truth for the expansion prompt.
_EXPAND_PROMPT_TEMPLATE = (
    "Expand this padel match scenario seed into a richer JSON spec. "
    "Output ONLY valid JSON, no commentary, no markdown fences. "
    "Include: seed_id (echo back), match_id (e.g. 'sim-N'), n_players, "
    "n_cameras (2-4), duration_seconds (1.0-3.0 for synth test), fps, "
    "sync_offset, error_injection_pct, court_size, "
    "shot_types (list of strings: volea, derecha, reves, smash, bandeja, "
    "vibora, globo). Seed JSON: {seed_json}"
)


def _load_seed(seed_path: Path) -> dict:
    return json.loads(seed_path.read_text(encoding="utf-8"))


def _build_scenario_from_seed(seed: dict, *, dry_run_marker: bool = False) -> dict:
    """Synthesize a deterministic scenario dict from a seed.

    Used both for dry-run mode and as the fallback shape when an Ollama
    response is unparseable.
    """
    scenario = {
        "seed_id": seed["seed_id"],
        "match_id": f"sim-{seed['seed_id']}",
        "n_players": seed.get("n_players", 4),
        "n_cameras": min(2, seed.get("n_players", 4)),
        "duration_seconds": 1.0,
        "fps": seed.get("fps", 30.0),
        "sync_offset": seed.get("sync_offset", 0.0),
        "error_injection_pct": seed.get("error_injection_pct", 0.0),
        "court_size": seed.get("court_size", [20.0, 10.0]),
        "shot_types": ["volea", "derecha", "globo"],
    }
    if dry_run_marker:
        scenario["_dry_run"] = True
    return scenario


@click.group()
def cli() -> None:
    """Internal stress harness helper. Used by scripts/stress_test.sh."""


@cli.command("split-seeds")
@click.option(
    "--seeds",
    "seeds_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-dir", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
def split_seeds(seeds_path: Path, out_dir: Path) -> None:
    """Split a JSONL seed file into one JSON file per seed (seed_<id>.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for line in seeds_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        seed = json.loads(line)
        seed_id = seed["seed_id"]
        target = out_dir / f"seed_{seed_id}.json"
        target.write_text(json.dumps(seed, indent=2) + "\n", encoding="utf-8")
        count += 1
    click.echo(f"split {count} seeds into {out_dir}")


@cli.command("read-seed-id")
@click.option(
    "--seed",
    "seed_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def read_seed_id(seed_path: Path) -> None:
    """Print the seed_id of a seed file to stdout."""
    seed = _load_seed(seed_path)
    click.echo(seed["seed_id"], nl=False)


@cli.command("build-prompt")
@click.option(
    "--seed",
    "seed_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def build_prompt(seed_path: Path) -> None:
    """Build the Ollama expansion prompt for a seed; print to stdout."""
    seed = _load_seed(seed_path)
    prompt = _EXPAND_PROMPT_TEMPLATE.format(seed_json=json.dumps(seed))
    click.echo(prompt, nl=False)


@cli.command("fake-expand")
@click.option(
    "--seed",
    "seed_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--out-dir", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
def fake_expand(seed_path: Path, out_dir: Path) -> None:
    """Synthesize a fake scenario file (dry-run mode, no Ollama call)."""
    seed = _load_seed(seed_path)
    scenario = _build_scenario_from_seed(seed, dry_run_marker=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"sim_{seed['seed_id']}.json"
    target.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")


@cli.command("wrap-ollama-output")
@click.option(
    "--seed",
    "seed_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--raw", "raw_output", type=str, required=True)
@click.option("--model", "model", type=str, required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def wrap_ollama_output(seed_path: Path, raw_output: str, model: str, out_path: Path) -> None:
    """Parse Ollama's raw response; if invalid JSON, write a fallback scenario marked as parse error."""
    seed = _load_seed(seed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = raw_output.strip()
    # Strip markdown fences if model added them despite the prompt
    if cleaned.startswith("```"):
        # Drop the first line and trailing ``` if present
        lines = cleaned.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        # Ensure seed_id and model are stamped
        parsed.setdefault("seed_id", seed["seed_id"])
        parsed["_model"] = model
        out_path.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
    except json.JSONDecodeError as exc:
        # Fallback: synthesize a scenario from the seed + mark parse error
        scenario = _build_scenario_from_seed(seed)
        scenario["_error"] = f"ollama_returned_invalid_json: {exc}"
        scenario["_model"] = model
        out_path.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")


@cli.command("write-error-marker")
@click.option(
    "--seed",
    "seed_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--error", "error", type=str, required=True)
@click.option("--model", "model", type=str, required=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), required=True)
def write_error_marker(seed_path: Path, error: str, model: str, out_path: Path) -> None:
    """Write a sim file marking an upstream Ollama failure."""
    seed = _load_seed(seed_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed_id": seed["seed_id"],
        "_error": error,
        "_model": model,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@cli.command("fake-pipeline-run")
@click.option(
    "--sim", "sim_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--out-dir", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True
)
def fake_pipeline_run(sim_path: Path, out_dir: Path) -> None:
    """Synthesize a MatchAnalysis-shaped JSON pipeline output for one scenario."""
    try:
        sim = json.loads(sim_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Cannot read sim — emit a crash marker tagged by sim filename
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_guess = sim_path.stem.removeprefix("sim_") or "unknown"
        crash_path = out_dir / f"run_{seed_guess}.json"
        crash_path.write_text(
            json.dumps({"_error": f"pipeline_could_not_read_sim: {exc}"}) + "\n",
            encoding="utf-8",
        )
        return

    seed_id = sim.get("seed_id") or sim_path.stem.removeprefix("sim_") or "unknown"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"run_{seed_id}.json"

    # Propagate upstream errors so the validator can attribute them
    if "_error" in sim:
        out_path.write_text(
            json.dumps(
                {
                    "_error": f"upstream_sim_error: {sim['_error']}",
                    "seed_id": seed_id,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return

    # Build a minimal valid MatchAnalysis envelope
    n_cameras = sim.get("n_cameras", 2)
    if not isinstance(n_cameras, int) or n_cameras < 1:
        n_cameras = 2

    match_analysis = {
        "meta": {
            "match_id": sim.get("match_id", f"sim-{seed_id}"),
            "cameras": [
                {
                    "cam_id": f"cam{i + 1}",
                    "intrinsics": [[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]],
                    "extrinsics": [
                        [1.0, 0.0, 0.0, -10.0],
                        [0.0, 1.0, 0.0, 8.0],
                        [0.0, 0.0, 1.0, -4.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "image_size": [64, 48],
                }
                for i in range(n_cameras)
            ],
            "fps": sim.get("fps", 30.0),
        },
        "frames": [
            {
                "frame_id": 0,
                "ts": 0.0,
                "players_per_cam": {},
                "ball": {
                    "position_2d_per_cam": {},
                    "position_3d": None,
                    "confidence": 0.0,
                },
            }
        ],
    }
    out_path.write_text(json.dumps(match_analysis, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    cli()
