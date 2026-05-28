"""Deterministic scenario seed generator for the padelgraph-ai stress harness.

Emits N seed dictionaries as JSON Lines (one JSON object per line). Each seed
captures the *minimum* state needed for an Ollama Cloud worker to expand into
a full padel match scenario spec downstream.

Determinism: a fixed --seed flag (default 42) guarantees byte-identical output
across runs. This is essential for reproducing stress test failures.

Usage:
    uv run python scripts/scenario_seed.py --n 50 --out simulations/seeds.jsonl

Apache 2.0.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np


def _generate_seed(rng: np.random.Generator, seed_id: int) -> dict:
    """Build one seed dict. All values bounded to reasonable padel ranges."""
    # n_players: 4 (doubles, canonical) or 2 (singles, rare)
    n_players = int(rng.choice([2, 4], p=[0.1, 0.9]))

    # court_size: padel court is 20m × 10m (canonical). Allow ±5% drift for fuzz.
    court_w = float(20.0 + rng.uniform(-1.0, 1.0))
    court_h = float(10.0 + rng.uniform(-0.5, 0.5))

    # fps: realistic camera capture rates
    fps = float(rng.choice([24.0, 25.0, 30.0, 50.0, 60.0]))

    # sync_offset: seconds between cameras (0 = perfect sync, up to 0.5s drift)
    sync_offset = float(rng.uniform(0.0, 0.5))

    # error_injection_pct: fraction of frames to corrupt (0% clean, up to 15% noisy)
    error_injection_pct = float(rng.uniform(0.0, 0.15))

    return {
        "seed_id": f"{seed_id:04d}",
        "n_players": n_players,
        "court_size": [round(court_w, 2), round(court_h, 2)],
        "fps": fps,
        "sync_offset": round(sync_offset, 3),
        "error_injection_pct": round(error_injection_pct, 3),
    }


@click.command()
@click.option(
    "--n",
    "n",
    type=click.IntRange(min=1, max=10000),
    default=50,
    help="Number of seeds to generate (default: 50)",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to write JSONL output (one seed per line)",
)
@click.option(
    "--seed",
    "rng_seed",
    type=int,
    default=42,
    help="Deterministic RNG seed for reproducibility (default: 42)",
)
def main(n: int, out_path: Path, rng_seed: int) -> None:
    """Generate N deterministic padel scenario seeds as JSON Lines."""
    rng = np.random.default_rng(rng_seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(n):
            seed = _generate_seed(rng, i)
            fh.write(json.dumps(seed) + "\n")

    click.echo(f"wrote {n} seeds to {out_path} (rng_seed={rng_seed})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        click.echo(f"scenario_seed.py error: {exc}", err=True)
        sys.exit(1)
