#!/usr/bin/env bash
# stress_test.sh — Ollama Cloud parallel stress harness for padelgraph-ai
#
# Pattern: 4-phase pipeline (seed → scenario → pipeline → validate).
# Stages 1+2 are xargs -P parallel. Stage 3 is sequential.
#
# Usage:
#   bash scripts/stress_test.sh [N] [MAX_PARALLEL] [OLLAMA_MODEL] [--dry-run]
#
# Defaults: N=50, MAX_PARALLEL=8, OLLAMA_MODEL=devstral-2:cloud
#
# --dry-run: skip real Ollama calls, fake scenario JSON via internal helper.
#            For smoke tests. Does NOT consume Ollama Cloud API budget.
#
# Outputs:
#   simulations/seeds.jsonl
#   simulations/seed_<seed_id>.json   (N per-seed files for parallel workers)
#   simulations/sim_<seed_id>.json    (N expanded scenarios)
#   runs/run_<seed_id>.json           (N pipeline outputs)
#   artifacts/stress-test-<timestamp>.md (one report)
#
# Exit codes:
#   0 — all phases completed (anomalies may exist; check report)
#   1 — phase 1 (seed gen) failed
#   2 — phase 2 (Ollama scenario expansion) failed catastrophically
#   3 — phase 3 (pipeline runs) failed catastrophically
#   4 — phase 4 (validation report) failed

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────
N="${1:-50}"
MAX_PARALLEL="${2:-8}"
OLLAMA_MODEL="${3:-devstral-2:cloud}"
DRY_RUN=false

# Allow --dry-run anywhere in args
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN=true
    fi
done

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_DIR="${REPO_ROOT}/simulations"
RUN_DIR="${REPO_ROOT}/runs"
ART_DIR="${REPO_ROOT}/artifacts"
SCRIPT_DIR="${REPO_ROOT}/scripts"
WORKER_HELPER="${SCRIPT_DIR}/_stress_worker.py"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
REPORT_PATH="${ART_DIR}/stress-test-${TIMESTAMP}.md"

mkdir -p "$SIM_DIR" "$RUN_DIR" "$ART_DIR"

# ── Colors (TTY only) ─────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_GREEN='\033[0;32m'
    C_RED='\033[0;31m'
    C_YELLOW='\033[1;33m'
    C_CYAN='\033[0;36m'
    C_RESET='\033[0m'
else
    C_GREEN=''
    C_RED=''
    C_YELLOW=''
    C_CYAN=''
    C_RESET=''
fi

log() {
    printf '%b[stress-test]%b %s\n' "$C_CYAN" "$C_RESET" "$*"
}

err() {
    printf '%b[stress-test ERROR]%b %s\n' "$C_RED" "$C_RESET" "$*" >&2
}

ok() {
    printf '%b[stress-test OK]%b %s\n' "$C_GREEN" "$C_RESET" "$*"
}

warn() {
    printf '%b[stress-test WARN]%b %s\n' "$C_YELLOW" "$C_RESET" "$*"
}

# ── Banner ────────────────────────────────────────────────────────────────────
log "padelgraph-ai stress harness"
log "  N=$N  MAX_PARALLEL=$MAX_PARALLEL  MODEL=$OLLAMA_MODEL  DRY_RUN=$DRY_RUN"
log "  repo: $REPO_ROOT"
log "  report: $REPORT_PATH"

# ── Phase 1: Seed generation ──────────────────────────────────────────────────
log "Phase 1/4 — generating $N seeds"
SEEDS_PATH="${SIM_DIR}/seeds.jsonl"

if ! uv run python "${SCRIPT_DIR}/scenario_seed.py" --n "$N" --out "$SEEDS_PATH"; then
    err "phase 1 failed: scenario_seed.py exited non-zero"
    exit 1
fi

SEED_COUNT="$(wc -l < "$SEEDS_PATH" | tr -d ' ')"
if [[ "$SEED_COUNT" -ne "$N" ]]; then
    err "phase 1 produced $SEED_COUNT seeds but $N requested"
    exit 1
fi

# Split JSONL into per-seed files so xargs has clean filename arguments.
# This avoids shell-quoting issues with JSON content passed as xargs args.
if ! uv run python "${WORKER_HELPER}" split-seeds \
    --seeds "$SEEDS_PATH" \
    --out-dir "$SIM_DIR"; then
    err "phase 1 failed: could not split seeds.jsonl into per-seed files"
    exit 1
fi

SEED_FILE_COUNT="$(find "$SIM_DIR" -name "seed_*.json" -type f | wc -l | tr -d ' ')"
if [[ "$SEED_FILE_COUNT" -ne "$N" ]]; then
    err "phase 1 produced $SEED_FILE_COUNT per-seed files but $N requested"
    exit 1
fi
ok "phase 1 complete — $SEED_COUNT seeds at $SEEDS_PATH + $SEED_FILE_COUNT per-seed files"

# ── Phase 2: Ollama Cloud scenario expansion (parallel) ──────────────────────
log "Phase 2/4 — expanding seeds into scenarios via $OLLAMA_MODEL (parallel=$MAX_PARALLEL)"

# Pre-flight: real Ollama needs OLLAMA_API_KEY + ollama CLI
if [[ "$DRY_RUN" == "false" ]]; then
    if ! command -v ollama >/dev/null 2>&1; then
        err "phase 2 pre-flight: ollama CLI not found in PATH"
        exit 2
    fi
    if [[ -z "${OLLAMA_API_KEY:-}" ]]; then
        # Try sourcing credentials file
        if [[ -f "$HOME/.freejack-credentials.env" ]]; then
            # shellcheck disable=SC1091
            set +u
            # shellcheck source=/dev/null
            source "$HOME/.freejack-credentials.env"
            set -u
        fi
        if [[ -z "${OLLAMA_API_KEY:-}" ]]; then
            err "phase 2 pre-flight: OLLAMA_API_KEY not set (and not in ~/.freejack-credentials.env)"
            exit 2
        fi
    fi
fi

# Worker function: takes one seed file path, produces one sim file.
# Uses the Python worker helper for JSON manipulation (no jq/shell quoting hazards).
expand_seed() {
    local seed_path="$1"
    local dry_run="$2"
    local model="$3"
    local sim_dir="$4"
    local worker_helper="$5"

    if [[ "$dry_run" == "true" ]]; then
        # Dry run: helper synthesizes a fake scenario JSON locally.
        uv run python "$worker_helper" fake-expand \
            --seed "$seed_path" \
            --out-dir "$sim_dir" 2>/dev/null || return 1
    else
        # Real run: call ollama, capture output, then helper wraps the result
        # into a valid scenario file (with error fallback if ollama failed).
        local seed_id
        seed_id="$(uv run python "$worker_helper" read-seed-id --seed "$seed_path" 2>/dev/null)" || return 1
        local out_path="${sim_dir}/sim_${seed_id}.json"
        local prompt_text
        prompt_text="$(uv run python "$worker_helper" build-prompt --seed "$seed_path" 2>/dev/null)" || return 1

        # Capture raw output (or empty on failure)
        local raw_output=""
        if raw_output="$(ollama run "$model" "$prompt_text" 2>/dev/null)"; then
            # Wrap & validate the raw response
            uv run python "$worker_helper" wrap-ollama-output \
                --seed "$seed_path" \
                --raw "$raw_output" \
                --model "$model" \
                --out "$out_path" 2>/dev/null || true
        else
            # Ollama call itself failed: write error marker
            uv run python "$worker_helper" write-error-marker \
                --seed "$seed_path" \
                --error "ollama_call_failed" \
                --model "$model" \
                --out "$out_path" 2>/dev/null || true
        fi
    fi
}

export -f expand_seed

# xargs -P fans out workers. -n 1 = one seed file per worker invocation.
if ! find "$SIM_DIR" -name "seed_*.json" -type f -print0 \
    | xargs -0 -I {} -P "$MAX_PARALLEL" \
        bash -c 'expand_seed "$@"' _ {} "$DRY_RUN" "$OLLAMA_MODEL" "$SIM_DIR" "$WORKER_HELPER"; then
    err "phase 2 xargs failed"
    exit 2
fi

SIM_COUNT="$(find "$SIM_DIR" -name "sim_*.json" -type f | wc -l | tr -d ' ')"
ok "phase 2 complete — $SIM_COUNT scenarios at $SIM_DIR"

# ── Phase 3: Pipeline runs (parallel, local) ─────────────────────────────────
log "Phase 3/4 — running pipeline against $SIM_COUNT scenarios (parallel=4)"

# For this harness, the pipeline run is a metadata-only invocation:
# we write a MatchAnalysis-shaped JSON per sim. Running the real CLI with
# synthesized video + calibration on every iteration doubles wall clock and
# isn't the gap this harness measures — phase 4 validates the JSON shape.
#
# When Alan / Sephirot wants to wire the real pipeline:
#   change the helper's `fake-pipeline-run` subcommand to invoke
#   `uv run padelgraph-ai-infer ...` driven by synthetic video gen
#   from tests/test_pipeline.py:_write_synthetic_video

pipeline_run() {
    local sim_path="$1"
    local run_dir="$2"
    local worker_helper="$3"

    uv run python "$worker_helper" fake-pipeline-run \
        --sim "$sim_path" \
        --out-dir "$run_dir" 2>/dev/null || return 1
}

export -f pipeline_run

if ! find "$SIM_DIR" -name "sim_*.json" -type f -print0 \
    | xargs -0 -I {} -P 4 \
        bash -c 'pipeline_run "$@"' _ {} "$RUN_DIR" "$WORKER_HELPER"; then
    err "phase 3 xargs failed"
    exit 3
fi

RUN_COUNT="$(find "$RUN_DIR" -name "run_*.json" -type f | wc -l | tr -d ' ')"
ok "phase 3 complete — $RUN_COUNT runs at $RUN_DIR"

# ── Phase 4: Validation report ───────────────────────────────────────────────
log "Phase 4/4 — validating runs + writing report"

# validate_runs.py exits non-zero if critical anomalies found, but we still
# consider the harness "successful" if a report was written. The critical
# exit is informational — the report path is the primary output.
if uv run python "${SCRIPT_DIR}/validate_runs.py" \
    --runs-dir "$RUN_DIR" \
    --report "$REPORT_PATH"; then
    ok "phase 4 complete — report at $REPORT_PATH (no critical anomalies)"
else
    rc=$?
    if [[ ! -f "$REPORT_PATH" ]]; then
        err "phase 4 failed: validate_runs.py exited $rc and produced no report"
        exit 4
    fi
    warn "phase 4 complete — report at $REPORT_PATH (critical anomalies detected, exit $rc)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
log "Summary:"
log "  seeds:   $SEED_COUNT"
log "  sims:    $SIM_COUNT"
log "  runs:    $RUN_COUNT"
log "  report:  $REPORT_PATH"

if [[ "$DRY_RUN" == "true" ]]; then
    warn "DRY-RUN mode — no Ollama Cloud API calls made"
fi

ok "stress test complete"
exit 0
