"""run.py — benchmark dispatch via SSE + thought inspector"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import AsyncIterator

import dataclasses

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from rbac_benchmark.core.config import BenchmarkConfig
from rbac_benchmark.paths import data_path

router = APIRouter()

_RESULTS_FILE = data_path("benchmark_results.json")

# Run registry. Each entry:
#   {"cfg_path": str, "proc": Process | None, "status": "running"|"done"|"error",
#    "lines": list[str], "cond": asyncio.Condition}
#
# The master subprocess is spawned by /start and pumped into `lines` by a
# background task — NOT inside the SSE generator. An SSE client is therefore
# just a resumable view onto the buffer: if the stream drops mid-run, the
# benchmark keeps going, the browser reconnects with Last-Event-ID, and no
# second process is ever spawned for the same run.
_RUNS: dict[str, dict] = {}


def _active_run_id() -> str | None:
    """Returns the id of a run whose master process is still alive, if any."""
    for rid, entry in _RUNS.items():
        proc = entry.get("proc")
        if proc is not None and proc.returncode is None:
            return rid
    return None


async def _pump_process_output(entry: dict) -> None:
    """Background task: drains the master's stdout into the run's line buffer.

    Owns the run's lifecycle end-to-end — when stdout closes it awaits the exit
    code, flips `status`, removes the temp config, and wakes every SSE client
    waiting on the condition so they can emit __DONE__/__ERROR__ and finish.
    """
    proc = entry["proc"]
    cond: asyncio.Condition = entry["cond"]
    try:
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            async with cond:
                entry["lines"].append(line)
                cond.notify_all()
    finally:
        await proc.wait()
        entry["status"] = "done" if proc.returncode == 0 else "error"
        try:
            Path(entry["cfg_path"]).unlink(missing_ok=True)
        except Exception:
            pass
        async with cond:
            cond.notify_all()


# ── /api/run/thoughts ─────────────────────────────────────────────────────────
@router.get("/thoughts")
def get_thoughts():
    """Return stored [THOUGHT] traces from the last benchmark run."""
    p = Path(_RESULTS_FILE)
    if not p.exists():
        return {"thoughts": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"thoughts": []}

    thoughts = []
    for key, metrics in data.items():
        parts = key.split(" | ")
        if len(parts) != 3:
            continue
        model, defense, _attack = parts
        texts  = metrics.get("raw_texts", [])
        aware  = metrics.get("judge_awareness_labels", [])
        levers = metrics.get("judge_lever_labels", [])
        for i, text in enumerate(texts):
            if not text or text == "[NO TEXT GENERATED - SILENT EXECUTION]":
                continue
            thoughts.append({
                "model":    model,
                "defense":  defense,
                "text":     text,
                "awareness": aware[i]  if i < len(aware)  else None,
                "lever":    levers[i]  if i < len(levers) else None,
            })
    return {"thoughts": thoughts}


# ── /api/run/start ────────────────────────────────────────────────────────────
@router.post("/start")
async def start_run(body: dict):
    """Write a temp config JSON and spawn rbac-master as a subprocess."""
    models = body.get("models", [])
    if not models:
        raise HTTPException(400, "models list required")

    # Refuse to start while another master is alive. Two concurrent runs would
    # double the inference load on every worker and interleave 30-second
    # checkpoints into the same benchmark_results.json, corrupting it.
    active = _active_run_id()
    if active:
        raise HTTPException(409, f"A benchmark run is already active ({active}). Abort it first.")

    cfg = BenchmarkConfig(
        models=models,
        iterations=int(body.get("iterations", 3)),
        max_turns=int(body.get("max_turns", 2)),
        max_retries=int(body.get("max_retries", 2)),
        exclude_master=bool(body.get("exclude_master", False)),
        use_custom_prompts=bool(body.get("use_custom_prompts", True)),
        use_generated_injections=bool(body.get("use_generated_injections", True)),
        use_generated_defenses=bool(body.get("use_generated_defenses", True)),
        ollama_port=int(body.get("ollama_port", 11434)),
        udp_discovery_port=int(body.get("udp_discovery_port", 5005)),
        timeout=float(body.get("timeout", 3.0)),
        request_timeout=float(body.get("request_timeout", 300.0)),
        use_judge=bool(body.get("use_judge", False)),
        judge_model=body.get("judge_model", "qwen3.5:9b"),
        ref_model=body.get("ref_model", models[0]),
        attack_validity_threshold=float(body.get("attack_validity_threshold", 0.1)),
        kappa_samples_per_category=int(body.get("kappa_samples_per_category", 20)),
        kappa_seed=int(body.get("kappa_seed", 42)),
        show_thoughts=bool(body.get("show_thoughts", False)),
        # Persist results to the same file the dashboard / thoughts endpoints read.
        # Without this, config.output is None and master_node skips both the periodic
        # checkpoint and the final save, so a UI-triggered run leaves the dashboard stale.
        output=str(_RESULTS_FILE),
    )

    run_id = str(uuid.uuid4())
    cfg_path = Path(data_path(f"run_config_{run_id}.json"))
    cfg_path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8")

    # Drop finished runs so the registry doesn't grow unbounded — only the run
    # being started (and any client still replaying its predecessor's final
    # lines) matters, and those clients terminate on the buffered __DONE__.
    for rid in [rid for rid, e in _RUNS.items() if e.get("proc") is None or e["proc"].returncode is not None]:
        _RUNS.pop(rid, None)

    entry = {
        "cfg_path": str(cfg_path),
        "proc": None,
        "status": "running",
        "lines": [],
        "cond": asyncio.Condition(),
    }
    _RUNS[run_id] = entry

    # Spawn the master here, decoupled from any SSE connection: the run's
    # lifetime must not depend on a browser tab staying connected.
    master_module = "rbac_benchmark.orchestration.master_node"
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", master_module, "--config", str(cfg_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    entry["proc"] = proc
    entry["pump_task"] = asyncio.create_task(_pump_process_output(entry))

    return {"run_id": run_id, "total_inferences": 0}


# ── /api/run/abort/{run_id} ───────────────────────────────────────────────────
@router.post("/abort/{run_id}")
async def abort_run(run_id: str):
    entry = _RUNS.get(run_id)
    if entry and entry.get("proc"):
        try:
            entry["proc"].terminate()
        except Exception:
            pass
    return {"ok": True}


# ── /api/run/stream/{run_id} ──────────────────────────────────────────────────
@router.get("/stream/{run_id}")
async def stream_run(run_id: str, request: Request):
    """SSE endpoint: streams the run's buffered log lines as they arrive.

    Purely a *view* onto the buffer filled by _pump_process_output — connecting,
    disconnecting, or reconnecting has no effect on the benchmark process. Each
    event carries its buffer index as the SSE id; on auto-reconnect the browser
    sends it back via Last-Event-ID and the stream resumes where it left off
    instead of replaying (or worse, re-spawning) from scratch.
    """
    entry = _RUNS.get(run_id)
    if not entry:
        async def _not_found():
            yield "data: __ERROR__\n\n"
        return StreamingResponse(_not_found(), media_type="text/event-stream")

    start_idx = 0
    last_event_id = request.headers.get("last-event-id")
    if last_event_id is not None:
        try:
            start_idx = int(last_event_id) + 1
        except ValueError:
            start_idx = 0

    async def generate() -> AsyncIterator[str]:
        idx = start_idx
        cond: asyncio.Condition = entry["cond"]
        while True:
            while idx < len(entry["lines"]):
                payload = json.dumps({"type": "log", "line": entry["lines"][idx]})
                yield f"id: {idx}\ndata: {payload}\n\n"
                idx += 1
            if entry["status"] != "running":
                break
            async with cond:
                # Re-check under the lock: a line may have landed (or the run
                # finished) between draining the buffer and acquiring the lock.
                if idx >= len(entry["lines"]) and entry["status"] == "running":
                    await cond.wait()

        if entry["status"] == "done":
            yield "data: __DONE__\n\n"
        else:
            yield "data: __ERROR__\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
