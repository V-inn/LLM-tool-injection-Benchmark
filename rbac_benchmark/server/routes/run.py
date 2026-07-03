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

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from rbac_benchmark.core.config import BenchmarkConfig
from rbac_benchmark.paths import data_path

router = APIRouter()

_RESULTS_FILE = data_path("benchmark_results.json")
_RUNS: dict[str, asyncio.subprocess.Process] = {}


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
        from fastapi import HTTPException
        raise HTTPException(400, "models list required")

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

    _RUNS[run_id] = {"cfg_path": str(cfg_path), "proc": None, "status": "pending"}
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
async def stream_run(run_id: str):
    """SSE endpoint: spawns rbac-master, streams stdout as JSON log lines."""
    entry = _RUNS.get(run_id)
    if not entry:
        async def _not_found():
            yield "data: __ERROR__\n\n"
        return StreamingResponse(_not_found(), media_type="text/event-stream")

    cfg_path = entry["cfg_path"]

    async def generate() -> AsyncIterator[str]:
        # Locate the rbac-master entry-point script
        master_module = "rbac_benchmark.orchestration.master_node"
        cmd = [sys.executable, "-m", master_module, "--config", cfg_path]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        entry["proc"] = proc

        try:
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                payload = json.dumps({"type": "log", "line": line})
                yield f"data: {payload}\n\n"
        finally:
            await proc.wait()
            # Clean up temp config
            try:
                Path(cfg_path).unlink(missing_ok=True)
            except Exception:
                pass
            _RUNS.pop(run_id, None)

        rc = proc.returncode
        if rc == 0:
            yield "data: __DONE__\n\n"
        else:
            yield "data: __ERROR__\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
