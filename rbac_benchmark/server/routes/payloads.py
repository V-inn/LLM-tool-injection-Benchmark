"""payloads.py — injection/defense generation API"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from rbac_benchmark.paths import data_path

router = APIRouter()

_INJECTIONS_FILE = data_path("generated_injections.json")
_RESULTS_FILE    = data_path("benchmark_results.json")
_PROMPTS_FILE    = data_path("custom_prompts.json")


def _load_json(path: str) -> dict | list | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── GET /api/payloads/injections ──────────────────────────────────────────────
@router.get("/injections")
def list_injections():
    raw = _load_json(_INJECTIONS_FILE)
    if raw is None:
        return {"injections": [], "generated_at": None}
    items = raw if isinstance(raw, list) else raw.get("injections", [])
    generated_at = raw.get("generated_at") if isinstance(raw, dict) else None
    return {"injections": items, "generated_at": generated_at}


# ── POST /api/payloads/generate-injections ────────────────────────────────────
@router.post("/generate-injections")
async def generate_injections(body: dict):
    levers     = body.get("levers", [])
    severities = body.get("severities", [1, 2, 3])
    count      = int(body.get("count", 5))
    context    = body.get("context", "")

    try:
        from rbac_benchmark.generation.injection_generator import generate_injections as _gen
        injections = _gen(levers=levers, severities=severities, count=count, context=context)
        import datetime
        output = {"injections": injections, "generated_at": datetime.datetime.now().isoformat()}
        Path(_INJECTIONS_FILE).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        return output
    except ImportError:
        raise HTTPException(501, "injection_generator not available")
    except Exception as e:
        raise HTTPException(500, str(e))


# ── POST /api/payloads/generate-defenses ─────────────────────────────────────
@router.post("/generate-defenses")
async def generate_defenses(body: dict):
    strategies = body.get("strategies", [])
    levers     = body.get("levers", [])
    count      = int(body.get("count", 3))

    try:
        from rbac_benchmark.generation.defense_generator import generate_defenses as _gen
        defenses = _gen(strategies=strategies, levers=levers, count=count)
        # Merge into custom_prompts.json
        existing = _load_json(_PROMPTS_FILE) or {}
        for d in defenses:
            key = d.get("key") or d.get("name", "generated")
            existing[key] = {"name": d.get("name", key), "system_prompt": d.get("text", ""), "levers": levers}
        Path(_PROMPTS_FILE).write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"defenses": defenses, "count": len(defenses)}
    except ImportError:
        raise HTTPException(501, "defense_generator not available")
    except Exception as e:
        raise HTTPException(500, str(e))


# ── GET /api/payloads/calibration ────────────────────────────────────────────
@router.get("/calibration")
def get_calibration(ref_model: str = "", threshold: float = 0.10):
    results = _load_json(_RESULTS_FILE)
    if not results:
        return {"validity": {}}
    # Find first model as ref if not specified
    if not ref_model:
        for key in results:
            parts = key.split(" | ")
            if len(parts) == 3:
                ref_model = parts[0]
                break
    if not ref_model:
        return {"validity": {}}
    from rbac_benchmark.evaluation.analyzer import validate_attack_strength
    validity = validate_attack_strength(_RESULTS_FILE, ref_model, threshold=threshold)
    return {"validity": validity, "ref_model": ref_model}
