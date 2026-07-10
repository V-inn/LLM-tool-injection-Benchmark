"""payloads.py — injection/defense generation API"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from rbac_benchmark.paths import data_path


@contextlib.contextmanager
def _gemini_key(token: str | None):
    """Temporarily override GEMINI_API_KEY for the duration of a generation call."""
    if not token:
        yield
        return
    prev = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = token
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = prev

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


def _payload_row(key: str, value) -> dict:
    """Render one injection payload for the UI, tolerating BOTH the legacy bare-string
    schema and the tagged {text, lever, target_severity} schema written by the upgraded
    generator. Surfaces the design lever/severity so the dashboard can show/filter them.
    A "_disabled:" key prefix marks a payload toggled off in the UI — reported with a clean
    key and disabled=True so the list can render/re-enable it without running it."""
    disabled = key.startswith("_disabled:")
    clean = key.removeprefix("_disabled:")
    if isinstance(value, dict):
        text = value.get("text", "")
        return {"key": clean, "text": text, "prompt": text,
                "lever": value.get("lever"), "target_severity": value.get("target_severity"),
                "disabled": disabled}
    return {"key": clean, "text": value, "prompt": value,
            "lever": None, "target_severity": None, "disabled": disabled}


def _load_injections_map() -> dict:
    """Return the raw {key: payload} injection map (the schema the generator writes), or an
    empty dict for a missing/legacy-wrapped file. CRUD below operates on this shape."""
    raw = _load_json(_INJECTIONS_FILE)
    return raw if isinstance(raw, dict) and "injections" not in raw else {}


def _save_injections_map(data: dict) -> None:
    Path(_INJECTIONS_FILE).write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


# ── GET /api/payloads/injections ──────────────────────────────────────────────
@router.get("/injections")
def list_injections():
    raw = _load_json(_INJECTIONS_FILE)
    if raw is None:
        return {"injections": [], "generated_at": None}
    if isinstance(raw, dict) and "injections" not in raw:
        # generator writes a plain {key: payload} dict — payload is a bare string (legacy)
        # or a {text, lever, target_severity} object (tagged schema).
        items = [_payload_row(k, v) for k, v in raw.items()]
        generated_at = None
    elif isinstance(raw, dict):
        items = raw.get("injections", [])
        generated_at = raw.get("generated_at")
    else:
        items = raw
        generated_at = None
    return {"injections": items, "generated_at": generated_at}


# ── POST /api/payloads/injections/save ────────────────────────────────────────
@router.post("/injections/save")
def save_injection(body: dict):
    """Create or edit one injection payload (mirrors /api/prompts/save for defenses).
    Writes the tagged {text, lever, target_severity} schema. Saving re-enables a payload
    that was disabled (drops its "_disabled:" twin)."""
    key  = (body.get("key") or "").strip()
    text = (body.get("text") or "").strip()
    if not key or not text:
        raise HTTPException(400, "key and text required")
    if key.startswith("_disabled:"):
        raise HTTPException(400, "key may not start with '_disabled:'")
    lever = (body.get("lever") or "").strip().upper() or None
    sev = body.get("target_severity")
    try:
        sev = int(sev) if sev not in (None, "") else None
    except (TypeError, ValueError):
        sev = None
    entry: dict = {"text": text}
    if lever:
        entry["lever"] = lever
    if sev is not None:
        entry["target_severity"] = sev
    data = _load_injections_map()
    data.pop(f"_disabled:{key}", None)  # a save always lands as enabled
    data[key] = entry
    _save_injections_map(data)
    return {"ok": True}


# ── POST /api/payloads/injections/toggle/{key} ────────────────────────────────
@router.post("/injections/toggle/{key}")
def toggle_injection(key: str):
    """Flip a payload between enabled and disabled by moving it under/out of the
    "_disabled:" key prefix. Disabled payloads are kept in the file but excluded from the
    live attack matrix by core.prompts.load_all_prompts / load_payload_metadata."""
    data = _load_injections_map()
    if key in data:
        data[f"_disabled:{key}"] = data.pop(key)
    elif f"_disabled:{key}" in data:
        data[key] = data.pop(f"_disabled:{key}")
    else:
        raise HTTPException(404, f"Injection '{key}' not found")
    _save_injections_map(data)
    return {"ok": True}


# ── DELETE /api/payloads/injections/delete/{key} ──────────────────────────────
@router.delete("/injections/delete/{key}")
def delete_injection(key: str):
    data = _load_injections_map()
    data.pop(key, None)
    data.pop(f"_disabled:{key}", None)
    _save_injections_map(data)
    return {"ok": True}


# ── POST /api/payloads/generate-injections ────────────────────────────────────
@router.post("/generate-injections")
async def generate_injections(body: dict):
    model      = body.get("model", "gemini-2.5-flash")
    levers     = body.get("levers", []) or None
    severities = body.get("severities", []) or None
    count      = int(body.get("count", 5))
    context    = body.get("context", "").strip() or None
    token      = body.get("gemini_api_key", "").strip() or None

    try:
        from rbac_benchmark.generation.injection_generator import generate_injections as _gen
    except ImportError:
        raise HTTPException(501, "injection_generator not available")

    import datetime
    try:
        with _gemini_key(token):
            await _gen(
                attacker_model=model,
                num_payloads=count,
                output_file=_INJECTIONS_FILE,
                levers=levers,
                severities=severities,
                context=context,
            )
        raw = _load_json(_INJECTIONS_FILE)
        if isinstance(raw, dict) and "injections" not in raw:
            injections = [_payload_row(k, v) for k, v in raw.items()]
        else:
            injections = (raw or {}).get("injections", raw) if isinstance(raw, dict) else (raw or [])
        return {"injections": injections, "generated_at": datetime.datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── POST /api/payloads/generate-defenses ─────────────────────────────────────
@router.post("/generate-defenses")
async def generate_defenses(body: dict):
    model      = body.get("model", "gemini-2.5-flash")
    strategies = body.get("strategies", []) or None
    levers     = body.get("levers", []) or None
    count      = int(body.get("count", 3))
    token      = body.get("gemini_api_key", "").strip() or None
    _DEFENSES_FILE = data_path("generated_defenses.json")

    try:
        from rbac_benchmark.generation.defense_generator import generate_defenses as _gen
    except ImportError:
        raise HTTPException(501, "defense_generator not available")

    try:
        with _gemini_key(token):
            await _gen(model=model, output_file=_DEFENSES_FILE, strategies=strategies, levers=levers, count=count)
        raw = _load_json(_DEFENSES_FILE)
        if not isinstance(raw, dict):
            raise HTTPException(500, "defense_generator returned unexpected output")
        defenses = [{"key": k, "name": k, "text": v, "system_prompt": v} for k, v in raw.items()]
        existing = _load_json(_PROMPTS_FILE) or {}
        for d in defenses:
            existing[d["key"]] = {"name": d["name"], "system_prompt": d["text"], "levers": levers or []}
        Path(_PROMPTS_FILE).write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"defenses": defenses, "count": len(defenses)}
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
