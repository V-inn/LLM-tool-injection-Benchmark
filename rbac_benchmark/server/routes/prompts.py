"""prompts.py — defense system-prompt CRUD (custom + generated).

The Custom Prompts tab manages every EDITABLE defense the benchmark will run: prompts
written in the editor (custom_prompts.json) AND prompts produced by the Blue-Team
generator (generated_defenses.json). Both files feed core.prompts.load_all_prompts, so
both must be visible and manageable here — otherwise a generated defense runs in the
matrix with no way to inspect, disable, or delete it. The base prompts (S1–S3) live in
code and are intentionally not listed (immutable baseline, like the base injections).

Disable is a "_disabled:" key prefix; core.prompts skips those keys so a toggled-off
prompt stays in its file but never enters the live matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from rbac_benchmark.paths import data_path

router = APIRouter()

# Two defense stores. custom_prompts.json holds editor entries ({name, system_prompt,
# levers} objects); generated_defenses.json holds generator output (bare prompt strings).
# Each is written with the indent its original producer used, to keep diffs small.
_CUSTOM_FILE = data_path("custom_prompts.json")
_GEN_FILE    = data_path("generated_defenses.json")
_STORES = ((_CUSTOM_FILE, 2, "custom"), (_GEN_FILE, 4, "generated"))


def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(path: str, data: dict, indent: int) -> None:
    Path(path).write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")


def _entry(key: str, val, source: str) -> dict:
    """Normalise one stored value (bare string OR {name, system_prompt, levers}) into the
    row shape the UI consumes, carrying its source and enabled/disabled state."""
    disabled = key.startswith("_disabled:")
    clean = key.removeprefix("_disabled:")
    if isinstance(val, str):
        text, name, levers = val, clean, []
    else:
        text = val.get("system_prompt") or val.get("text") or ""
        name = val.get("name", clean)
        levers = val.get("levers", [])
    return {"key": clean, "name": name, "text": text, "system_prompt": text,
            "levers": levers, "disabled": disabled, "source": source}


def _locate(key: str):
    """Find the store holding `key` (enabled or disabled). Custom is checked first so a
    prompt promoted into custom_prompts.json wins over any stale generated copy. Returns
    (path, indent, stored_key, data) or None."""
    for path, indent, _src in _STORES:
        data = _load(path)
        if key in data:
            return path, indent, key, data
        if f"_disabled:{key}" in data:
            return path, indent, f"_disabled:{key}", data
    return None


# ── GET /api/prompts ──────────────────────────────────────────────────────────
@router.get("")
def list_prompts():
    # Generated first, then custom, so an edited/overriding custom entry wins the dedupe
    # on the same clean key (custom is the editable store).
    by_key: dict[str, dict] = {}
    for path, _indent, source in reversed(_STORES):  # generated, then custom
        for k, v in _load(path).items():
            e = _entry(k, v, source)
            by_key[e["key"]] = e
    return {"prompts": list(by_key.values())}


# ── POST /api/prompts/save ────────────────────────────────────────────────────
@router.post("/save")
def save_prompt(body: dict):
    key   = (body.get("key") or body.get("name", "")).strip()
    name  = body.get("name", key).strip()
    text  = body.get("text", "").strip()
    levers = body.get("levers", [])
    if not key or not text:
        raise HTTPException(400, "key and text required")
    if key.startswith("_disabled:"):
        raise HTTPException(400, "key may not start with '_disabled:'")
    # Edits always land in the editable custom store, enabled.
    data = _load(_CUSTOM_FILE)
    data.pop(f"_disabled:{key}", None)
    data[key] = {"name": name, "system_prompt": text, "levers": levers}
    _save(_CUSTOM_FILE, data, 2)
    # Promote out of the generated store: editing a generated defense moves it to custom so
    # there is no stale duplicate shadowing the edit (custom wins on load, but keeping both
    # would duplicate the row and trip the collision warning).
    gen = _load(_GEN_FILE)
    if gen.pop(key, None) is not None or gen.pop(f"_disabled:{key}", None) is not None:
        _save(_GEN_FILE, gen, 4)
    return {"ok": True}


# ── POST /api/prompts/toggle/{key} ───────────────────────────────────────────
@router.post("/toggle/{key}")
def toggle_prompt(key: str):
    loc = _locate(key)
    if not loc:
        raise HTTPException(404, f"Prompt '{key}' not found")
    path, indent, stored_key, data = loc
    if stored_key.startswith("_disabled:"):
        data[stored_key.removeprefix("_disabled:")] = data.pop(stored_key)
    else:
        data[f"_disabled:{stored_key}"] = data.pop(stored_key)
    _save(path, data, indent)
    return {"ok": True}


# ── DELETE /api/prompts/delete/{key} ─────────────────────────────────────────
@router.delete("/delete/{key}")
def delete_prompt(key: str):
    # Remove from whichever store(s) hold it, enabled or disabled.
    for path, indent, _src in _STORES:
        data = _load(path)
        removed = data.pop(key, None) is not None
        removed |= data.pop(f"_disabled:{key}", None) is not None
        if removed:
            _save(path, data, indent)
    return {"ok": True}
