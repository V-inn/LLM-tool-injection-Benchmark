"""prompts.py — custom system-prompt CRUD"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from rbac_benchmark.paths import data_path

router = APIRouter()

_PROMPTS_FILE = data_path("custom_prompts.json")


def _load() -> dict:
    p = Path(_PROMPTS_FILE)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    Path(_PROMPTS_FILE).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_list(raw: dict) -> list[dict]:
    out = []
    for key, val in raw.items():
        disabled = key.startswith("_disabled:")
        clean_key = key.removeprefix("_disabled:")
        if isinstance(val, str):
            out.append({"key": clean_key, "name": clean_key, "text": val,
                        "system_prompt": val, "levers": [], "disabled": disabled})
        else:
            entry = dict(val)
            entry.setdefault("name", clean_key)
            entry.setdefault("text", entry.get("system_prompt", ""))
            entry.setdefault("levers", [])
            entry["key"] = clean_key
            entry["disabled"] = disabled
            out.append(entry)
    return out


# ── GET /api/prompts ──────────────────────────────────────────────────────────
@router.get("")
def list_prompts():
    return {"prompts": _to_list(_load())}


# ── POST /api/prompts/save ────────────────────────────────────────────────────
@router.post("/save")
def save_prompt(body: dict):
    key   = (body.get("key") or body.get("name", "")).strip()
    name  = body.get("name", key).strip()
    text  = body.get("text", "").strip()
    levers = body.get("levers", [])
    if not key or not text:
        raise HTTPException(400, "key and text required")
    data = _load()
    # Remove any disabled version first
    data.pop(f"_disabled:{key}", None)
    data[key] = {"name": name, "system_prompt": text, "levers": levers}
    _save(data)
    return {"ok": True}


# ── POST /api/prompts/toggle/{key} ───────────────────────────────────────────
@router.post("/toggle/{key}")
def toggle_prompt(key: str):
    data = _load()
    if key in data:
        data[f"_disabled:{key}"] = data.pop(key)
    elif f"_disabled:{key}" in data:
        data[key] = data.pop(f"_disabled:{key}")
    else:
        raise HTTPException(404, f"Prompt '{key}' not found")
    _save(data)
    return {"ok": True}


# ── DELETE /api/prompts/delete/{key} ─────────────────────────────────────────
@router.delete("/delete/{key}")
def delete_prompt(key: str):
    data = _load()
    data.pop(key, None)
    data.pop(f"_disabled:{key}", None)
    _save(data)
    return {"ok": True}
