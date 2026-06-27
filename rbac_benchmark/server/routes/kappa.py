"""kappa.py — κ validation + blind annotation API"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException

from rbac_benchmark.core.config import AWARENESS_CATEGORIES, LEVER_CATEGORIES
from rbac_benchmark.evaluation.kappa_validation import (
    DEFAULT_SAMPLESET_PATH,
    build_sample_set_offline,
    compute_kappa_from_sampleset,
    extract_thought_samples,
    stratify_samples,
)
from rbac_benchmark.paths import data_path

router = APIRouter()

_RESULTS_FILE = data_path("benchmark_results.json")
_SAMPLE_FILE  = DEFAULT_SAMPLESET_PATH


def _load_sample() -> list[dict]:
    p = Path(_SAMPLE_FILE)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _save_sample(records: list[dict]) -> None:
    Path(_SAMPLE_FILE).write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ── GET /api/kappa ────────────────────────────────────────────────────────────
@router.get("")
def get_kappa():
    if not Path(_SAMPLE_FILE).exists():
        return {"kappa": None, "kappa_a": None, "kappa_b": None}
    try:
        result = compute_kappa_from_sampleset(_SAMPLE_FILE)
        return {
            "kappa":      result["awareness"]["kappa"],
            "kappa_a":    result["awareness"]["kappa"],
            "kappa_b":    result["lever"]["kappa"],
            "annotated":  result["annotated"],
            "total":      result["total"],
            "confusion_a": result["awareness"]["confusion"],
            "confusion_b": result["lever"]["confusion"],
            "interp_a":   result["awareness"]["interpretation"],
            "interp_b":   result["lever"]["interpretation"],
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ── GET /api/kappa/sample ─────────────────────────────────────────────────────
@router.get("/sample")
def get_sample():
    records = _load_sample()
    if not records:
        return {"sample": [], "annotations": {}, "breakdown": {}}
    annotations = {
        str(r["sample_id"]): {
            "axis_a": r.get("human_awareness"),
            "axis_b": r.get("human_lever"),
        }
        for r in records
        if r.get("human_awareness") is not None
    }
    breakdown = dict(Counter(r["machine_awareness"] for r in records))
    # Reformat for frontend: {id, text, judge_axis_a, judge_axis_b, ...}
    sample_out = [
        {
            "id":          str(r["sample_id"]),
            "model":       r.get("model", ""),
            "defense":     r.get("defense", ""),
            "attack":      r.get("attack", ""),
            "text":        r.get("text", ""),
            "agent_response": r.get("text", ""),
            "judge_axis_a": r.get("machine_awareness"),
            "judge_axis_b": r.get("machine_lever"),
        }
        for r in records
    ]
    return {"sample": sample_out, "annotations": annotations, "breakdown": breakdown}


# ── POST /api/kappa/build-sample ──────────────────────────────────────────────
@router.post("/build-sample")
def build_sample(body: dict):
    per_cat = int(body.get("per_cat", 20))
    seed    = int(body.get("seed", 42))
    if not Path(_RESULTS_FILE).exists():
        raise HTTPException(400, "No benchmark_results.json found. Run a benchmark first.")
    try:
        records = build_sample_set_offline(_RESULTS_FILE, _SAMPLE_FILE, per_cat, seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
    breakdown = dict(Counter(r["machine_awareness"] for r in records))
    return {"sample": records, "breakdown": breakdown}


# ── POST /api/kappa/annotate ──────────────────────────────────────────────────
@router.post("/annotate")
def annotate(body: dict):
    sample_id = str(body.get("sample_id", ""))
    axis_a    = body.get("axis_a")
    axis_b    = body.get("axis_b")
    if axis_a not in AWARENESS_CATEGORIES:
        raise HTTPException(400, f"Invalid Axis A: {axis_a}")
    if axis_b not in LEVER_CATEGORIES:
        raise HTTPException(400, f"Invalid Axis B: {axis_b}")
    records = _load_sample()
    for r in records:
        if str(r["sample_id"]) == sample_id:
            r["human_awareness"] = axis_a
            r["human_lever"]     = axis_b
            break
    _save_sample(records)
    return {"ok": True}


# ── POST /api/kappa/compute ───────────────────────────────────────────────────
@router.post("/compute")
def compute_kappa():
    if not Path(_SAMPLE_FILE).exists():
        raise HTTPException(400, "No sample set. Build sample first.")
    try:
        result = compute_kappa_from_sampleset(_SAMPLE_FILE)
        return {
            "kappa":      result["awareness"]["kappa"],
            "kappa_a":    result["awareness"]["kappa"],
            "kappa_b":    result["lever"]["kappa"],
            "annotated":  result["annotated"],
            "total":      result["total"],
            "confusion_a": result["awareness"]["confusion"],
            "confusion_b": result["lever"]["confusion"],
            "interp_a":   result["awareness"]["interpretation"],
            "interp_b":   result["lever"]["interpretation"],
        }
    except Exception as e:
        raise HTTPException(500, str(e))
