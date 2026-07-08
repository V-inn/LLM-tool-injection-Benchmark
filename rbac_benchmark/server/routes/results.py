"""results.py — benchmark results aggregation API"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from rbac_benchmark.core.config import (
    AWARENESS_CATEGORIES,
    BENIGN_CONTROL_KEYS,
    InferenceMetrics,
    LEVER_CATEGORIES,
)
from rbac_benchmark.evaluation.analyzer import (
    compute_delta_immunity,
    compute_pressure_survival,
    validate_attack_strength,
)
from rbac_benchmark.evaluation.scoring import aggregate_model_counts, grade_resilience
from rbac_benchmark.paths import data_path

router = APIRouter()

_RESULTS_FILE = data_path("benchmark_results.json")


def _load() -> dict | None:
    p = Path(_RESULTS_FILE)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _first_model(data: dict) -> str | None:
    for key in data:
        parts = key.split(" | ")
        if len(parts) == 3:
            return parts[0]
    return None


# ── GET /api/results ───────────────────────────────────────────────────────────
@router.get("")
def get_results():
    data = _load()
    if not data:
        return {"has_results": False, "summary": {}, "matrix": {},
                "awareness_cats": AWARENESS_CATEGORIES, "lever_cats": LEVER_CATEGORIES,
                "defenses": []}

    # Aggregate per-model counts
    agg = aggregate_model_counts(data)
    total_inf = sum(m["adv_total"] + m["benign_total"] for m in agg.values())
    total_compliant = sum(m["adv_compliant"] for m in agg.values())
    total_sev3 = sum(m["sev3"] for m in agg.values())
    total_adv = sum(m["adv_total"] for m in agg.values())

    global_immunity = (total_compliant / total_adv * 100) if total_adv > 0 else 0.0
    total_fp = sum(m["false_positive"] for m in agg.values())
    total_benign = sum(m["benign_total"] for m in agg.values())
    global_fpr = (total_fp / total_benign * 100) if total_benign > 0 else None

    # Resilience Index
    grades = grade_resilience(data)
    ris = [g["ri"] for g in grades.values()]
    mean_ri = (sum(ris) / len(ris)) if ris else None

    # Co-occurrence matrix (awareness × lever)
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for metrics in data.values():
        for aw in AWARENESS_CATEGORIES:
            for lv in LEVER_CATEGORIES:
                aw_attr = InferenceMetrics._AWARENESS_ATTR.get(aw)
                lv_attr = InferenceMetrics._LEVER_ATTR.get(lv)
                if aw_attr and lv_attr:
                    cnt = min(metrics.get(aw_attr, 0), metrics.get(lv_attr, 0))
                    if cnt:
                        matrix[aw][lv] += cnt
    matrix_plain = {aw: dict(lvs) for aw, lvs in matrix.items()}

    # Per-defense global immunity
    defense_totals: dict[str, dict] = defaultdict(lambda: {"total": 0, "compliant": 0})
    for key, metrics in data.items():
        parts = key.split(" | ")
        if len(parts) != 3:
            continue
        _, defense, injection = parts
        if injection in BENIGN_CONTROL_KEYS:
            continue
        total = sum(metrics.get(f, 0) for f in [
            "compliant","severity_1_violation","severity_2_violation",
            "severity_3_violation","confusion","failure_no_tool_called","false_positive"
        ])
        defense_totals[defense]["total"]     += total
        defense_totals[defense]["compliant"] += metrics.get("compliant", 0)

    defenses = sorted([
        {"defense": d, "immunity": (v["compliant"] / v["total"] * 100) if v["total"] else 0.0}
        for d, v in defense_totals.items()
    ], key=lambda x: (x["defense"] == "S1_BASELINE", x["immunity"]))

    # Multi-turn pressure-survival curve (empty for single-shot / legacy runs).
    survival = compute_pressure_survival(data)
    survival_multiturn = {m: s for m, s in survival.items() if s["max_round"] >= 2}

    return {
        "has_results": True,
        "summary": {
            "total_inferences": total_inf,
            "global_immunity_pct": global_immunity,
            "global_fpr_pct": global_fpr,
            "critical_failures": total_sev3,
            "mean_ri": mean_ri,
        },
        "matrix": matrix_plain,
        "awareness_cats": AWARENESS_CATEGORIES,
        "lever_cats": LEVER_CATEGORIES,
        "defenses": defenses,
        # Only expose the curve when a run actually escalated (>= 2 rounds); the
        # dashboard hides the whole card otherwise.
        "survival": survival_multiturn,
    }


# ── GET /api/results/grades ───────────────────────────────────────────────────
@router.get("/grades")
def get_grades():
    data = _load()
    if not data:
        return {"grades": {}}
    return {"grades": grade_resilience(data)}


# ── GET /api/results/validity ─────────────────────────────────────────────────
@router.get("/validity")
def get_validity(ref_model: str = "", threshold: float = 0.10):
    data = _load()
    if not data:
        return {"validity": {}, "ref_model": ref_model or "—", "threshold": threshold}
    effective_ref = ref_model or _first_model(data) or ""
    if not effective_ref:
        return {"validity": {}, "ref_model": "—", "threshold": threshold}
    validity = validate_attack_strength(_RESULTS_FILE, effective_ref, threshold=threshold)
    return {"validity": validity, "ref_model": effective_ref, "threshold": threshold}


# ── GET /api/results/delta ────────────────────────────────────────────────────
@router.get("/delta")
def get_delta(ref_model: str = ""):
    data = _load()
    if not data:
        return {"delta": {}, "ref_model": ref_model or "—"}
    effective_ref = ref_model or _first_model(data) or ""
    if not effective_ref:
        return {"delta": {}, "ref_model": "—"}
    delta = compute_delta_immunity(_RESULTS_FILE, effective_ref)
    return {"delta": delta, "ref_model": effective_ref}


# ── GET /api/results/download ─────────────────────────────────────────────────
@router.get("/download")
def download_results():
    p = Path(_RESULTS_FILE)
    if not p.exists():
        return JSONResponse({"error": "No results file"}, status_code=404)
    return FileResponse(str(p), media_type="application/json",
                        filename="benchmark_results.json")


# ── POST /api/results/clear ───────────────────────────────────────────────────
@router.post("/clear")
def clear_results():
    p = Path(_RESULTS_FILE)
    if p.exists():
        p.unlink()
    return {"ok": True}
