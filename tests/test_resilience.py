"""
test_resilience.py — crash-resume, infra-failure accounting, and live-membership
reconciliation. All offline: no Ollama, no network, no subprocess.

Covers the resilience layer added for master crash-resume + worker heal/join:
  • infra_error is excluded from total_inferences and from scoring's primary total;
  • resume rehydrates prior progress and enqueues only the missing iterations,
    including re-running iterations that only failed with infra errors;
  • a changed grid aborts a resume instead of merging incompatible data;
  • reconcile_registry evicts silent nodes, rejoins recovered ones, adopts new
    ones, and never evicts the master loopback.
"""
import json

import pytest

from rbac_benchmark.core.config import BenchmarkConfig, InferenceMetrics
from rbac_benchmark.evaluation.scoring import _primary_total
from rbac_benchmark.orchestration.master_node import (
    WorkerState,
    build_results_grid,
    reconcile_registry,
    resume_from_disk,
    write_provenance,
)


# --------------------------------------------------------------------------- #
# infra_error accounting
# --------------------------------------------------------------------------- #

def test_infra_error_excluded_from_total_inferences():
    m = InferenceMetrics(compliant=2, severity_1_violation=1, infra_error=5)
    # 2 + 1 scored outcomes; the 5 infra errors must NOT count as inferences.
    assert m.total_inferences == 3


def test_infra_error_excluded_from_scoring_primary_total():
    cell = {"compliant": 2, "severity_1_violation": 1, "infra_error": 5}
    # scoring's denominator must match total_inferences and ignore infra_error.
    assert _primary_total(cell) == 3


def test_infra_error_survives_dataclass_roundtrip():
    m = InferenceMetrics(compliant=1, infra_error=4)
    restored = InferenceMetrics.from_dict(m.to_dict())
    assert restored.infra_error == 4
    # Legacy files with no infra_error key load as 0.
    legacy = InferenceMetrics.from_dict({"compliant": 1})
    assert legacy.infra_error == 0


# --------------------------------------------------------------------------- #
# Resume reconstruction
# --------------------------------------------------------------------------- #

_SYS = {"S1": "system prompt text"}
_INJ = {"I1": "injection payload text"}
_META = {"I1": {"lever": "AUTHORITY", "target_severity": 3}}
_KEY = "m | S1 | I1"


def _cfg(tmp_path, **kw):
    kw.setdefault("iterations", 5)
    return BenchmarkConfig(models=["m"], output=str(tmp_path / "results.json"), **kw)


def _write_results(tmp_path, cell: dict):
    (tmp_path / "results.json").write_text(json.dumps({_KEY: cell}), encoding="utf-8")


def test_resume_enqueues_only_remaining(tmp_path):
    cfg = _cfg(tmp_path, resume=True)
    # 3 scored inferences already done in the only cell.
    _write_results(tmp_path, {"compliant": 2, "severity_1_violation": 1})
    grid = build_results_grid(cfg, _SYS, _INJ, _META)

    loaded, _ = resume_from_disk(cfg, grid, list(_SYS), list(_INJ), _META)
    assert loaded == 1
    done = grid[_KEY].total_inferences
    assert done == 3
    remaining = max(0, cfg.iterations - done)
    assert remaining == 2  # only the 2 missing iterations get re-enqueued


def test_resume_reruns_infra_failed_iterations(tmp_path):
    cfg = _cfg(tmp_path, resume=True)
    # A cell where every attempt only ever hit an infra error: 0 scored, 4 infra.
    _write_results(tmp_path, {"infra_error": 4})
    grid = build_results_grid(cfg, _SYS, _INJ, _META)

    resume_from_disk(cfg, grid, list(_SYS), list(_INJ), _META)
    done = grid[_KEY].total_inferences
    assert done == 0  # infra failures are not "done"
    assert max(0, cfg.iterations - done) == cfg.iterations  # all iterations re-run


def test_resume_heals_missing_lever_metadata(tmp_path):
    cfg = _cfg(tmp_path, resume=True)
    # Legacy cell without lever/severity metadata.
    _write_results(tmp_path, {"compliant": 1})
    grid = build_results_grid(cfg, _SYS, _INJ, _META)

    resume_from_disk(cfg, grid, list(_SYS), list(_INJ), _META)
    assert grid[_KEY].injection_lever == "AUTHORITY"
    assert grid[_KEY].target_severity == 3


def test_resume_missing_file_starts_fresh(tmp_path):
    cfg = _cfg(tmp_path, resume=True)  # no results file written
    grid = build_results_grid(cfg, _SYS, _INJ, _META)
    loaded, resumed_from = resume_from_disk(cfg, grid, list(_SYS), list(_INJ), _META)
    assert loaded == 0 and resumed_from is None
    assert grid[_KEY].total_inferences == 0


def test_resume_aborts_on_grid_mismatch(tmp_path):
    # Prior run recorded a provenance sidecar for iterations=5; now we try to
    # resume with iterations=7 — an incompatible grid, which must abort.
    old_cfg = _cfg(tmp_path, iterations=5, run_id="old-run")
    _write_results(tmp_path, {"compliant": 1})
    from rbac_benchmark.orchestration.master_node import _grid_signature
    grid_sig = _grid_signature(old_cfg.models, list(_SYS), list(_INJ), old_cfg.iterations)
    write_provenance(old_cfg, old_cfg.output, "2026-01-01T00:00:00Z", None, grid_sig)

    new_cfg = _cfg(tmp_path, iterations=7, resume=True)
    grid = build_results_grid(new_cfg, _SYS, _INJ, _META)
    with pytest.raises(SystemExit):
        resume_from_disk(new_cfg, grid, list(_SYS), list(_INJ), _META)


def test_resume_records_resumed_from_run_id(tmp_path):
    old_cfg = _cfg(tmp_path, iterations=5, run_id="run-A")
    _write_results(tmp_path, {"compliant": 1})
    from rbac_benchmark.orchestration.master_node import _grid_signature
    grid_sig = _grid_signature(old_cfg.models, list(_SYS), list(_INJ), old_cfg.iterations)
    write_provenance(old_cfg, old_cfg.output, "2026-01-01T00:00:00Z", None, grid_sig)

    new_cfg = _cfg(tmp_path, iterations=5, resume=True, run_id="run-B")
    grid = build_results_grid(new_cfg, _SYS, _INJ, _META)
    _, resumed_from = resume_from_disk(new_cfg, grid, list(_SYS), list(_INJ), _META)
    assert resumed_from == "run-A"


# --------------------------------------------------------------------------- #
# Live worker membership reconciliation
# --------------------------------------------------------------------------- #

def _mem_cfg(stale_after=30.0):
    return BenchmarkConfig(models=["m"], worker_stale_after=stale_after)


def test_membership_adopts_new_responder():
    cfg = _mem_cfg()
    registry: dict = {}
    transitions, to_spawn = reconcile_registry(registry, ["10.0.0.5"], now=100.0, config=cfg)
    assert "10.0.0.5" in registry
    assert registry["10.0.0.5"].healthy is True
    # A brand-new healthy node with no consumers yet is queued for spawning.
    assert [ws.ip for ws in to_spawn] == ["10.0.0.5"]


def test_membership_evicts_silent_node_then_rejoins():
    cfg = _mem_cfg(stale_after=30.0)
    registry = {"10.0.0.5": WorkerState(ip="10.0.0.5", last_seen=100.0, healthy=True,
                                        consumers_spawned=True)}
    # 40s later with no responder → past the 30s window → evicted.
    transitions, _ = reconcile_registry(registry, [], now=140.0, config=cfg)
    assert ("10.0.0.5", "evicted") in transitions
    assert registry["10.0.0.5"].healthy is False

    # Heartbeat resumes at t=150 → fresh again → rejoined.
    transitions, to_spawn = reconcile_registry(registry, ["10.0.0.5"], now=150.0, config=cfg)
    assert ("10.0.0.5", "rejoined") in transitions
    assert registry["10.0.0.5"].healthy is True
    # Consumers were already spawned before eviction, so no re-spawn on rejoin.
    assert to_spawn == []


def test_membership_never_evicts_master_loopback():
    cfg = _mem_cfg(stale_after=30.0)
    registry = {"127.0.0.1": WorkerState(ip="127.0.0.1", last_seen=0.0, healthy=True,
                                         is_master=True)}
    # Far past the staleness window, but the master has no heartbeat by design.
    transitions, to_spawn = reconcile_registry(registry, [], now=9999.0, config=cfg)
    assert transitions == []
    assert registry["127.0.0.1"].healthy is True
    # It still needs its consumers spawned once.
    assert [ws.ip for ws in to_spawn] == ["127.0.0.1"]


def test_membership_no_respawn_when_consumers_exist():
    cfg = _mem_cfg()
    registry = {"10.0.0.5": WorkerState(ip="10.0.0.5", last_seen=100.0, healthy=True,
                                        consumers_spawned=True)}
    _, to_spawn = reconcile_registry(registry, ["10.0.0.5"], now=100.0, config=cfg)
    assert to_spawn == []
