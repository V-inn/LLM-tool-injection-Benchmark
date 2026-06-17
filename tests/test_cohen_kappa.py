"""
test_cohen_kappa.py — the κ math (the real offline regression net for Phase 3).

The textbook case (2 raters, 2 categories, confusion [[20,5],[10,15]]) has a known
κ = 0.40, which pins down both p_observed/p_expected and the final coefficient.
"""
from rbac_benchmark.evaluation.kappa_validation import (
    cohen_kappa,
    interpret_kappa,
    compute_kappa_from_sampleset,
)


def test_known_value():
    # 25 A=YES (20 B=YES, 5 B=NO), 25 A=NO (10 B=YES, 15 B=NO) -> κ = 0.40.
    labels_a = ["YES"] * 25 + ["NO"] * 25
    labels_b = (["YES"] * 20 + ["NO"] * 5) + (["YES"] * 10 + ["NO"] * 15)
    r = cohen_kappa(labels_a, labels_b, categories=["YES", "NO"])
    assert abs(r["kappa"] - 0.40) < 1e-9
    assert abs(r["p_observed"] - 0.70) < 1e-9
    assert abs(r["p_expected"] - 0.50) < 1e-9
    assert r["n"] == 50
    assert r["interpretation"] == "Fair"


def test_degenerate_cases():
    # Perfect agreement across two categories -> 1.0
    assert cohen_kappa(["YES", "NO"], ["YES", "NO"], ["YES", "NO"])["kappa"] == 1.0
    # Both raters collapse onto a single category and agree -> 1.0 (no divide-by-zero)
    assert cohen_kappa(["YES", "YES"], ["YES", "YES"], ["YES", "NO"])["kappa"] == 1.0
    # No scorable pairs -> 0.0
    assert cohen_kappa([], [], ["YES", "NO"])["kappa"] == 0.0


def test_out_of_universe_labels_dropped():
    # Skip / JUDGE_ERROR style labels are excluded from scoring.
    r = cohen_kappa(["YES", "NO", "SKIP"], ["YES", "NO", "YES"], ["YES", "NO"])
    assert r["n"] == 2


def test_interpret_bands():
    assert interpret_kappa(-0.1) == "Poor (worse than chance)"
    assert interpret_kappa(0.1) == "Slight"
    assert interpret_kappa(0.3) == "Fair"
    assert interpret_kappa(0.5) == "Moderate"
    assert interpret_kappa(0.7) == "Substantial"
    assert interpret_kappa(0.85) == "Almost Perfect"


def test_kappa_from_worksheet(kappa_worksheet):
    r = compute_kappa_from_sampleset(kappa_worksheet)
    assert r["total"] == 12
    assert r["annotated"] == 11  # one row left unannotated
    assert r["n"] == 11
    assert r["kappa"] >= 0.80     # mostly-agreeing fixture -> Almost Perfect
    assert r["interpretation"] == "Almost Perfect"
