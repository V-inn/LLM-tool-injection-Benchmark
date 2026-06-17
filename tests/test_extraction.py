"""
test_extraction.py — Phase-3 offline extraction + stratified worksheet build.

Verifies the faithful path: stored per-trace Judge labels are read straight out of the
results file (no re-classification) and a blank-annotation worksheet is produced.
"""
from rbac_benchmark.evaluation.kappa_validation import (
    CATEGORIES,
    extract_thought_samples,
    build_sample_set_offline,
)


def test_extract_picks_up_stored_labels(results_file):
    samples = extract_thought_samples(results_file)
    # Two seeded cells, two traces each.
    assert len(samples) == 4
    assert all(s["machine_label"] in CATEGORIES for s in samples)
    # Provenance is preserved.
    assert all(s["matrix_key"].count(" | ") == 2 for s in samples)
    labels = sorted(s["machine_label"] for s in samples)
    assert labels == ["AUTHORITY_BIAS", "AUTHORITY_BIAS", "INSTRUCTION_AMNESIA", "URGENCY_PANIC"]


def test_build_sample_set_offline(results_file, tmp_path):
    out = tmp_path / "ks.json"
    built = build_sample_set_offline(results_file, output_path=str(out), per_category=20, seed=42)
    assert len(built) == 4
    assert all(b["human_label"] is None for b in built)  # worksheet starts blank
    assert out.exists()
