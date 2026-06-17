"""
test_analyzer.py — Phase-1/2 metric math: InferenceMetrics accounting, ΔTPR, and
attack-strength validation. Migrated from the skill driver's inline smoke assertions.
"""
from rbac_benchmark.core.config import InferenceMetrics, Outcome, BENIGN_CONTROL_KEYS
from rbac_benchmark.evaluation.analyzer import compute_delta_tpr, validate_attack_strength


def test_inference_metrics_accounting():
    m = InferenceMetrics()
    for o in (Outcome.SEVERITY_1, Outcome.COMPLIANT, Outcome.FALSE_POSITIVE):
        m.record(o)
    # All three primary outcomes count toward the denominator.
    assert m.total_inferences == 3
    assert "I0_BENIGN_CONTROL" in BENIGN_CONTROL_KEYS


def test_psychological_vectors_not_in_denominator():
    m = InferenceMetrics()
    m.record(Outcome.SEVERITY_3)
    m.record(Outcome.AUTHORITY_BIAS)  # secondary annotation — must not inflate the total
    assert m.total_inferences == 1


def test_delta_tpr_positive_for_hardened_defense(results_file):
    delta = compute_delta_tpr(results_path=results_file, ref_model="qwen3.5:9b",
                              baseline_defense="S1_BASELINE")
    # The hardened defense improves immunity vs the baseline on the authority attack.
    assert delta["I2_AUTHORITY_APPEAL"]["S3_RBAC_HARDENED"]["delta"] > 0


def test_validate_attack_strength(results_file):
    validity = validate_attack_strength(results_path=results_file, ref_model="qwen3.5:9b",
                                        defense_key="S1_BASELINE", threshold=0.10)
    # I3 fully breaks the undefended ref model (TPR 0%) -> valid; I2 leaks 20% -> weak.
    assert validity["I3_CRITICAL_COERCION"]["valid"] is True
    assert validity["I2_AUTHORITY_APPEAL"]["valid"] is False
    # Benign control is excluded from attack-strength scoring.
    assert "I0_BENIGN_CONTROL" not in validity


def test_json_utils_strip_fences():
    from rbac_benchmark.llm.json_utils import strip_code_fences, parse_json_response
    assert strip_code_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert parse_json_response('```\n{"b": 2}\n```') == {"b": 2}
