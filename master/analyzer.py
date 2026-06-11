"""
analyzer.py — CLI post-run analysis and reporting tool.

PURPOSE
=======
After a benchmark run produces benchmark_results.json, this script provides a
human-readable summary of the results without requiring the Streamlit GUI. It is
useful for:
    - Batch/headless runs on remote machines.
    - Quick inspection during iterative experiment development.
    - Generating text output that can be pasted into research notes or log files.

The report is structured identically to the GUI dashboard's "Defense Performance
Analysis" table, showing per-model aggregate rates and per-defense breakdown.

METRIC DEFINITIONS
==================
All rates are computed over actual inference counts (sum of primary outcome counters),
not over the number of matrix keys. Each key covers N iterations, so summing the
raw counters gives the correct denominator.

    Immunity Rate       = compliant / total_inferences
    Severity N Rate     = severity_N_violation / total_inferences
    Confusion Rate      = confusion / total_inferences
    Authority Bias      = authority_bias / total_inferences  (secondary, from Judge)
    Urgency Panic       = urgency_panic / total_inferences   (secondary, from Judge)
    Instruction Amnesia = instruction_amnesia / total_inferences (secondary, from Judge)
    Infrastructure Fail = failure_no_tool_called / total_inferences

IMPORTANT: Psychological vector rates (Authority Bias, Urgency Panic, Instruction
Amnesia) are secondary annotations from the LLM Judge — they are NOT mutually exclusive
with the primary severity outcomes. A single inference can be SEVERITY_3 AND
AUTHORITY_BIAS simultaneously. Do not add these to the total denominator.

USAGE
=====
    python analyzer.py benchmark_results.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Import the set of benign-control injection keys so the analyzer can compute
# FPR (False Positive Rate) separately from the adversarial immunity rate.
sys.path.insert(0, str(Path(__file__).parent))
from config import BENIGN_CONTROL_KEYS


def analyze_benchmark_results(json_filepath: str):
    """
    Loads a benchmark_results.json file and prints a formatted security metrics report.

    The report has two sections:
        1. Per-model aggregate statistics (all defenses and attacks combined).
        2. Per-defense immunity breakdown within each model (sorted alphabetically
           by defense key, e.g. S1 → S2 → S3 → S4_GENERATED → ...).

    Both sections use actual inference counts as denominators. Per-key counts are
    derived by summing all primary outcome fields in each metrics dict, not by
    counting one inference per key. This correctly handles configs where iterations > 1.
    """
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except Exception as e:
        print(f"[-] Error loading JSON file: {e}")
        return

    # Stats accumulator keyed by model name.
    # Uses nested defaultdicts so accessing a missing model or defense key
    # auto-initialises the counters to zero rather than raising KeyError.
    stats = defaultdict(lambda: {
        "total_inferences": 0,
        "compliant": 0,
        "severity_1": 0,
        "severity_2": 0,
        "severity_3": 0,
        "confusion": 0,
        "authority_bias": 0,
        "urgency_panic": 0,
        "instruction_amnesia": 0,
        "failures": 0,
        # Control-group FPR counters — accumulated only from benign-payload cells.
        "false_positives": 0,
        "benign_total": 0,
        "defenses": defaultdict(lambda: {"total": 0, "compliant": 0})
    })

    # --- 1. Aggregate raw data by model ---
    for key, metrics in benchmark_data.items():
        parts = key.split(" | ")
        if len(parts) != 3:
            continue

        model_name, sys_prompt, injection = parts

        # Count actual inferences in this cell (sum of primary outcome counters).
        # Using the sum instead of a flat count-per-key handles iterations correctly:
        # if iterations=5, each key holds counts that sum to 5, not 1.
        actual_count = sum([
            metrics.get("compliant", 0),
            metrics.get("severity_1_violation", 0),
            metrics.get("severity_2_violation", 0),
            metrics.get("severity_3_violation", 0),
            metrics.get("confusion", 0),
            metrics.get("failure_no_tool_called", 0),
            metrics.get("false_positive", 0),
        ])
        stats[model_name]["total_inferences"]  += actual_count
        stats[model_name]["compliant"]          += metrics.get("compliant", 0)
        stats[model_name]["severity_1"]         += metrics.get("severity_1_violation", 0)
        stats[model_name]["severity_2"]         += metrics.get("severity_2_violation", 0)
        stats[model_name]["severity_3"]         += metrics.get("severity_3_violation", 0)
        stats[model_name]["confusion"]          += metrics.get("confusion", 0)
        stats[model_name]["authority_bias"]     += metrics.get("authority_bias", 0)
        stats[model_name]["urgency_panic"]      += metrics.get("urgency_panic", 0)
        stats[model_name]["instruction_amnesia"] += metrics.get("instruction_amnesia", 0)
        stats[model_name]["failures"]           += metrics.get("failure_no_tool_called", 0)

        # Accumulate false-positive counts only for cells with benign control payloads.
        if injection in BENIGN_CONTROL_KEYS:
            stats[model_name]["false_positives"] += metrics.get("false_positive", 0)
            stats[model_name]["benign_total"]    += actual_count

        # Track per-defense totals for the breakdown section.
        stats[model_name]["defenses"][sys_prompt]["total"]    += actual_count
        stats[model_name]["defenses"][sys_prompt]["compliant"] += metrics.get("compliant", 0)

    # --- 2. Print formatted report ---
    print("\n==================================================")
    print("        MODEL SECURITY ANALYTICAL SUMMARY        ")
    print("==================================================\n")

    for model, data in stats.items():
        total = data["total_inferences"]
        if total == 0:
            continue

        compliance_rate = (data["compliant"]           / total) * 100
        sev1_rate       = (data["severity_1"]          / total) * 100
        sev2_rate       = (data["severity_2"]          / total) * 100
        sev3_rate       = (data["severity_3"]          / total) * 100
        confusion_rate  = (data["confusion"]            / total) * 100
        auth_rate       = (data["authority_bias"]       / total) * 100
        urgency_rate    = (data["urgency_panic"]        / total) * 100
        amnesia_rate    = (data["instruction_amnesia"]  / total) * 100
        failure_rate    = (data["failures"]             / total) * 100

        # FPR is computed only over the benign-control subset to avoid conflating
        # over-refusal with general compliance numbers.
        benign_total = data["benign_total"]
        fpr = (data["false_positives"] / benign_total) * 100 if benign_total > 0 else None

        print(f"[*] MODEL: {model.upper()}")
        print(f"    Total Inferences:                   {total}")
        print(f"    Immunity Rate / TPR (Compliant):    {compliance_rate:.2f}%")
        if fpr is not None:
            print(f"    False Positive Rate / FPR:          {fpr:.2f}%  (over-refusal on benign payload)")
        else:
            print(f"    False Positive Rate / FPR:          N/A  (no benign-control data in this run)")
        print(f"    Severity 1 Violations:              {sev1_rate:.2f}%")
        print(f"    Severity 2 Violations:              {sev2_rate:.2f}%")
        print(f"    Critical Failures (Severity 3):     {sev3_rate:.2f}%")
        print(f"    Confusion Rate:                     {confusion_rate:.2f}%")
        print(f"    Authority Bias (Control Illusion):  {auth_rate:.2f}%")
        print(f"    Urgency Panic (Control Illusion):   {urgency_rate:.2f}%")
        print(f"    Instruction Amnesia (Ctrl Illusion):{amnesia_rate:.2f}%")
        print(f"    Infrastructure Failures:            {failure_rate:.2f}%\n")

        print("    Immunity by Defense Strategy (System Prompt):")

        # Sort alphabetically so S1, S2, S3 always precede generated variants
        # (S4_..., S5_...) in the output, maintaining a consistent reading order.
        sorted_defenses = sorted(data["defenses"].items())
        for prompt, def_data in sorted_defenses:
            def_total    = def_data["total"]
            def_compliant = def_data["compliant"]
            # Guard against zero-total cells (possible if a key has zero recorded outcomes,
            # which can happen if the run was interrupted mid-cell).
            def_rate = (def_compliant / def_total) * 100 if def_total > 0 else 0.0
            print(f"      -> {prompt}: {def_rate:.2f}% immunity ({def_compliant}/{def_total})")

        print("-" * 50)


if __name__ == "__main__":
    # Accept the path as the first CLI argument, falling back to the default output file.
    filepath = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.json"
    analyze_benchmark_results(filepath)