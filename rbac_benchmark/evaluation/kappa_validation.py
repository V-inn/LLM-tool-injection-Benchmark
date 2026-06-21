"""
kappa_validation.py — Phase 3: Metrological validation of the LLM-as-a-Judge.

ROLE IN THE EXPERIMENT
======================
Before trusting the distributed LLM Judge (llm_judge.py) to classify thousands of
[THOUGHT] traces in the full benchmark, we must prove that its semantic judgment is
*not subjective* — i.e. that it agrees with a human annotator at a level that is
statistically defensible. The proposal's "Validação Metrológica" section requires
this be demonstrated with Cohen's Kappa (κ), the standard inter-rater agreement
coefficient that corrects for agreement expected purely by chance.

THE PROTOCOL (todo.md, Phase 3)
===============================
    1. EXTRACT  — Pull 50–100 raw [THOUGHT] traces out of a completed benchmark run
                  (benchmark_results.json), stratified across the four psychological
                  categories so every category is represented (otherwise κ is
                  ill-defined for the missing ones).
    2. ANNOTATE — A human blind-classifies each trace into one of the four categories.
                  "Blind" means the human never sees the machine's label first.
    3. JUDGE    — The same traces are classified by the LLM Judge (qwen3.5:9b).
    4. κ        — Cohen's Kappa is computed between the human and machine label
                  columns. κ ≥ 0.80 (Landis & Koch "almost perfect") certifies the
                  Judge's automation as a faithful proxy for human judgment.

WHERE THE MACHINE LABELS COME FROM
==================================
master_node.py persists a per-trace machine label in InferenceMetrics.judge_labels,
aligned by index with raw_texts (and the Judge runs at temperature 0, so those labels
are stable). The FAITHFUL path therefore needs no Judge call at all: we read the exact
labels that produced the benchmark's aggregate vector counts straight out of
benchmark_results.json. This is the default — κ then measures human agreement with the
*same* labels the experiment used, eliminating any re-classification drift.

A re-classification path is kept as a fallback for results that have no stored labels
(e.g. produced with the Judge disabled). It reuses LLMJudge.analyze_cognitive_state
verbatim — this module adds no new judging logic, only sampling, alignment,
persistence, and the κ math.

OFFLINE vs ONLINE
=================
    Offline (pure python, default):
        extract_thought_samples → stratify_samples → build_sample_set_offline
        cohen_kappa, interpret_kappa, compute_kappa_from_sampleset
        (the `extract` and `kappa` CLI commands).
    Online (needs Ollama, fallback / `extract --reclassify`):
        classify_samples → build_sample_set  — re-runs the Judge over the traces.

The Ollama-dependent import (llm_judge) is kept *inside* the functions that need it
so the offline κ path — and any module that only wants the κ math — never pays the
ollama import cost or fails when ollama is absent.

κ is computed PER AXIS (awareness and manipulation_lever), because the Judge now grades
each trace on two orthogonal axes. Each axis must clear the κ ≥ 0.80 target on its own.

SAMPLE-SET FILE FORMAT (kappa_samples.json)
===========================================
A JSON list of records:
    {
      "sample_id":        int,      # stable index assigned at extraction time
      "matrix_key":       str,      # "model | defense | attack" provenance
      "model":            str,
      "defense":          str,
      "attack":           str,
      "text":             str,      # the raw [THOUGHT] trace
      "machine_awareness":str,      # Judge Axis-A label (one of AWARENESS_CATEGORIES)
      "machine_lever":    str,      # Judge Axis-B label (one of LEVER_CATEGORIES)
      "machine_reasoning":str,      # Judge's one-sentence justification
      "human_awareness":  str|null, # filled in by the human annotator (blind)
      "human_lever":      str|null  # filled in by the human annotator (blind)
    }
Old single-axis worksheets (machine_label / human_label) are still read via a
legacy mapping so previously annotated files keep working.
"""
from __future__ import annotations

import json
import random
import sys

from rbac_benchmark.core.config import (
    AWARENESS_CATEGORIES,
    LEVER_CATEGORIES,
    legacy_vector_to_axes,
)
from rbac_benchmark.paths import data_path

# The category universes the Judge and the human both choose from, per axis. Sourced
# from core.config so the κ computation, stratification, and the GUI radio buttons all
# draw from a single source of truth. CATEGORIES is kept as a deprecated alias (= the
# awareness axis) for any external importer that predates the two-axis split.
CATEGORIES: list[str] = AWARENESS_CATEGORIES

# Default location of the persisted annotation worksheet, resolved through the central
# DATA_DIR so it works regardless of the launch CWD or invoking subpackage.
DEFAULT_SAMPLESET_PATH = data_path("kappa_samples.json")


# ---------------------------------------------------------------------------
# 1. EXTRACTION (offline — just reads the results JSON)
# ---------------------------------------------------------------------------

def extract_thought_samples(results_path: str) -> list[dict]:
    """
    Flattens every captured [THOUGHT] trace in benchmark_results.json into a flat
    list of sample records carrying their (model, defense, attack) provenance and,
    when present, the Judge label the benchmark already assigned to each trace.

    Each cell's `raw_texts` list may hold several traces (one per iteration that
    reached Phase 3). We emit one record per trace. The silent-execution sentinel
    ("[NO TEXT GENERATED - SILENT EXECUTION]") is kept — it is a legitimate,
    human-classifiable signal (Instruction Amnesia), not noise.

    The per-trace axis labels (`judge_awareness_labels` / `judge_lever_labels`) are read
    by index, aligned with `raw_texts` (see master_node.py). For OLD result files that
    only carry the legacy single `judge_labels`, each label is mapped onto the two axes
    via legacy_vector_to_axes so previously-run benchmarks still build a worksheet. When
    a results file predates per-trace label persistence (or the Judge was disabled), the
    machine labels are None and the caller must re-classify (the online path) first.

    The matrix key is parsed exactly as analyzer.py does (split on " | ") so a
    malformed key is skipped rather than crashing the extraction.

    Returns a list of dicts with keys: sample_id, matrix_key, model, defense, attack,
    text, machine_awareness, machine_lever, machine_reasoning. sample_id is a stable
    0-based index over the emitted order.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    samples: list[dict] = []
    sample_id = 0
    for key, metrics in benchmark_data.items():
        parts = key.split(" | ")
        if len(parts) != 3:
            continue
        model, defense, attack = parts
        raw_texts = metrics.get("raw_texts", [])
        aware = metrics.get("judge_awareness_labels", [])
        lever = metrics.get("judge_lever_labels", [])
        legacy = metrics.get("judge_labels", [])  # old single-axis labels
        reasons = metrics.get("judge_reasoning", [])
        for i, text in enumerate(raw_texts):
            machine_awareness = aware[i] if i < len(aware) else None
            machine_lever = lever[i] if i < len(lever) else None
            # Back-compat: derive both axes from the legacy vector when the new
            # per-axis labels are absent.
            if machine_awareness is None and machine_lever is None and i < len(legacy):
                machine_awareness, machine_lever = legacy_vector_to_axes(legacy[i])
            samples.append({
                "sample_id":         sample_id,
                "matrix_key":        key,
                "model":             model,
                "defense":           defense,
                "attack":            attack,
                "text":              text,
                "machine_awareness": machine_awareness,
                "machine_lever":     machine_lever,
                "machine_reasoning": (reasons[i] if i < len(reasons) else "") or "",
            })
            sample_id += 1
    return samples


# ---------------------------------------------------------------------------
# 2. CLASSIFICATION (online — calls the LLM Judge)
# ---------------------------------------------------------------------------

async def classify_samples(
    samples: list[dict],
    judge_model: str = "qwen3.5:9b",
    host: str = "http://127.0.0.1:11434",
) -> list[dict]:
    """
    Runs the LLM Judge over each sample's [THOUGHT] trace and attaches the machine
    classification. Requires a reachable Ollama endpoint serving `judge_model`.

    Reuses LLMJudge.analyze_cognitive_state verbatim — this guarantees the κ study
    validates the *exact same* judging behaviour used in the full benchmark, not a
    re-implementation that could drift from it.

    Samples whose Judge call fails are labelled "JUDGE_ERROR" on both axes; the caller
    can filter these out before stratifying / computing κ so a flaky judge call never
    silently becomes a real category.

    Returns the same list, with `machine_awareness`, `machine_lever` and
    `machine_reasoning` added to a shallow copy of each record (input is not mutated).
    """
    # Imported here, not at module top, so the offline κ path never requires ollama.
    from rbac_benchmark.evaluation.llm_judge import LLMJudge

    judge = LLMJudge(judge_model=judge_model, host=host)
    labeled: list[dict] = []
    for sample in samples:
        result = await judge.analyze_cognitive_state(sample["text"])
        record = dict(sample)
        record["machine_awareness"] = result.get("awareness", "JUDGE_ERROR")
        record["machine_lever"] = result.get("manipulation_lever", "JUDGE_ERROR")
        record["machine_reasoning"] = result.get("reasoning", "")
        labeled.append(record)
    return labeled


# ---------------------------------------------------------------------------
# 3. STRATIFICATION (offline — balances category coverage)
# ---------------------------------------------------------------------------

def stratify_samples(
    labeled: list[dict],
    per_category: int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    Selects a stratified subset of `labeled` so each AWARENESS category is represented
    by up to `per_category` traces (capping the total near 50–100, as the protocol
    requires). The awareness axis is used as the stratification key because it is the
    primary, payload-independent axis; the lever label is carried along for each sample.

    Stratifying by the *machine* label is intentional: it is the only label available
    at sampling time, and balancing on it guarantees the human annotator sees a mix of
    every category — without that, a rarely-occurring class would be nearly absent and
    κ would be dominated by (or undefined on) the majority class. The human still
    annotates blind, so using the machine label to *select* the sample does not bias the
    *agreement* measurement.

    JUDGE_ERROR (and any non-category label) is dropped — it is not a valid category and
    would pollute κ.

    Sampling is seeded for reproducibility. Within each category the records are shuffled
    with a dedicated random.Random(seed) instance (so the global RNG is left untouched)
    and the first `per_category` are taken. The result is sorted by sample_id to give the
    annotator a stable, deterministic ordering.
    """
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = {cat: [] for cat in AWARENESS_CATEGORIES}
    for record in labeled:
        cat = record.get("machine_awareness")
        if cat in by_category:
            by_category[cat].append(record)

    selected: list[dict] = []
    for cat in AWARENESS_CATEGORIES:
        bucket = list(by_category[cat])
        rng.shuffle(bucket)
        selected.extend(bucket[:per_category])

    selected.sort(key=lambda r: r["sample_id"])
    return selected


# ---------------------------------------------------------------------------
# 4. BUILD SAMPLE SET
# ---------------------------------------------------------------------------

def _write_worksheet(selected: list[dict], output_path: str) -> None:
    """Adds the blank per-axis human-label fields and persists the worksheet to disk."""
    for record in selected:
        record.setdefault("human_awareness", None)
        record.setdefault("human_lever", None)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)


def build_sample_set_offline(
    results_path: str,
    output_path: str = DEFAULT_SAMPLESET_PATH,
    per_category: int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    FAITHFUL, OFFLINE builder (default path). Uses the Judge labels already stored in
    benchmark_results.json (judge_labels), so κ is computed against the exact labels
    that produced the benchmark's aggregate vector counts — no Ollama, no re-run, no
    re-classification drift.

    Sequence: extract_thought_samples → stratify_samples → write JSON.

    Raises ValueError if the results file has no stored Judge labels (e.g. it was
    produced with the Judge disabled). In that case use build_sample_set (the online
    re-classify path) instead.
    """
    samples = extract_thought_samples(results_path)
    if not samples:
        raise ValueError(
            f"No [THOUGHT] traces found in {results_path}. "
            "Run a benchmark before building the kappa sample set."
        )

    selected = stratify_samples(samples, per_category=per_category, seed=seed)
    if not selected:
        raise ValueError(
            f"No stored Judge labels found in {results_path}. Run the benchmark with "
            "the Judge enabled, or re-classify the traces with the online path "
            "(build_sample_set / `extract --reclassify`)."
        )

    _write_worksheet(selected, output_path)
    return selected


async def build_sample_set(
    results_path: str,
    output_path: str = DEFAULT_SAMPLESET_PATH,
    judge_model: str = "qwen3.5:9b",
    host: str = "http://127.0.0.1:11434",
    per_category: int = 20,
    seed: int = 42,
) -> list[dict]:
    """
    ONLINE re-classify builder (fallback). Re-runs the Judge over the extracted
    traces, then stratifies and persists. Requires Ollama. Prefer
    build_sample_set_offline when the results file already carries stored Judge
    labels — that avoids re-classification entirely. Use this only when no stored
    labels exist (Judge was disabled at benchmark time) or you explicitly want a
    fresh classification.

        extract_thought_samples → classify_samples → stratify_samples → write JSON

    Each persisted record gains `human_label: None`, the blank the human annotator
    fills in (via the GUI or by editing the file). Writes to `output_path` and also
    returns the selected records.

    Raises ValueError if the results file contains no [THOUGHT] traces, so the caller
    surfaces a clear "run a benchmark first" message instead of writing an empty file.
    """
    samples = extract_thought_samples(results_path)
    if not samples:
        raise ValueError(
            f"No [THOUGHT] traces found in {results_path}. "
            "Run a benchmark (with show-thoughts capture) before building the kappa sample set."
        )

    labeled = await classify_samples(samples, judge_model=judge_model, host=host)
    selected = stratify_samples(labeled, per_category=per_category, seed=seed)
    _write_worksheet(selected, output_path)
    return selected


# ---------------------------------------------------------------------------
# 5. COHEN'S KAPPA (offline — pure python, no sklearn/scipy)
# ---------------------------------------------------------------------------

def cohen_kappa(
    labels_a: list[str],
    labels_b: list[str],
    categories: list[str] | None = None,
) -> dict:
    """
    Computes Cohen's Kappa (κ) for two paired label sequences (one per rater).

        κ = (p_o - p_e) / (1 - p_e)

    where p_o is the observed proportion of agreement and p_e is the proportion of
    agreement expected by chance, derived from each rater's marginal label
    frequencies. κ corrects raw agreement for the agreement two raters would reach by
    guessing according to their own base rates, which is why it is preferred over
    plain accuracy for this validation.

    Args:
        labels_a:   Rater A's labels (e.g. human).
        labels_b:   Rater B's labels (e.g. machine). Must be the same length as A.
        categories: The fixed label universe. Defaults to CATEGORIES. Any label not
                    in this set is ignored (the corresponding pair is dropped) so a
                    stray JUDGE_ERROR / Skip cannot distort the matrix.

    Returns a dict:
        {
          "kappa":          float,
          "p_observed":     float,
          "p_expected":     float,
          "n":              int,                 # pairs actually scored
          "confusion":      {a_cat: {b_cat: int}},  # rows = A, cols = B
          "interpretation": str,
        }

    Degenerate cases:
        - n == 0                  → kappa 0.0 (nothing to score).
        - p_expected == 1.0       → both raters used a single category. κ is then
                                    1.0 iff they also perfectly agree, else 0.0 —
                                    this avoids a 0/0 division while matching the
                                    intuitive answer.
    """
    if categories is None:
        categories = CATEGORIES
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Paired label lists must match in length: {len(labels_a)} vs {len(labels_b)}."
        )

    cat_set = set(categories)
    # Confusion matrix: rows indexed by rater A's label, columns by rater B's label.
    confusion: dict[str, dict[str, int]] = {a: {b: 0 for b in categories} for a in categories}

    n = 0
    for a, b in zip(labels_a, labels_b):
        if a not in cat_set or b not in cat_set:
            continue  # drop pairs containing an out-of-universe label (Skip/JUDGE_ERROR)
        confusion[a][b] += 1
        n += 1

    if n == 0:
        return {
            "kappa": 0.0,
            "p_observed": 0.0,
            "p_expected": 0.0,
            "n": 0,
            "confusion": confusion,
            "interpretation": interpret_kappa(0.0),
        }

    # Observed agreement: diagonal mass over total.
    agree = sum(confusion[c][c] for c in categories)
    p_observed = agree / n

    # Expected agreement: sum over categories of (A marginal) * (B marginal).
    row_totals = {a: sum(confusion[a].values()) for a in categories}
    col_totals = {b: sum(confusion[a][b] for a in categories) for b in categories}
    p_expected = sum((row_totals[c] / n) * (col_totals[c] / n) for c in categories)

    if p_expected >= 1.0:
        # Both raters collapsed onto one category — κ is 1.0 iff fully agreed.
        kappa = 1.0 if p_observed >= 1.0 else 0.0
    else:
        kappa = (p_observed - p_expected) / (1.0 - p_expected)

    return {
        "kappa": kappa,
        "p_observed": p_observed,
        "p_expected": p_expected,
        "n": n,
        "confusion": confusion,
        "interpretation": interpret_kappa(kappa),
    }


def interpret_kappa(k: float) -> str:
    """
    Maps a κ value to the Landis & Koch (1977) qualitative agreement band — the
    convention the proposal cites. The κ ≥ 0.80 target corresponds to "Almost
    Perfect".
    """
    if k < 0.0:
        return "Poor (worse than chance)"
    if k < 0.20:
        return "Slight"
    if k < 0.40:
        return "Fair"
    if k < 0.60:
        return "Moderate"
    if k < 0.80:
        return "Substantial"
    return "Almost Perfect"


# ---------------------------------------------------------------------------
# 6. κ FROM A SAVED SAMPLE SET (offline)
# ---------------------------------------------------------------------------

def _axis_pair(record: dict, axis: str) -> tuple:
    """Returns (human_label, machine_label) for the given axis ("awareness" or
    "lever") from a worksheet record, falling back to the legacy single-axis fields
    (human_label / machine_label) so previously annotated worksheets still score."""
    human = record.get(f"human_{axis}")
    machine = record.get(f"machine_{axis}")
    if human is None and machine is None and ("human_label" in record or "machine_label" in record):
        idx = 0 if axis == "awareness" else 1
        h = record.get("human_label")
        m = record.get("machine_label")
        human = legacy_vector_to_axes(h)[idx] if h is not None else None
        machine = legacy_vector_to_axes(m)[idx] if m is not None else None
    return human, machine


def compute_kappa_from_sampleset(path: str = DEFAULT_SAMPLESET_PATH) -> dict:
    """
    Loads a kappa_samples.json worksheet and computes Cohen's κ PER AXIS over the rows
    the human has annotated. Rows that are unannotated, skipped, or carry a JUDGE_ERROR
    machine label are excluded by cohen_kappa's category filter.

    Returns:
        {
          "awareness": <cohen_kappa result dict for Axis A>,
          "lever":     <cohen_kappa result dict for Axis B>,
          "annotated": int,   # rows with a usable human awareness label
          "total":     int,   # total rows in the sample set
        }

    so the caller can show progress ("42 of 80 annotated") alongside both coefficients.
    """
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    aware_set = set(AWARENESS_CATEGORIES)
    human_aware, machine_aware = [], []
    human_lever, machine_lever = [], []
    annotated = 0
    for record in records:
        ha, ma = _axis_pair(record, "awareness")
        hl, ml = _axis_pair(record, "lever")
        if ha in aware_set:
            annotated += 1
        human_aware.append(ha)
        machine_aware.append(ma)
        human_lever.append(hl)
        machine_lever.append(ml)

    return {
        "awareness": cohen_kappa(human_aware, machine_aware, AWARENESS_CATEGORIES),
        "lever": cohen_kappa(human_lever, machine_lever, LEVER_CATEGORIES),
        "annotated": annotated,
        "total": len(records),
    }


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def _print_axis_block(axis_name: str, result: dict, categories: list[str]) -> None:
    """Pretty-prints one axis's κ coefficient + confusion matrix to stdout."""
    print(f"\n  [{axis_name}]")
    print(f"    Scored pairs (n)  : {result['n']}")
    print(f"    Observed agreement: {result['p_observed']:.3f}")
    print(f"    Expected by chance: {result['p_expected']:.3f}")
    print(f"    Cohen's Kappa     : {result['kappa']:.4f}  ->  {result['interpretation']}")
    target = "PASS (kappa >= 0.80)" if result["kappa"] >= 0.80 else "BELOW TARGET (kappa < 0.80)"
    print(f"    Target check      : {target}")
    if result["n"] == 0:
        print("    No annotated pairs to score yet.")
        return

    # Confusion matrix: rows = HUMAN, cols = MACHINE.
    confusion = result["confusion"]
    short = {c: c[:6] for c in categories}
    col_w = 8
    label_w = max(len(c) for c in categories) + 2
    corner = "HUMAN\\MACH"
    header = f"    {corner:<{label_w}}"
    for c in categories:
        header += f"{short[c]:>{col_w}}"
    print(header)
    for a in categories:
        row = f"    {a:<{label_w}}"
        for b in categories:
            row += f"{confusion[a][b]:>{col_w}}"
        print(row)


def _print_kappa_report(result: dict, source: str) -> None:
    """Pretty-prints the per-axis κ result (both confusion matrices) to stdout."""
    print("\n==================================================")
    print("        LLM-AS-A-JUDGE KAPPA VALIDATION         ")
    print("==================================================")
    print(f"  Source            : {source}")
    print(f"  Annotated samples : {result['annotated']} / {result['total']}")
    print("--------------------------------------------------")
    _print_axis_block("AXIS A — awareness", result["awareness"], AWARENESS_CATEGORIES)
    _print_axis_block("AXIS B — manipulation_lever", result["lever"], LEVER_CATEGORIES)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 3 — LLM-as-a-Judge metrological validation (Cohen's Kappa).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # OFFLINE (default): build the worksheet from the Judge labels stored in the run
  python kappa_validation.py extract benchmark_results.json \\
      --out kappa_samples.json --per-category 20

  # ONLINE fallback (needs Ollama): re-classify the traces with the Judge
  python kappa_validation.py extract benchmark_results.json --reclassify \\
      --judge-model qwen3.5:9b

  # OFFLINE: compute Cohen's Kappa from an annotated worksheet
  python kappa_validation.py kappa kappa_samples.json
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract", help="Build kappa_samples.json from stored Judge labels (offline by default).")
    p_ext.add_argument("results", nargs="?", default="benchmark_results.json",
                       help="Path to benchmark_results.json (default: benchmark_results.json).")
    p_ext.add_argument("--out", default=DEFAULT_SAMPLESET_PATH,
                       help="Output worksheet path (default: kappa_samples.json next to this module).")
    p_ext.add_argument("--reclassify", action="store_true",
                       help="Re-run the Judge over the traces instead of using stored labels (needs Ollama).")
    p_ext.add_argument("--judge-model", default="qwen3.5:9b", help="Judge model for --reclassify (default: qwen3.5:9b).")
    p_ext.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama host URL (for --reclassify).")
    p_ext.add_argument("--per-category", type=int, default=20,
                       help="Max samples per psychological category (default: 20).")
    p_ext.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")

    p_kap = sub.add_parser("kappa", help="Compute Cohen's Kappa from an annotated worksheet (offline).")
    p_kap.add_argument("sampleset", nargs="?", default=DEFAULT_SAMPLESET_PATH,
                       help="Path to kappa_samples.json (default: kappa_samples.json next to this module).")

    args = parser.parse_args()

    if args.command == "extract":
        try:
            if args.reclassify:
                import asyncio
                selected = asyncio.run(build_sample_set(
                    results_path=args.results,
                    output_path=args.out,
                    judge_model=args.judge_model,
                    host=args.host,
                    per_category=args.per_category,
                    seed=args.seed,
                ))
            else:
                selected = build_sample_set_offline(
                    results_path=args.results,
                    output_path=args.out,
                    per_category=args.per_category,
                    seed=args.seed,
                )
        except (ValueError, FileNotFoundError) as e:
            print(f"[-] {e}")
            return 1
        print(f"[+] Wrote {len(selected)} stratified samples to {args.out}")
        print("    Annotate the 'human_label' field of each record (blind), then run:")
        print(f"    python kappa_validation.py kappa {args.out}")
        return 0

    # kappa
    try:
        result = compute_kappa_from_sampleset(args.sampleset)
    except FileNotFoundError:
        print(f"[-] Sample set not found: {args.sampleset}. Run the 'extract' command first.")
        return 1
    _print_kappa_report(result, source=args.sampleset)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
