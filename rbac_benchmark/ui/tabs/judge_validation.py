"""
judge_validation.py — "Judge Validation (κ)" tab (Phase 3).

Three sections: build the annotation worksheet (from stored Judge labels offline, or by
re-running the Judge), blind human annotation, and the Cohen's-κ result + confusion
matrix. See rbac_benchmark.evaluation.kappa_validation for the underlying logic.
"""
import os
import json
import asyncio

import streamlit as st

from rbac_benchmark.evaluation.kappa_validation import (
    CATEGORIES,
    build_sample_set,
    build_sample_set_offline,
    compute_kappa_from_sampleset,
)
from rbac_benchmark.ui import charts


def _results_have_stored_labels(path: str) -> bool:
    """True if the results file carries per-trace Judge labels (so κ can be computed
    offline against the exact labels the benchmark used)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    return any(
        lbl in CATEGORIES
        for cell in data.values()
        for lbl in cell.get("judge_labels", [])
    )


def render(ctx):
    results_file = ctx.results_file
    kappa_samples_file = ctx.kappa_samples_file
    available_models = ctx.available_models
    online = ctx.online

    st.header("Phase 3 — LLM-as-a-Judge Metrological Validation")
    st.markdown(
        "Proves the Judge is **not subjective** by measuring its agreement with a "
        "human annotator via **Cohen's Kappa (κ)**. Workflow: build a stratified "
        "sample of `[THOUGHT]` traces (labelled by the Judge during the run), classify "
        "each one **blind** yourself, then read κ. Target: **κ ≥ 0.80** (Landis & Koch "
        "“Almost Perfect”)."
    )

    # ----------------------------------------------------------------
    # Section 1 — Build the annotation sample set
    # ----------------------------------------------------------------
    st.subheader("1 · Build Annotation Sample Set")
    st.write(
        "Extracts `[THOUGHT]` traces from the latest `benchmark_results.json`, keeps "
        "up to *N per category* so every psychological vector is represented, and "
        "writes the worksheet. **Preferred:** reuse the Judge labels stored during the "
        "run — κ is then measured against the *exact* labels that fed the metrics, with "
        "no non-deterministic re-run."
    )

    results_exists = os.path.exists(results_file)
    has_stored = results_exists and _results_have_stored_labels(results_file)

    # Default to the faithful offline path when stored labels exist; otherwise the
    # only way to get labels is to re-run the Judge (needs Ollama).
    reclassify = st.checkbox(
        "Re-run the Judge instead of using stored labels (non-deterministic; requires Ollama)",
        value=not has_stored,
        key="kappa_reclassify",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if reclassify:
            if available_models:
                k_default_idx = available_models.index("qwen3.5:9b") if "qwen3.5:9b" in available_models else 0
                kappa_judge_model = st.selectbox("Judge Model", available_models, index=k_default_idx, key="kappa_judge_model")
            else:
                kappa_judge_model = st.text_input("Judge Model", value="qwen3.5:9b", key="kappa_judge_model")
        else:
            kappa_judge_model = "qwen3.5:9b"
            st.caption("Using stored Judge labels (no model needed).")
    with col_b:
        kappa_per_category = st.number_input("Samples per category", min_value=1, max_value=100, value=20, key="kappa_per_cat")
    with col_c:
        kappa_seed_val = st.number_input("Sampling seed", min_value=0, value=42, key="kappa_seed_val")

    if not results_exists:
        st.info("No `benchmark_results.json` found. Run a benchmark in the Control Center first.")
        build_disabled = True
        build_label = "Build Sample Set"
    elif reclassify:
        build_disabled = not online
        build_label = "Build Sample Set (re-run Judge)"
        if not online:
            st.info("**Ollama offline** — can't re-run the Judge. Untick the box to use stored labels instead.")
    else:
        build_disabled = not has_stored
        build_label = "Build Sample Set (from stored labels)"
        if not has_stored:
            st.info(
                "This results file has **no stored Judge labels** (was the Judge enabled for the run?). "
                "Tick *Re-run the Judge* to classify now, or re-run the benchmark with the Judge on."
            )

    if st.button(build_label, type="primary", disabled=build_disabled):
        with st.spinner("Building sample set..."):
            try:
                if reclassify:
                    selected = asyncio.run(build_sample_set(
                        results_path=results_file,
                        output_path=kappa_samples_file,
                        judge_model=kappa_judge_model,
                        per_category=int(kappa_per_category),
                        seed=int(kappa_seed_val),
                    ))
                else:
                    selected = build_sample_set_offline(
                        results_path=results_file,
                        output_path=kappa_samples_file,
                        per_category=int(kappa_per_category),
                        seed=int(kappa_seed_val),
                    )
                st.session_state.kappa_annotation_index = 0
                st.success(f"Built **{len(selected)}** stratified samples → `kappa_samples.json`.")
            except Exception as e:
                st.error(f"Failed to build sample set: {e}")

    st.divider()

    # ----------------------------------------------------------------
    # Section 2 — Blind human annotation
    # ----------------------------------------------------------------
    st.subheader("2 · Blind Human Annotation")

    if not os.path.exists(kappa_samples_file):
        st.info("No sample set yet. Build one above, or place a `kappa_samples.json` in the data directory.")
    else:
        with open(kappa_samples_file, "r", encoding="utf-8") as f:
            kappa_samples = json.load(f)

        total = len(kappa_samples)
        annotated = sum(1 for s in kappa_samples if s.get("human_label") in CATEGORIES)

        if total == 0:
            st.warning("The sample set is empty.")
        else:
            st.progress(annotated / total, text=f"{annotated} / {total} annotated")

            # Clamp the current index into range (the file may have changed/shrunk).
            idx = max(0, min(st.session_state.kappa_annotation_index, total - 1))
            st.session_state.kappa_annotation_index = idx
            sample = kappa_samples[idx]

            st.caption(
                f"Sample **{idx + 1} of {total}** · provenance: `{sample.get('matrix_key', '?')}` "
                "· the Judge's label is hidden to keep your annotation blind."
            )
            st.code(sample.get("text") or "[NO TEXT]", language=None)

            existing = sample.get("human_label")
            default_idx = CATEGORIES.index(existing) if existing in CATEGORIES else None
            choice = st.radio(
                "Your classification (blind):",
                CATEGORIES,
                index=default_idx,
                key=f"kappa_radio_{idx}",
                horizontal=True,
            )

            nav_prev, nav_save, nav_next = st.columns(3)
            with nav_prev:
                if st.button("◀ Previous", use_container_width=True, disabled=idx == 0):
                    st.session_state.kappa_annotation_index = idx - 1
                    st.rerun()
            with nav_save:
                if st.button("💾 Save & Next ▶", type="primary", use_container_width=True, disabled=choice is None):
                    kappa_samples[idx]["human_label"] = choice
                    with open(kappa_samples_file, "w", encoding="utf-8") as f:
                        json.dump(kappa_samples, f, indent=2, ensure_ascii=False)
                    # Jump to the next still-unannotated sample, else just step forward.
                    nxt = next(
                        (i for i in range(idx + 1, total) if kappa_samples[i].get("human_label") not in CATEGORIES),
                        min(idx + 1, total - 1),
                    )
                    st.session_state.kappa_annotation_index = nxt
                    st.rerun()
            with nav_next:
                if st.button("Skip ▶", use_container_width=True, disabled=idx >= total - 1):
                    st.session_state.kappa_annotation_index = idx + 1
                    st.rerun()

    st.divider()

    # ----------------------------------------------------------------
    # Section 3 — Cohen's Kappa result
    # ----------------------------------------------------------------
    st.subheader("3 · Cohen's Kappa Result")

    if not os.path.exists(kappa_samples_file):
        st.info("Build and annotate a sample set first.")
    else:
        kappa_result = compute_kappa_from_sampleset(kappa_samples_file)
        if kappa_result["n"] == 0:
            st.info(f"{kappa_result['annotated']} / {kappa_result['total']} samples annotated — annotate above to compute κ.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Cohen's κ", f"{kappa_result['kappa']:.3f}")
            m2.metric("Agreement", kappa_result["interpretation"])
            m3.metric("Scored pairs (n)", kappa_result["n"])

            if kappa_result["kappa"] >= 0.80:
                st.success("**κ ≥ 0.80** — Almost Perfect agreement. The Judge is a faithful proxy for human annotation.")
            else:
                st.warning("**κ < 0.80** — below target. Refine the Judge rubric or annotate more samples before trusting the automation.")

            st.caption(
                f"Observed agreement {kappa_result['p_observed']:.3f} vs chance-expected "
                f"{kappa_result['p_expected']:.3f}. Confusion matrix below: rows = your "
                "labels, columns = the Judge; the diagonal is agreement."
            )

            st.plotly_chart(charts.confusion_matrix_heatmap(kappa_result["confusion"], CATEGORIES), use_container_width=True)
