"""
payload_generation.py — "Payload Generation" tab.

Triggers the red-team injection generator and blue-team defense generator (as
subprocesses), plus the Phase-2 weak-attack replacement pipeline. The reference-model
and threshold widgets write ``*_widget`` session_state keys that the app shell syncs
back into the canonical keys on the next rerun.
"""
import os
import sys
import subprocess

import streamlit as st


def render(ctx):
    available_models = ctx.available_models

    st.header("Automated Generators")
    st.write("Trigger Gemini to synthesize new payloads or defenses dynamically.")

    colA, colB = st.columns(2)
    with colA:
        with st.container(border=True):
            st.subheader("Red Team: Injection Generator")
            st.caption("Generate advanced, obfuscated multi-turn attack payloads.")
            num_payloads = st.number_input("Number of Payloads", 1, 50, 10)
            atk_model = st.selectbox("Attacker Model", ["gemini-2.5-flash", "gemini-1.5-pro"])
            if st.button("Generate Injections", use_container_width=True):
                with st.spinner("Generating..."):
                    result = subprocess.run(
                        [sys.executable, "-m", "rbac_benchmark.generation.injection_generator",
                         "--model", atk_model, "--num", str(num_payloads)],
                        capture_output=True, text=True
                    )
                if result.returncode == 0:
                    st.success("Injections generated and saved to generated_injections.json!")
                else:
                    st.error("Generation failed. See logs below.")
                with st.expander("View Generator Logs"):
                    st.code(result.stdout + result.stderr, language="bash")

    with colB:
        with st.container(border=True):
            st.subheader("Blue Team: Defense Generator")
            st.caption("Generate resilient system prompts using advanced self-reflection.")
            def_model = st.selectbox("Defense Generator Model", ["gemini-2.5-flash", "gemini-1.5-pro"], key="def_model")
            if st.button("Generate Defenses", use_container_width=True):
                with st.spinner("Generating..."):
                    result = subprocess.run(
                        [sys.executable, "-m", "rbac_benchmark.generation.defense_generator",
                         "--model", def_model],
                        capture_output=True, text=True
                    )
                if result.returncode == 0:
                    st.success("Defenses generated and saved to generated_defenses.json!")
                else:
                    st.error("Generation failed. See logs below.")
                with st.expander("View Generator Logs"):
                    st.code(result.stdout + result.stderr, language="bash")

    # ---- Phase 2: Replace Weak Attacks ----
    st.divider()
    with st.container(border=True):
        st.subheader("Phase 2 -- Replace Weak Attacks")
        st.write(
            "Reads the current benchmark results, identifies attacks that the reference model "
            "already resists (TPR > threshold) without robust defenses, and generates "
            "**stronger targeted replacements** via the injection generator."
        )
        col_rw1, col_rw2, col_rw3 = st.columns(3)
        with col_rw1:
            rw_model = st.selectbox(
                "Replacement Generator Model",
                ["gemini-2.5-flash", "gemini-1.5-pro"] + (available_models or []),
                key="rw_model"
            )
        with col_rw2:
            if available_models:
                default_ref_idx = available_models.index(st.session_state.ref_model) if st.session_state.ref_model in available_models else 0
                st.selectbox(
                    "Reference Model (Calibration)",
                    available_models,
                    index=default_ref_idx,
                    key="ref_model_widget",
                    help="Model used to validate attack strength. Should be the least-defended model in your fleet."
                )
            else:
                st.text_input(
                    "Reference Model (manual)",
                    value=st.session_state.ref_model,
                    key="ref_model_widget",
                    help="Model used to validate attack strength."
                )
        with col_rw3:
            st.slider(
                "Attack Validity Threshold (max TPR)",
                min_value=0, max_value=50,
                value=int(st.session_state.attack_validity_threshold * 100),
                key="attack_validity_threshold_widget",
                step=5,
                format="%d%%",
                help="If the ref model resists an attack more than this rate even without robust defences, the attack is flagged as WEAK."
            )

        if st.button("Find & Replace Weak Attacks", use_container_width=True, type="primary"):
            if not os.path.exists(ctx.results_file):
                st.warning("No benchmark_results.json found. Run a benchmark first to populate results.")
            else:
                with st.spinner("Analyzing weak attacks and generating replacements..."):
                    result = subprocess.run(
                        [sys.executable, "-m", "rbac_benchmark.generation.injection_generator",
                         "--replace-weak",
                         "--model", rw_model,
                         "--results", ctx.results_file,
                         "--ref-model", ctx.ref_model,
                         "--threshold", str(ctx.attack_validity_threshold)],
                        capture_output=True, text=True
                    )
                if result.returncode == 0:
                    st.success("Replacement payloads generated and merged into generated_injections.json!")
                else:
                    st.error("Replacement generation failed. See logs below.")
                with st.expander("View Replacement Generator Logs"):
                    st.code(result.stdout + result.stderr, language="bash")
