"""
control_center.py — "Control Center" tab: configure, dispatch, and monitor a run.

Spawns the master orchestrator as a subprocess (``python -m
rbac_benchmark.orchestration.master_node``) and streams its stdout into a live log +
progress bar + outcome pie. The subprocess handle lives in session_state so the Abort
button can terminate it across Streamlit reruns.
"""
import os
import sys
import subprocess

import streamlit as st

from rbac_benchmark.core.config import BenchmarkConfig
from rbac_benchmark.core.prompts import load_all_prompts
from rbac_benchmark.ui.data import load_and_parse_results
from rbac_benchmark.ui import charts


def render(ctx):
    st.subheader("Cluster Execution")
    col1, col2 = st.columns([1, 1])
    with col1:
        run_btn = st.button("▶ Run Benchmark", type="primary", use_container_width=True, disabled=st.session_state.benchmark_running)
    with col2:
        abort_btn = st.button("🛑 Abort Benchmark", disabled=not st.session_state.benchmark_running, type="secondary", use_container_width=True)

    # Handle Abort — terminates the stored subprocess handle if it is still alive.
    if abort_btn:
        proc = st.session_state.get("benchmark_process")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        st.session_state.benchmark_process = None
        st.session_state.benchmark_running = False
        st.session_state.run_status = {"type": "warning", "message": "Benchmark manually aborted by user."}
        load_and_parse_results.clear()  # Invalidate stale cache so the results tab shows accurate data
        st.warning(st.session_state.run_status["message"])
        st.stop()

    # Handle Run — validate inputs, then transition to the running state.
    if run_btn:
        # Require at least one model to be selected before dispatching — an empty
        # model list would silently produce a zero-task queue.
        if not ctx.selected_models:
            st.error("Please select at least one model before running the benchmark.")
            st.stop()
        st.session_state.benchmark_running = True
        st.session_state.log_history = []
        st.session_state.live_outcomes = {}
        st.session_state.run_status = {"type": "info", "message": "Initializing cluster components..."}
        st.rerun()

    # Active Benchmark Execution Block
    if st.session_state.benchmark_running:

        run_config = BenchmarkConfig(
            models=ctx.selected_models,
            iterations=ctx.iterations,
            max_turns=ctx.max_turns,
            max_retries=ctx.max_retries,
            timeout=ctx.timeout,
            use_custom_prompts=ctx.use_custom,
            use_generated_injections=ctx.use_gen_inj,
            use_generated_defenses=ctx.use_gen_def,
            use_judge=ctx.use_judge,
            judge_model=ctx.judge_model,
            ref_model=ctx.ref_model,
            attack_validity_threshold=ctx.attack_validity_threshold,
            output=ctx.results_file,
            show_thoughts=ctx.show_thoughts
        )

        run_config.to_json(ctx.temp_config_file)

        # Pre-compute the total inference count here so the progress bar has an
        # accurate denominator before any subprocess output is received. This avoids
        # a mismatch that would occur if master_node loaded a different set of
        # prompts than the GUI expects (e.g. due to a file change between tab loads).
        system_prompts, injection_payloads = load_all_prompts(
            use_custom=ctx.use_custom,
            use_gen_inj=ctx.use_gen_inj,
            use_gen_def=ctx.use_gen_def
        )
        total_inferences = len(ctx.selected_models) * len(system_prompts) * len(injection_payloads) * ctx.iterations

        st.info(f"Target Queue: {total_inferences} inferences scheduled.")

        progress_bar = st.progress(0)
        status_text = st.empty()

        col_logs, col_chart = st.columns([2, 1])
        with col_logs:
            with st.expander("Live Cluster Logs", expanded=True):
                logs_container = st.empty()
        with col_chart:
            live_chart = st.empty()
            st.caption("Live Outcome Distribution")

        process = subprocess.Popen(
            # Use sys.executable (the current Python interpreter) so the subprocess
            # inherits the same virtualenv/conda environment as the GUI process.
            [sys.executable, "-u", "-m", "rbac_benchmark.orchestration.master_node", "--config", ctx.temp_config_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Persist the handle so the Abort button can terminate it in a future rerun.
        st.session_state.benchmark_process = process

        completed_inferences = 0
        is_interrupted = False

        try:
            for line in iter(process.stdout.readline, ''):
                clean_line = line.strip()
                if not clean_line: continue

                st.session_state.log_history.append(clean_line)

                # Show only the most recent 15 lines to keep the log terminal scrollable
                # without the browser DOM becoming very large over long runs.
                display_logs = "\n".join(st.session_state.log_history[-15:])
                logs_container.code(display_logs, language="bash")

                if "Completed test:" in clean_line:
                    completed_inferences += 1
                    outcome_name = clean_line.split(" -> ")[-1].strip()

                    if outcome_name not in st.session_state.live_outcomes:
                        st.session_state.live_outcomes[outcome_name] = 0
                    st.session_state.live_outcomes[outcome_name] += 1

                    if st.session_state.live_outcomes:
                        fig = charts.live_outcome_pie(st.session_state.live_outcomes)
                        live_chart.plotly_chart(fig, use_container_width=True)

                    progress_fraction = min(completed_inferences / total_inferences, 1.0)
                    progress_bar.progress(progress_fraction)

                    status_msg = clean_line.split("Completed test: ")[-1]
                    st.session_state.run_status = {
                        "type": "running",
                        "message": f"Processing {completed_inferences}/{total_inferences}: {status_msg}"
                    }
                    status_text.info(st.session_state.run_status["message"])

                elif "INITIATING DISTRIBUTED JUDGMENT" in clean_line or "LLM-AS-A-JUDGE" in clean_line:
                    st.session_state.run_status = {
                        "type": "warning",
                        "message": f"Tests finished! Waiting for LLM Judge ({ctx.judge_model}) batch evaluation..."
                    }
                    status_text.warning(st.session_state.run_status["message"])

            process.stdout.close()
            process.wait()

            if process.returncode == 0:
                st.session_state.run_status = {
                    "type": "success",
                    "message": "🎉 Benchmark completed successfully! Check the Results tab."
                }
                load_and_parse_results.clear()
            else:
                st.session_state.run_status = {
                    "type": "error",
                    "message": f"Execution failed with return code {process.returncode}"
                }
                load_and_parse_results.clear()  # Clear cache even on failure to surface partial results

        except BaseException:
            is_interrupted = True
            st.session_state.run_status = {
                "type": "error",
                "message": "Benchmark processing was interrupted or terminated."
            }
            raise

        finally:
            if 'process' in locals() and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()

            st.session_state.benchmark_process = None  # Clear stored handle
            st.session_state.benchmark_running = False
            if os.path.exists(ctx.temp_config_file):
                os.remove(ctx.temp_config_file)

            if not is_interrupted:
                st.rerun()

    # Persistent display block — shown after a run completes (running=False, logs non-empty)
    if not st.session_state.benchmark_running and st.session_state.log_history:
        st.divider()
        st.subheader("Execution Summary (Last Run)")

        if st.session_state.run_status["type"] == "success":
            st.success(st.session_state.run_status["message"])
        elif st.session_state.run_status["type"] == "error":
            st.error(st.session_state.run_status["message"])
        elif st.session_state.run_status["type"] == "warning":
            st.warning(st.session_state.run_status["message"])
        elif st.session_state.run_status["type"] == "info":
            st.info(st.session_state.run_status["message"])

        with st.expander("Final Run Terminal Logs", expanded=True):
            st.code("\n".join(st.session_state.log_history), language="bash")
