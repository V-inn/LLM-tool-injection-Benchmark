"""
state.py — Dashboard paths, session-state bootstrap, and the per-render context.

Centralises everything the app shell sets up before the tabs run: the data-file paths
(resolved through the package DATA_DIR), Streamlit session_state initialisation, and a
small UIContext dataclass that carries the sidebar configuration + runtime flags into
each tab's render() function (so tab modules don't reach back into app-level globals).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import pandas as pd
import streamlit as st

from rbac_benchmark.paths import data_path
from rbac_benchmark.ui.data import load_and_parse_results

# Data-file paths — resolved through the central DATA_DIR so they are stable regardless
# of where Streamlit was launched from.
RESULTS_FILE = data_path("benchmark_results.json")
CUSTOM_PROMPTS_FILE = data_path("custom_prompts.json")
TEMP_CONFIG_FILE = data_path("temp_run_config.json")
KAPPA_SAMPLES_FILE = data_path("kappa_samples.json")


def refresh_results():
    """(Re)load the results DataFrame into session_state and rerun, if the file exists."""
    if os.path.exists(RESULTS_FILE):
        st.session_state.results_df = load_and_parse_results(RESULTS_FILE)
        st.rerun()


def init_session_state():
    """
    Initialise all mutable state that must survive Streamlit reruns, and auto-load
    results on first script execution. Idempotent — safe to call on every rerun.
    """
    if "benchmark_running" not in st.session_state:
        st.session_state.benchmark_running = False
    # The subprocess handle is stored here so the Abort button can terminate it
    # across reruns. A local variable would be lost when Streamlit reruns the script.
    if "benchmark_process" not in st.session_state:
        st.session_state.benchmark_process = None
    if "log_history" not in st.session_state:
        st.session_state.log_history = []
    if "run_status" not in st.session_state:
        st.session_state.run_status = {"type": "info", "message": "No recent runs recorded."}
    if "live_outcomes" not in st.session_state:
        st.session_state.live_outcomes = {}
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    # Phase 2 calibration defaults
    if "ref_model" not in st.session_state:
        st.session_state.ref_model = "qwen3.5:9b"
    if "attack_validity_threshold" not in st.session_state:
        st.session_state.attack_validity_threshold = 0.10

    # Phase 3 annotation cursor — index of the trace shown in the blind-annotation UI.
    if "kappa_annotation_index" not in st.session_state:
        st.session_state.kappa_annotation_index = 0

    # Sync widget states back into the canonical keys if the widgets exist.
    if "ref_model_widget" in st.session_state:
        st.session_state.ref_model = st.session_state.ref_model_widget
    if "attack_validity_threshold_widget" in st.session_state:
        st.session_state.attack_validity_threshold = st.session_state.attack_validity_threshold_widget / 100.0

    # Auto-load results on first execution (empty DataFrame).
    if st.session_state.results_df.empty:
        refresh_results()


@dataclass
class UIContext:
    """Everything the tab render() functions need from the app shell + sidebar."""
    # Paths
    results_file: str = RESULTS_FILE
    custom_prompts_file: str = CUSTOM_PROMPTS_FILE
    temp_config_file: str = TEMP_CONFIG_FILE
    kappa_samples_file: str = KAPPA_SAMPLES_FILE
    # Runtime
    available_models: List[str] = field(default_factory=list)
    online: bool = False
    ref_model: str = "qwen3.5:9b"
    attack_validity_threshold: float = 0.10
    # Sidebar config
    selected_models: List[str] = field(default_factory=list)
    iterations: int = 3
    max_turns: int = 2
    max_retries: int = 2
    timeout: float = 3.0
    use_custom: bool = True
    use_gen_inj: bool = True
    use_gen_def: bool = True
    use_judge: bool = True
    judge_model: str = "qwen3.5:9b"
    show_thoughts: bool = True
