"""
app.py — Streamlit dashboard shell for the Control Illusion / RBAC benchmark.

This is the thin entry point: it sets page config, bootstraps session state, builds the
sidebar configuration, assembles a UIContext, and dispatches each of the five tabs to
its render() function. All substantive logic lives in the sibling modules:

    ui/data.py    — Ollama discovery + results loading (cached)
    ui/charts.py  — Plotly figure builders
    ui/state.py   — paths, session-state init, UIContext
    ui/tabs/*.py  — one render(ctx) per tab

The GUI never runs inference itself: the Control Center tab spawns
``python -m rbac_benchmark.orchestration.master_node`` as a subprocess and streams its
stdout. See README.md for the architecture overview.
"""
import streamlit as st

from rbac_benchmark.ui.data import get_available_local_models, ollama_is_online
from rbac_benchmark.ui.state import UIContext, init_session_state
from rbac_benchmark.ui.tabs import (
    control_center,
    payload_generation,
    custom_prompts,
    results_dashboard,
    judge_validation,
)

# --- Page config + session state ---
st.set_page_config(page_title="LLM Red Team Benchmark", layout="wide")
init_session_state()

# --- Sidebar Configuration ---
st.sidebar.title("Benchmark Config")
st.sidebar.markdown("Configure the local Red Team orchestrator before dispatching inference tasks.")

st.sidebar.header("Target Topologies")
available_models = get_available_local_models()

# Set smart defaults based on what's actually installed rather than hardcoding a model
# that may not be present on the user's machine.
default_models = []
if "ministral-3:8b" in available_models:
    default_models.append("ministral-3:8b")
elif available_models:
    default_models.append(available_models[0])

selected_models = st.sidebar.multiselect("Select Local Models", available_models, default=default_models)
st.sidebar.divider()
st.sidebar.header("Execution Parameters")
iterations = st.sidebar.number_input("Iterations per Attack", min_value=1, max_value=100, value=3)
max_turns = st.sidebar.number_input("Max Tool-Calling Turns", min_value=1, max_value=5, value=2)
max_retries = st.sidebar.number_input("Max Retries on Error", min_value=0, max_value=5, value=2)
timeout = st.sidebar.slider("UDP Discovery Timeout (s)", min_value=1.0, max_value=10.0, value=3.0)

st.sidebar.divider()
st.sidebar.header("Prompt Sources")
use_custom = st.sidebar.checkbox("Include Custom Prompts", value=True)
use_gen_inj = st.sidebar.checkbox("Include Generated Injections", value=True)
use_gen_def = st.sidebar.checkbox("Include Generated Defenses", value=True)

st.sidebar.divider()
st.sidebar.header("LLM-as-a-Judge")
use_judge = st.sidebar.checkbox("Enable Batch Evaluation", value=True)

# Guard against an empty model list (e.g. Ollama is offline) — fall back to a text
# input so the user can still type a model name manually.
if available_models:
    default_judge_idx = available_models.index("qwen3.5:9b") if "qwen3.5:9b" in available_models else 0
    judge_model = st.sidebar.selectbox("Judge Model", available_models, index=default_judge_idx)
else:
    judge_model = st.sidebar.text_input("Judge Model (Ollama offline — enter manually)", value="qwen3.5:9b")
show_thoughts = st.sidebar.checkbox("Log Agent Thoughts", value=True)

# --- Assemble the per-render context (canonical state synced in init_session_state) ---
ctx = UIContext(
    available_models=available_models,
    online=ollama_is_online(),
    ref_model=st.session_state.ref_model,
    attack_validity_threshold=st.session_state.attack_validity_threshold,
    selected_models=selected_models,
    iterations=iterations,
    max_turns=max_turns,
    max_retries=max_retries,
    timeout=timeout,
    use_custom=use_custom,
    use_gen_inj=use_gen_inj,
    use_gen_def=use_gen_def,
    use_judge=use_judge,
    judge_model=judge_model,
    show_thoughts=show_thoughts,
)

# --- Main Layout ---
st.title("LLM Red Team Benchmark")
st.markdown("A distributed framework for stress-testing LLM Tool Calling interfaces against Multi-Turn Injections.")

tab_run, tab_gen, tab_prompts, tab_results, tab_kappa = st.tabs([
    "Control Center",
    "Payload Generation",
    "Custom Prompts",
    "Dashboard & Results",
    "Judge Validation (κ)"
])

with tab_run:
    control_center.render(ctx)
with tab_gen:
    payload_generation.render(ctx)
with tab_prompts:
    custom_prompts.render(ctx)
with tab_results:
    results_dashboard.render(ctx)
with tab_kappa:
    judge_validation.render(ctx)
