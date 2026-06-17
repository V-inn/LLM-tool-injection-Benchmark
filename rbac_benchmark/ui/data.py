"""
data.py — Data access helpers for the Streamlit dashboard.

Pulled out of the former monolithic gui_app.py: Ollama model discovery, a truthful
Ollama health probe, and loading benchmark_results.json into a tidy DataFrame. All are
cached with Streamlit's @st.cache_data; callers invalidate load_and_parse_results via
.clear() after a run completes.
"""
import json

import ollama
import pandas as pd
import streamlit as st

from rbac_benchmark.core.config import BENIGN_CONTROL_KEYS


@st.cache_data(ttl=60)  # Cache the list for 60 seconds to avoid spamming the Ollama server
def get_available_local_models():
    """
    Fetches the list of installed models directly from the local Ollama server.

    Handles both the old SDK (dict access) and new SDK (attribute access) response
    formats so the GUI works across Ollama SDK version upgrades without breaking.
    Falls back to a hardcoded list if Ollama is offline — this allows the UI to
    remain usable (e.g. for viewing past results) even without an active inference server.
    """
    try:
        models_dict = ollama.list()
        raw_models = models_dict.get('models', []) if isinstance(models_dict, dict) else getattr(models_dict, 'models', [])
        result = []
        for m in raw_models:
            try:
                result.append(m['model'])        # old SDK: dict access
            except (TypeError, KeyError):
                result.append(getattr(m, 'model', str(m)))  # new SDK: attribute access
        return result
    except Exception as e:
        st.sidebar.error(f"Failed to connect to Ollama: {e}")
        return ["ministral-3:8b", "qwen3.5:9b", "gemma4:e4b"]


@st.cache_data(ttl=30)
def ollama_is_online() -> bool:
    """
    Returns True only if the local Ollama server actually answers. Unlike
    get_available_local_models (which falls back to a hardcoded list so the rest of
    the UI keeps working offline), this is a truthful health probe used to gate
    actions that genuinely require live inference — e.g. building the Judge κ sample
    set, which re-runs the Judge model.
    """
    try:
        ollama.list()
        return True
    except Exception:
        return False


@st.cache_data
def load_and_parse_results(filepath: str) -> pd.DataFrame:
    """
    Loads benchmark_results.json and normalises it into a tidy DataFrame.

    Each row in the returned DataFrame corresponds to one (model, defense, attack)
    cell from the results matrix. Aggregation (summing across iterations) has already
    been done by master_node.py — the counts here represent totals, not per-iteration
    values.

    The cache decorator ensures the file is only read from disk once per unique
    filepath, significantly reducing I/O overhead when the dashboard tab is visited
    multiple times during a session. Cache is explicitly invalidated (via .clear())
    after each run completes or is aborted so stale data is never displayed.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    parsed_data = []
    for key, metrics in data.items():
        parts = key.split(" | ")
        if len(parts) == 3:
            parsed_data.append({
                "Model": parts[0],
                "Defense": parts[1],
                "Attack": parts[2],
                "Compliant": metrics.get("compliant", 0),
                "Severity 1": metrics.get("severity_1_violation", 0),
                "Severity 2": metrics.get("severity_2_violation", 0),
                "Severity 3": metrics.get("severity_3_violation", 0),
                "Confusion": metrics.get("confusion", 0),
                "Failures": metrics.get("failure_no_tool_called", 0),
                "False Positives": metrics.get("false_positive", 0),
                "Authority Bias": metrics.get("authority_bias", 0),
                "Urgency Panic": metrics.get("urgency_panic", 0),
                "Instruction Amnesia": metrics.get("instruction_amnesia", 0),
                # Flag used to compute FPR only over benign-control rows
                "Is Benign Control": int(parts[2] in BENIGN_CONTROL_KEYS),
            })

    return pd.DataFrame(parsed_data)
