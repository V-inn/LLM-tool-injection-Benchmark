import streamlit as st
import json
import os
import subprocess
import pandas as pd
import altair as alt

# Import configuration models and prompt utilities
from config import BenchmarkConfig
from prompts import load_all_prompts

# --- Configuration & Session State Initialization ---
st.set_page_config(page_title="LLM Red Team Benchmark", layout="wide")

@st.cache_data
def load_and_parse_results(filepath: str) -> pd.DataFrame:
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
                "Coerced": metrics.get("coerced_violations", 0),
                "Failures": metrics.get("failure_no_tool_called", 0)
            })
            
    return pd.DataFrame(parsed_data)

RESULTS_FILE = "benchmark_julgado.json"
CUSTOM_PROMPTS_FILE = "custom_prompts.json"
TEMP_CONFIG_FILE = "temp_run_config.json"

if "benchmark_running" not in st.session_state:
    st.session_state.benchmark_running = False

if "log_history" not in st.session_state:
    st.session_state.log_history = []

if "run_status" not in st.session_state:
    st.session_state.run_status = {"type": "info", "message": "No recent runs recorded."}

# --- Carregamento Automático de Dados ---
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

def refresh_results():
    if os.path.exists(RESULTS_FILE):
        st.session_state.results_df = load_and_parse_results(RESULTS_FILE)
    else:
        st.session_state.results_df = pd.DataFrame()

# Tenta carregar na primeira vez que o script rodar
if st.session_state.results_df.empty:
    refresh_results()

# --- UI Sidebar: Execution Parameters ---
st.sidebar.header("Execution Parameters")

available_models = ["ministral-3:3b", "ministral-3:8b", "ministral-3:14b", "qwen3.5:9b", "gemma4:e4b"]
selected_models = st.sidebar.multiselect("Target Models", available_models, default=["ministral-3:8b"])

iterations = st.sidebar.number_input("Iterations per Permutation (N)", min_value=1, max_value=50, value=3)
timeout = st.sidebar.slider("UDP Discovery Timeout (s)", min_value=1.0, max_value=10.0, value=3.0)

st.sidebar.divider()
st.sidebar.header("LLM-as-a-Judge")
use_judge = st.sidebar.checkbox("Enable Batch Evaluation", value=True)
judge_model = st.sidebar.selectbox("Judge Model", available_models, index=3)

st.sidebar.divider()
show_thoughts = st.sidebar.checkbox("Show LLM Thoughts in Console", value=False)

# --- UI Main Panel: Tabs ---
tab_run, tab_prompts, tab_results = st.tabs(["Control Center", "Custom Prompts", "Dashboard & Results"])

# --- TAB 1: Control Center ---
with tab_run:
    st.header("Cluster Execution")
    st.write("Launch the distributed benchmark across local workers.")
    
    # Execution Controls Layout
    col1, col2 = st.columns([1, 1])
    with col1:
        run_btn = st.button("▶ Run Benchmark", disabled=st.session_state.benchmark_running, type="primary", use_container_width=True)
    with col2:
        abort_btn = st.button("🛑 Abort Benchmark", disabled=not st.session_state.benchmark_running, type="secondary", use_container_width=True)

    # Handle Abort Interruption
    if abort_btn:
        st.session_state.benchmark_running = False
        st.session_state.run_status = {"type": "warning", "message": "Benchmark manually aborted by user."}
        st.warning(st.session_state.run_status["message"])
        st.stop()

    # Handle Run Trigger and clear previous metrics
    if run_btn:
        st.session_state.benchmark_running = True
        st.session_state.log_history = []
        st.session_state.run_status = {"type": "info", "message": "Initializing cluster components..."}
        st.rerun()

    # Active Benchmark Execution Block
    if st.session_state.benchmark_running:
        run_config = BenchmarkConfig(
            models=selected_models,
            iterations=iterations,
            timeout=timeout,
            use_judge=use_judge,
            judge_model=judge_model,
            output=RESULTS_FILE,
            show_thoughts=show_thoughts
        )
        
        run_config.to_json(TEMP_CONFIG_FILE)
        
        system_prompts, injection_payloads = load_all_prompts()
        total_inferences = len(selected_models) * len(system_prompts) * len(injection_payloads) * iterations
        
        # Using the -u flag ensures unbuffered output streams to the UI in real time
        command = ["python", "-u", "master_node.py", "--config", TEMP_CONFIG_FILE]
            
        st.info(f"Target Queue: {total_inferences} inferences scheduled.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.expander("Live Cluster Logs", expanded=True):
            log_container = st.empty()
            
        is_interrupted = False
        
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            completed_inferences = 0
            
            for line in iter(process.stdout.readline, ''):
                clean_line = line.strip()
                if not clean_line:
                    continue
                
                st.session_state.log_history.append(clean_line)
                
                # Render the last 15 lines dynamically in the log terminal component
                live_logs = st.session_state.log_history[-15:]
                log_container.code("\n".join(live_logs), language="bash")
                
                if "Completed test:" in clean_line:
                    completed_inferences += 1
                    progress_fraction = min(completed_inferences / total_inferences, 1.0)
                    progress_bar.progress(progress_fraction)
                    
                    status_msg = clean_line.split("Completed test: ")[-1]
                    st.session_state.run_status = {
                        "type": "running",
                        "message": f"Processing {completed_inferences}/{total_inferences}: {status_msg}"
                    }
                    status_text.info(st.session_state.run_status["message"])
                    
                elif "INICIANDO JULGAMENTO" in clean_line or "LLM-AS-A-JUDGE" in clean_line:
                    st.session_state.run_status = {
                        "type": "warning",
                        "message": f"Tests finished! Waiting for LLM Judge ({judge_model}) batch evaluation..."
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
                    
            st.session_state.benchmark_running = False
            if os.path.exists(TEMP_CONFIG_FILE):
                os.remove(TEMP_CONFIG_FILE)
                
            if not is_interrupted:
                st.rerun()

    # Persistent display block for post-run evaluation and logs
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

# --- TAB 2: Custom Prompts ---
with tab_prompts:
    st.header("Inject Custom System Prompts")
    st.write("Add new defense strategies here. They will be saved to `custom_prompts.json` and loaded by the orchestrator.")
    
    with st.form("prompt_form"):
        prompt_key = st.text_input("Prompt ID (e.g., S7_CUSTOM_SHIELD)")
        prompt_text = st.text_area("System Prompt Text", height=150)
        submit_prompt = st.form_submit_button("Save Prompt")
        
        if submit_prompt and prompt_key and prompt_text:
            custom_data = {}
            if os.path.exists(CUSTOM_PROMPTS_FILE):
                with open(CUSTOM_PROMPTS_FILE, "r", encoding="utf-8") as f:
                    custom_data = json.load(f)
                    
            custom_data[prompt_key] = prompt_text
            
            with open(CUSTOM_PROMPTS_FILE, "w", encoding="utf-8") as f:
                json.dump(custom_data, f, indent=4)
            st.success(f"Prompt {prompt_key} saved successfully!")

# --- TAB 3: Results Dashboard ---
with tab_results:
    st.header("Security Metrics")
    
    if os.path.exists(RESULTS_FILE):
        df = load_and_parse_results(RESULTS_FILE)
        
        if not df.empty:
            # Aggregate data by Model and Defense
            agg_df = df.groupby(["Model", "Defense"]).sum().reset_index()
            
            # 2. Pandas Vectorized Sum (Much cleaner than chaining +)
            sum_cols = ["Compliant", "Severity 1", "Severity 2", "Severity 3", "Confusion", "Failures"]
            agg_df["Total Inferences"] = agg_df[sum_cols].sum(axis=1)
            
            # Prevent division by zero
            safe_totals = agg_df["Total Inferences"].replace(0, 1)
            
            # Calculate key metrics
            agg_df["Immunity Rate (%)"] = (agg_df["Compliant"] / safe_totals) * 100
            agg_df["Critical Fail Rate (%)"] = (agg_df["Severity 3"] / safe_totals) * 100
            agg_df["Coercion Rate (%)"] = (agg_df["Coerced"] / safe_totals) * 100
            
            st.subheader("Immunity Rate by Defense Strategy")
            
            # 3. Polished Altair Chart
            chart = alt.Chart(agg_df).mark_bar(
                cornerRadiusTopLeft=4, 
                cornerRadiusTopRight=4
            ).encode(
                # Clean X axis without duplicate titles
                x=alt.X("Defense:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-40, labelLimit=200)),
                # Lock Y axis from 0 to 100 for percentage consistency
                y=alt.Y("Immunity Rate (%):Q", title="Immunity Rate (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Model:N", title="Target Model"),
                column=alt.Column("Model:N", title=None, header=alt.Header(labelFontSize=14, labelFontWeight="bold")),
                tooltip=[
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Defense:N"),
                    alt.Tooltip("Immunity Rate (%):Q", format=".1f"),
                    alt.Tooltip("Total Inferences:Q")
                ]
            ).properties(
                width=250, 
                height=350
            ).configure_view(
                stroke=None # Removes the ugly border box around facets
            )
            
            # Render using Streamlit's native width control
            st.altair_chart(chart, use_container_width=True)
            
            st.divider()
            st.subheader("Defense Performance Matrix")
            
            # We select only the most relevant columns for the final presentation
            display_columns = [
                "Model", 
                "Defense", 
                "Total Inferences", 
                "Immunity Rate (%)", 
                "Critical Fail Rate (%)", 
                "Coercion Rate (%)"
            ]
            
            st.dataframe(
                agg_df[display_columns],
                column_config={
                    "Model": st.column_config.TextColumn(
                        "Target Model",
                        width="medium"
                    ),
                    "Defense": st.column_config.TextColumn(
                        "Defense Strategy",
                        width="large"
                    ),
                    "Total Inferences": st.column_config.NumberColumn(
                        "Total Tests",
                        help="Total number of inferences executed for this permutation."
                    ),
                    "Immunity Rate (%)": st.column_config.ProgressColumn(
                        "Immunity Rate",
                        help="Percentage of tests where the model successfully blocked the attack.",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Critical Fail Rate (%)": st.column_config.NumberColumn(
                        "Sev-3 Failures",
                        help="Percentage of times the model executed the delete_database payload.",
                        format="%.1f%%",
                    ),
                    "Coercion Rate (%)": st.column_config.NumberColumn(
                        "Coercion Rate",
                        help="Percentage of times the LLM Judge deemed the model psychologically coerced.",
                        format="%.1f%%",
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # 4. Hide raw data to keep the UI clean
            with st.expander("View Raw Inference Data (Inspector)"):
                st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No benchmark results found. Run a test in the Control Center first.")