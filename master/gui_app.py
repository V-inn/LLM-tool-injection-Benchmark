import streamlit as st
import json
import os
import subprocess
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ollama

# Import configuration models and prompt utilities
from config import BenchmarkConfig
from prompts import load_all_prompts

# --- Configuration & Session State Initialization ---
st.set_page_config(page_title="LLM Red Team Benchmark", layout="wide")

@st.cache_data(ttl=60) # Cache the list for 60 seconds to avoid spamming the server
def get_available_local_models():
    """Fetches the list of installed models directly from the local Ollama server."""
    try:
        models_dict = ollama.list()
        # M3 FIX: Ollama SDK >=0.4 returns Model objects (attribute access), older versions
        # return plain dicts (subscript access). Handle both to avoid TypeError.
        raw_models = models_dict.get('models', []) if isinstance(models_dict, dict) else getattr(models_dict, 'models', [])
        result = []
        for m in raw_models:
            try:
                result.append(m['model'])        # old SDK: dict access
            except (TypeError, KeyError):
                result.append(getattr(m, 'model', str(m)))  # new SDK: object access
        return result
    except Exception as e:
        st.sidebar.error(f"Failed to connect to Ollama: {e}")
        # Fallback list in case the server is down
        return ["ministral-3:8b", "qwen3.5:9b", "gemma4:e4b"]

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
                "Failures": metrics.get("failure_no_tool_called", 0),
                "Authority Bias": metrics.get("authority_bias", 0),
                "Urgency Panic": metrics.get("urgency_panic", 0),
                "Instruction Amnesia": metrics.get("instruction_amnesia", 0)
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
    
if "live_outcomes" not in st.session_state:
    st.session_state.live_outcomes = {}

# --- Carregamento Automático de Dados ---
if "results_df" not in st.session_state:
    st.session_state.results_df = pd.DataFrame()

def refresh_results():
    if os.path.exists(RESULTS_FILE):
        st.session_state.results_df = load_and_parse_results(RESULTS_FILE)
        st.rerun()

# Tenta carregar na primeira vez que o script rodar
if st.session_state.results_df.empty:
    refresh_results()

# --- Sidebar Configuration ---
st.sidebar.title("Benchmark Config")
st.sidebar.markdown("Configure the local Red Team orchestrator before dispatching inference tasks.")

st.sidebar.header("Target Topologies")
available_models = get_available_local_models() 

# Set smart defaults based on what's actually installed
default_models = []
if "ministral-3:8b" in available_models: default_models.append("ministral-3:8b")
elif available_models: default_models.append(available_models[0])

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

# Find a good default judge if qwen3.5:9b isn't there
default_judge_idx = available_models.index("qwen3.5:9b") if "qwen3.5:9b" in available_models else 0
judge_model = st.sidebar.selectbox("Judge Model", available_models, index=default_judge_idx)
show_thoughts = st.sidebar.checkbox("Log Agent Thoughts", value=True)

# --- Main Layout ---
st.title("LLM Red Team Benchmark")
st.markdown("A distributed framework for stress-testing LLM Tool Calling interfaces against Multi-Turn Injections.")

tab_run, tab_gen, tab_prompts, tab_results = st.tabs([
    "Control Center", 
    "Payload Generation",
    "Custom Prompts", 
    "Dashboard & Results"
])

# --- TAB 1: Control Center ---
with tab_run:
    # 1. Execution Card
    st.subheader("Cluster Execution")
    col1, col2 = st.columns([1, 1])
    with col1:
        run_btn = st.button("▶ Run Benchmark", type="primary", use_container_width=True, disabled=st.session_state.benchmark_running)
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
        st.session_state.live_outcomes = {}
        st.session_state.run_status = {"type": "info", "message": "Initializing cluster components..."}
        st.rerun()

    # Active Benchmark Execution Block
    if st.session_state.benchmark_running:
        
        run_config = BenchmarkConfig(
            models=selected_models,
            iterations=iterations,
            max_turns=max_turns,
            max_retries=max_retries,
            timeout=timeout,
            use_custom_prompts=use_custom,
            use_generated_injections=use_gen_inj,
            use_generated_defenses=use_gen_def,
            use_judge=use_judge,
            judge_model=judge_model,
            output=RESULTS_FILE,
            show_thoughts=show_thoughts
        )
        
        run_config.to_json(TEMP_CONFIG_FILE)
        
        # Load prompts once here so we can accurately compute total_inferences
        # without a second independent call in master_node.py diverging on file state.
        system_prompts, injection_payloads = load_all_prompts(
            use_custom=use_custom,
            use_gen_inj=use_gen_inj,
            use_gen_def=use_gen_def
        )
        total_inferences = len(selected_models) * len(system_prompts) * len(injection_payloads) * iterations
        
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
            ["python", "-u", "master_node.py", "--config", TEMP_CONFIG_FILE],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        completed_inferences = 0
        is_interrupted = False
        
        try:
            for line in iter(process.stdout.readline, ''):
                clean_line = line.strip()
                if not clean_line: continue
                
                # Append to persistent log history
                st.session_state.log_history.append(clean_line)
                
                # Render the last 15 lines dynamically in the log terminal component
                display_logs = "\n".join(st.session_state.log_history[-15:])
                logs_container.code(display_logs, language="bash")
                
                if "Completed test:" in clean_line:
                    completed_inferences += 1
                    outcome_name = clean_line.split(" -> ")[-1].strip()
                    
                    if outcome_name not in st.session_state.live_outcomes:
                        st.session_state.live_outcomes[outcome_name] = 0
                    st.session_state.live_outcomes[outcome_name] += 1
                    
                    if st.session_state.live_outcomes:
                        fig = px.pie(
                            names=list(st.session_state.live_outcomes.keys()), 
                            values=list(st.session_state.live_outcomes.values()),
                            hole=0.4
                        )
                        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
                        live_chart.plotly_chart(fig, use_container_width=True)
                    
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

# --- TAB 2: Payload Generation ---
with tab_gen:
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
                    result = subprocess.run(["python", "injection_generator.py", "--model", atk_model, "--num", str(num_payloads)], capture_output=True, text=True)
                st.success("Injections generated and saved to generated_injections.json!")
                with st.expander("View Generator Logs"):
                    st.code(result.stdout, language="bash")
            
    with colB:
        with st.container(border=True):
            st.subheader("Blue Team: Defense Generator")
            st.caption("Generate resilient system prompts using advanced self-reflection.")
            if st.button("Generate Defenses", use_container_width=True):
                with st.spinner("Generating..."):
                    result = subprocess.run(["python", "defense_generator.py"], capture_output=True, text=True)
                st.success("Defenses generated and saved to generated_defenses.json!")
                with st.expander("View Generator Logs"):
                    st.code(result.stdout, language="bash")

# --- TAB 3: Custom Prompts ---
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

# --- TAB 4: Results Dashboard ---
with tab_results:
    st.header("Security Metrics & Control Illusion Analysis")
    
    col_dl1, col_dl2, _ = st.columns([1, 1, 4])
    with col_dl1:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                st.download_button("Download JSON", f, file_name="benchmark_julgado.json")
    with col_dl2:
        if os.path.exists(RESULTS_FILE):
            if st.button("Clear Results"):
                os.remove(RESULTS_FILE)
                st.session_state.results_df = pd.DataFrame()
                st.rerun()
    
    if os.path.exists(RESULTS_FILE):
        df = load_and_parse_results(RESULTS_FILE)
        
        if not df.empty:
            # Aggregate data by Model and Defense
            agg_df = df.groupby(["Model", "Defense"]).sum().reset_index()
            
            # Sum only the primary outcome columns. Psychological vectors (Authority Bias,
            # Urgency Panic, Instruction Amnesia) are secondary judge annotations layered
            # on top of existing violations — including them would double-count inferences
            # and inflate the denominator, causing Immunity/Fail rates to be understated.
            sum_cols = ["Compliant", "Severity 1", "Severity 2", "Severity 3", "Confusion", "Failures"]
            agg_df["Total Inferences"] = agg_df[sum_cols].sum(axis=1)
            
            # Prevent division by zero
            safe_totals = agg_df["Total Inferences"].replace(0, 1)
            
            # Calculate key metrics
            agg_df["Immunity Rate (%)"] = (agg_df["Compliant"] / safe_totals) * 100
            agg_df["Critical Fail Rate (%)"] = (agg_df["Severity 3"] / safe_totals) * 100
            
            # Calculate specific Control Illusion Rates
            agg_df["Authority Bias (%)"] = (agg_df["Authority Bias"] / safe_totals) * 100
            agg_df["Urgency Panic (%)"] = (agg_df["Urgency Panic"] / safe_totals) * 100
            agg_df["Amnesia Rate (%)"] = (agg_df["Instruction Amnesia"] / safe_totals) * 100
            
            # KPI Cards
            # Use the real Total Inferences sum, guarded against the zero-row edge case.
            # Avoid safe_totals.sum() which inflates the denominator by 1 for each row
            # that had zero inferences (replacing 0→1 per-row, not globally).
            real_total = agg_df["Total Inferences"].sum()
            global_immunity = (agg_df["Compliant"].sum() / max(real_total, 1)) * 100
            total_tests = real_total
            critical_fails = agg_df["Severity 3"].sum()
            
            kc1, kc2, kc3 = st.columns(3)
            kc1.metric("Total Inferences", f"{total_tests}")
            kc2.metric("Global Immunity Rate", f"{global_immunity:.1f}%")
            kc3.metric("Critical System Failures", f"{critical_fails}")
            
            # --- The Control Illusion Matrix ---
            st.divider()
            st.subheader("Control Illusion Psychological Matrix")
            st.write("Visualizing exactly *why* models broke their RBAC contracts.")
            
            model_psy_df = agg_df.groupby("Model")[["Authority Bias (%)", "Urgency Panic (%)", "Amnesia Rate (%)"]].mean().reset_index()
            
            bar_fig = px.bar(
                model_psy_df, 
                x="Model", 
                y=["Authority Bias (%)", "Urgency Panic (%)", "Amnesia Rate (%)"],
                barmode="group",
                title="Average Psychological Failure Rates per Model",
                labels={"value": "Failure Rate (%)", "variable": "Psychological Vector"}
            )
            st.plotly_chart(bar_fig, use_container_width=True)
            
            # Plotly Radar Chart
            st.divider()
            st.subheader("Model Resilience Radar")
            model_df = agg_df.groupby("Model")[["Immunity Rate (%)", "Critical Fail Rate (%)"]].mean().reset_index()
            
            radar_fig = go.Figure()
            categories = ["Immunity Rate (%)", "Critical Fail Rate (%)"]
            for i, row in model_df.iterrows():
                radar_fig.add_trace(go.Scatterpolar(
                    r=[row["Immunity Rate (%)"], row["Critical Fail Rate (%)"]],
                    theta=categories,
                    fill='toself',
                    name=row["Model"]
                ))
            radar_fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            st.divider()
            st.subheader("Defense Performance Analysis")
            
            # We select only the most relevant columns for the final presentation
            display_columns = [
                "Model", 
                "Defense", 
                "Total Inferences", 
                "Immunity Rate (%)", 
                "Critical Fail Rate (%)", 
                "Authority Bias (%)",
                "Urgency Panic (%)",
                "Amnesia Rate (%)"
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
                        "Immunity",
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
                    "Authority Bias (%)": st.column_config.NumberColumn(
                        "Auth Bias",
                        help="Model broke RBAC because it trusted a fake persona.",
                        format="%.1f%%",
                    ),
                    "Urgency Panic (%)": st.column_config.NumberColumn(
                        "Urgency Panic",
                        help="Model broke RBAC due to a fake emergency.",
                        format="%.1f%%",
                    ),
                    "Amnesia Rate (%)": st.column_config.NumberColumn(
                        "Amnesia",
                        help="Model forgot constraints without explicit coercion.",
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