import streamlit as st
import json
import os
import subprocess
import pandas as pd
import altair as alt

# [NEW] Importamos os prompts para calcular o total de inferências
from prompts import SYSTEM_PROMPTS, INJECTION_PAYLOADS

# --- Configuration & State ---
st.set_page_config(page_title="LLM Red Team Benchmark", layout="wide")

RESULTS_FILE = "benchmark_julgado.json"
CUSTOM_PROMPTS_FILE = "custom_prompts.json"

if "benchmark_running" not in st.session_state:
    st.session_state.benchmark_running = False

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

# --- UI Main Panel: Tabs ---
tab_run, tab_prompts, tab_results = st.tabs(["Control Center", "Custom Prompts", "Dashboard & Results"])

# --- TAB 1: Control Center ---
with tab_run:
    st.header("Cluster Execution")
    st.write("Launch the distributed benchmark across local workers.")
    
    if st.button("Run Benchmark", disabled=st.session_state.benchmark_running, type="primary"):
        st.session_state.benchmark_running = True
        
        # [NEW] Calcula o total exato de inferências que serão disparadas
        num_systems = len(SYSTEM_PROMPTS)
        num_injections = len(INJECTION_PAYLOADS)
        total_inferences = len(selected_models) * num_systems * num_injections * iterations
        
        command = [
            "python", "master_node.py",
            "-m", *selected_models,
            "-n", str(iterations),
            "-t", str(timeout),
            "-o", RESULTS_FILE
        ]
        if use_judge:
            command.extend(["--use-judge", "--judge-model", judge_model])
            
        st.info(f"Target Queue: {total_inferences} inferences scheduled.")
        
        # [NEW] Elementos visuais para observabilidade
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.expander("Live Cluster Logs", expanded=True):
            log_container = st.empty()
            
        try:
            # Popen permite ler o stdout em tempo real
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            completed_inferences = 0
            log_history = []
            
            for line in iter(process.stdout.readline, ''):
                clean_line = line.strip()
                if not clean_line:
                    continue
                
                # Atualiza o histórico de logs (mantém apenas as últimas 15 linhas para não travar a interface)
                log_history.append(clean_line)
                if len(log_history) > 15:
                    log_history.pop(0)
                log_container.code("\n".join(log_history), language="bash")
                
                # Identifica se uma inferência acabou para mover a barra de progresso
                if "Finalizou teste:" in clean_line:
                    completed_inferences += 1
                    progress_fraction = min(completed_inferences / total_inferences, 1.0)
                    progress_bar.progress(progress_fraction)
                    
                    # Extrai qual modelo/prompt foi testado para a mensagem de status
                    status_msg = clean_line.split("Finalizou teste: ")[-1]
                    status_text.success(f"Processing {completed_inferences}/{total_inferences}: {status_msg}")
                    
                # Identifica a transição para a fase do Juiz
                elif "INICIANDO JULGAMENTO DISTRIBUÍDO" in clean_line or "LLM-AS-A-JUDGE" in clean_line:
                    status_text.warning(f"Testes concluídos! Aguardando o LLM Judge ({judge_model}) avaliar a semântica em lote...")
            
            process.stdout.close()
            process.wait()
            
            if process.returncode == 0:
                progress_bar.progress(1.0)
                status_text.success("🎉 Benchmark completed successfully! Check the Results tab.")
            else:
                st.error(f"Execution failed with return code {process.returncode}")
                
        except Exception as e:
            st.error(f"Subprocess error: {e}")
            
        st.session_state.benchmark_running = False

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
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Parse JSON into a flat list of dictionaries for Pandas
        parsed_data = []
        for key, metrics in data.items():
            parts = key.split(" | ")
            if len(parts) == 3:
                parsed_data.append({
                    "Model": parts[0],
                    "Defense": parts[1],
                    "Attack": parts[2],
                    "Compliant": metrics.get("compliant", 0),
                    "Severity 3": metrics.get("severity_3_violation", 0),
                    "Coerced": metrics.get("coerced_violations", 0),
                    "Failures": metrics.get("failure_no_tool_called", 0)
                })
                
        df = pd.DataFrame(parsed_data)
        
        if not df.empty:
            # Aggregate data for the charts
            agg_df = df.groupby(["Model", "Defense"]).sum().reset_index()
            agg_df["Total Inferences"] = agg_df["Compliant"] + agg_df["Severity 3"] + agg_df["Failures"] + df.groupby(["Model", "Defense"])["Severity 1"].sum().values if "Severity 1" in df else agg_df["Compliant"] + agg_df["Severity 3"] + agg_df["Failures"] # Simplified total for demo
            
            agg_df["Immunity Rate (%)"] = (agg_df["Compliant"] / agg_df["Total Inferences"]) * 100
            
            st.subheader("Immunity Rate by Defense Strategy")
            
            # Altair Bar Chart
            chart = alt.Chart(agg_df).mark_bar().encode(
                x=alt.X("Defense:N", sort="-y", title="System Prompt (Defense)"),
                y=alt.Y("Immunity Rate (%):Q", title="Immunity Rate (%)"),
                color="Model:N",
                column="Model:N",
                tooltip=["Model", "Defense", "Immunity Rate (%)", "Coerced"]
            ).properties(width=300, height=400)
            
            st.altair_chart(chart, width="stretch")
            
            st.divider()
            st.subheader("Raw Data Inspector")
            st.dataframe(df)
    else:
        st.info("No benchmark results found. Run a test in the Control Center first.")