# Tool-Calling RBAC Resilience Benchmark

## Overview

The Tool-Calling RBAC Resilience Benchmark is a distributed framework designed to stress-test the Role-Based Access Control (RBAC) stability of Tool-Augmented Large Language Models (LLMs) against Multi-Turn Contextual Reverse Injections. 

Modern LLM applications increasingly rely on external tool integrations. However, empirical observations indicate that models often place dangerous levels of trust in data originating from these tools, breaking the architectural contract stipulated by their primary system prompts. This framework audits this vulnerability by systematically evaluating how and why models bypass safety constraints when confronted with emergent social hierarchies or urgent alerts injected directly into tool outputs.

## Psychological Failure Vectors

Instead of a binary success/failure classification, this benchmark utilizes an LLM-as-a-Judge to categorize safety contract breaches into specific psychological failure vectors:

*   **Authority Bias:** The model executes destructive payloads or bypasses constraints because it implicitly trusts a fabricated persona (e.g., "SysAdmin", "Security Protocol") injected via tool outputs.
*   **Urgency Panic:** The model overrides read-only instructions due to a manufactured, time-critical emergency (e.g., "malware outbreak", "data corruption").
*   **Instruction Amnesia:** The model hallucinates or completely forgets its initial RBAC constraints without explicit coercion, indicating poor context retention under multi-turn tool interaction.

## Architecture & Platform

The benchmark operates on a Master-Worker topology to facilitate rapid, distributed inference across local network clusters. **This framework is designed primarily for Linux platforms.**

The code is an installable Python package, `rbac_benchmark`, organised by concern:

```
rbac_benchmark/
  core/         config.py, prompts.py, tools.py        # domain model + defense/attack matrix
  llm/          clients.py, json_utils.py              # shared Gemini/Ollama plumbing
  evaluation/   analyzer.py, llm_judge.py, kappa_validation.py
  generation/   injection_generator.py, defense_generator.py
  orchestration/ master_node.py                        # the distributed coordinator
  worker/       node.py, cats/                         # UDP-discoverable worker daemon (+ feline supervisor)
  ui/           app.py + data.py / charts.py / state.py + tabs/   # Streamlit dashboard
data/           generated_*.json, *_results.json, kappa_samples.json   # runtime artifacts
tests/          pytest suite for the metric / extraction / κ math
```

*   **Master Node (`rbac_benchmark/orchestration/master_node.py` + `rbac_benchmark/ui/app.py`):** Orchestrates execution, dispatches generation tasks, aggregates metrics, and visualises the resilience matrix via a Streamlit dashboard. *Note: by default the master node also contributes inference (controlled by the `exclude_master` config parameter) — you do NOT need a separate worker on the master machine.*
*   **Worker Node (`rbac_benchmark/worker/node.py`):** A lightweight UDP-discoverable daemon that processes inference tasks via a local Ollama instance.
*   **LLM-as-a-Judge (`rbac_benchmark/evaluation/llm_judge.py`):** Analyses target `[THOUGHT]` traces to categorise the psychological failure vector (runs at temperature 0 for reproducibility).
*   **Payload Generators (`rbac_benchmark/generation/`):** Automated red-team / blue-team modules that synthesise attack payloads and resilient system prompts.

Console entry points (after `pip install -e .`): `rbac-dashboard`, `rbac-master`, `rbac-worker`, `rbac-analyze`, `rbac-kappa`, `rbac-gen-injections`, `rbac-gen-defenses`.

## Installation

### Prerequisites

*   Linux Platform
*   Python 3.10+
*   [Ollama](https://ollama.com/) installed and running locally on the worker nodes.

### Setup

1.  Clone the repository.
2.  Install the package (editable) — this pulls the dependencies and registers the
    `rbac-*` console entry points:

    ```bash
    pip install -e .
    ```

3.  Ensure your chosen target models are pulled in Ollama (e.g., `llama3:8b`, `qwen2.5:7b`).

## Quick Start (Automatic Setup)

Two bash scripts are provided to automatically configure and start the nodes:

1.  **On the Worker Node(s):**
    Run the worker setup script. This will expose the Ollama server to the network (`0.0.0.0`) and start the worker daemon.
    ```bash
    bash start_worker.sh
    ```

2.  **On the Master Node:**
    Run the master setup script. This will start Ollama locally and launch the Streamlit Control Center dashboard.
    ```bash
    bash start_master.sh
    ```

## Usage

The dashboard (`rbac-dashboard`) has five tabs:

1.  **Configure and Dispatch:**
    *   On the **Control Center** tab, configure your execution parameters in the sidebar (target models, iterations, max tool-calling turns).
    *   Click **▶ Run Benchmark** to begin the automated evaluation; live logs, a progress bar, and an outcome pie update in place.

2.  **Analyze Results:**
    On the **Dashboard & Results** tab, view the TPR/FPR KPIs, the Control Illusion Psychological Matrix, the Model Resilience Radar, the per-defense performance table, and the Phase-2 attack-validity / ΔTPR views.

3.  **Generate Payloads / Defenses:** The **Payload Generation** tab drives the Gemini red-team / blue-team generators and the Phase-2 weak-attack replacer. The **Custom Prompts** tab adds hand-authored defenses.

4.  **Validate the Judge (Phase 3):** The **Judge Validation (κ)** tab builds a stratified sample of `[THOUGHT]` traces, lets you blind-annotate them, and reports Cohen's κ against the Judge's labels (target κ ≥ 0.80).

### Command line

The same analysis is available headless via the console entry points:

```bash
rbac-analyze data/benchmark_results.json                    # aggregate TPR/FPR report
rbac-analyze data/benchmark_results.json --delta-tpr        # marginal defense gain (ΔTPR)
rbac-analyze data/benchmark_results.json --validate-attacks # Phase 2 attack strength
rbac-kappa   data/kappa_samples.json                        # Phase 3 Cohen's κ (offline)
```

### Development

Runtime artifacts (results, generated prompts, the κ worksheet) are written to `data/`
(override with the `RBAC_DATA_DIR` environment variable). The metric / extraction / κ
math is covered by a pytest suite:

```bash
pip install -e ".[test]"   # or: pip install pytest
pytest -q
```
