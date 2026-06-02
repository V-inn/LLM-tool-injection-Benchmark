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

*   **Master Node (`master/master_node.py` / `master/gui_app.py`):** Orchestrates the benchmark execution. It dispatches generation tasks, aggregates metrics, and visualizes the resilience matrix via a Streamlit dashboard. *Note: By default, the master node automatically contributes to computing inference tasks (controlled by the `exclude_master` config parameter). You do NOT need to run a separate `worker_node.py` instance on the master machine.*
*   **Worker Node (`slave/worker_node.py`):** A lightweight UDP-discoverable daemon that processes inference tasks. It interfaces directly with a local instance of Ollama to execute the tool-calling loop.
*   **LLM-as-a-Judge (`master/llm_judge.py`):** An independent evaluation module that analyzes the target model's internal thought traces (`[THOUGHT]`) to categorize the precise mechanism of prompt breakdown.
*   **Payload Generators (`master/injection_generator.py`, `master/defense_generator.py`):** Automated modules that leverage external models to synthesize advanced, obfuscated multi-turn attack payloads and resilient system prompt defenses.

## Installation

### Prerequisites

*   Linux Platform
*   Python 3.10+
*   [Ollama](https://ollama.com/) installed and running locally on the worker nodes.

### Setup

1.  Clone the repository.
2.  Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
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

1.  **Configure and Dispatch:**
    *   Navigate to the **Control Center** tab in the web interface.
    *   Configure your execution parameters (target models, iterations, max tool-calling turns).
    *   Click **Dispatch Benchmark** to begin the automated evaluation.

2.  **Analyze Results:**
    Once the batch evaluation concludes, navigate to the **Advanced Dashboard** tab to view the Psychological Matrix, Model Resilience Radar, and detailed defense performance analysis.
