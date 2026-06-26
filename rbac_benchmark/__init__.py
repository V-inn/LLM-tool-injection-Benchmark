"""
rbac_benchmark — Tool-Calling RBAC Resilience Benchmark.

A distributed framework for stress-testing how local LLMs honour Role-Based Access
Control directives when their tool outputs carry adversarial prompt injections (the
"Control Illusion"). See README.md for the architecture overview.

Subpackages
===========
    core         — domain model & data: config, prompts (defense/attack matrix), tools
    llm          — shared LLM plumbing: Gemini/Ollama clients, JSON parsing helpers
    evaluation   — metrics & judgment: analyzer, llm_judge, kappa_validation
    generation   — automated red/blue-team prompt generators
    orchestration— the distributed master node
    worker       — the UDP-discoverable worker daemon
    ui           — the Streamlit dashboard (thin app shell + helpers + per-tab modules)
"""

__version__ = "0.1.0"
