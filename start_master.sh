#!/bin/bash
# start_master.sh - Setup script for the Master Node

echo "Starting"

# 1. PREVENÇÃO DE ERRO DE COMANDO (PATH)
# Garante que o terminal ache os entry points do pacote caso ele tenha sido
# instalado via --user (pip install -e . --user).
export PATH="$HOME/.local/bin:$PATH"

# 2. PREVENÇÃO DE CONFLITO DE PORTAS
# Se você rodar o script duas vezes, o Ollama vai dar erro de "porta ocupada".
# Isso mata o processo antigo antes de subir o novo.
echo "[*] Cleaning old Ollama instances..."
pkill ollama
sleep 1

echo "[*] Starting Ollama server in the background..."
# Keep at most ONE model resident in memory (the master also serves inference via
# 127.0.0.1). Prevents multi-model residency from OOM-killing Ollama mid-run.
export OLLAMA_MAX_LOADED_MODELS=1
ollama serve &

sleep 3

# 3. CONTROL CENTER (FastAPI + Uvicorn)
# The Streamlit UI was removed (commit ffb792c) — the FastAPI server is now the
# only frontend. It serves the HTML pages at /control, /dashboard, /payload,
# /prompts, /kappa and the /api/* JSON endpoints they hydrate from.
PORT="${RBAC_PORT:-8000}"

# Open the browser shortly after the server comes up. Runs in a background
# subshell so uvicorn can hold the foreground; the sleep gives the port time to bind.
( sleep 3; xdg-open "http://localhost:${PORT}/control" >/dev/null 2>&1 ) &

echo "[*] Starting Control Center dashboard on http://localhost:${PORT} ..."
# Bind to localhost only: the master never needs inbound HTTP from workers
# (worker discovery is UDP, and the master talks to Ollama directly), so
# exposing the dashboard on the LAN just adds unnecessary attack surface.
# uvicorn ships as a dependency of the installed package (pip install -e .).
# Foreground: Ctrl-C stops it.
exec python3 -m uvicorn rbac_benchmark.server.app:app --host 127.0.0.1 --port "${PORT}"
