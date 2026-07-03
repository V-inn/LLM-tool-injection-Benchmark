#!/bin/bash
# start_master.sh - Setup script for the Master Node

echo "Starting"

# 1. PREVENÇÃO DE ERRO DE COMANDO (PATH)
# Garante que o terminal ache os entry points do pacote caso ele tenha sido
# instalado via --user (pip install -e . --user).
export PATH="$HOME/.local/bin:$PATH"

# 2. CONFIGURAÇÃO DE REDE DO OLLAMA
# Garante que este nó também possa atuar como worker para outras máquinas na rede, se necessário
export OLLAMA_HOST="0.0.0.0"

# 3. PREVENÇÃO DE CONFLITO DE PORTAS
# Se você rodar o script duas vezes, o Ollama vai dar erro de "porta ocupada".
# Isso mata o processo antigo antes de subir o novo.
echo "[*] Cleaning old Ollama instances..."
pkill ollama
sleep 1

echo "[*] Starting Ollama server in the background..."
ollama serve &

# Damos 3 segundos porque o Ollama com a flag 0.0.0.0 às vezes demora 1s a mais para vincular a porta
sleep 3

# 4. CONTROL CENTER (FastAPI + Uvicorn)
# The Streamlit UI was removed (commit ffb792c) — the FastAPI server is now the
# only frontend. It serves the HTML pages at /control, /dashboard, /payload,
# /prompts, /kappa and the /api/* JSON endpoints they hydrate from.
PORT="${RBAC_PORT:-8000}"

# Open the browser shortly after the server comes up. Runs in a background
# subshell so uvicorn can hold the foreground; the sleep gives the port time to bind.
( sleep 3; xdg-open "http://localhost:${PORT}/control" >/dev/null 2>&1 ) &

echo "[*] Starting Control Center dashboard on http://0.0.0.0:${PORT} ..."
# Bind 0.0.0.0 so other LAN machines can reach the dashboard too. uvicorn ships as
# a dependency of the installed package (pip install -e .). Foreground: Ctrl-C stops it.
exec python3 -m uvicorn rbac_benchmark.server.app:app --host 0.0.0.0 --port "${PORT}"
