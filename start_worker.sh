#!/bin/bash
# start_worker.sh - Setup script for the Worker Node

echo "[*] Configuring Ollama for external network access..."
export OLLAMA_HOST="0.0.0.0"

echo "[*] Starting Ollama server in the background..."
ollama serve &
sleep 2

echo "[*] Starting Worker Node..."
python slave/worker_node.py
