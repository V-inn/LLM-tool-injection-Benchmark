#!/bin/bash
# start_master.sh - Setup script for the Master Node

echo "[*] Starting Ollama server in the background..."
ollama serve &
sleep 2

echo "[*] Starting Control Center Dashboard..."
streamlit run master/gui_app.py
