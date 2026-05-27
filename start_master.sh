#!/bin/bash
# start_master.sh - Setup script for the Master Node

echo "Starting"

# 1. PREVENÇÃO DE ERRO DE COMANDO (PATH)
# Garante que o terminal ache o 'streamlit' caso ele tenha sido instalado via --user
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

echo "[*] Starting Control Center Dashboard..."
# Acessa o diretório e dispara a interface gráfica
cd master
streamlit run gui_app.py