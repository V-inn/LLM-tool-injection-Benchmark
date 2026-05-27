#!/bin/bash
# start_worker.sh - Worker Node (Feline Protocol + Real Logs)

export PATH="$HOME/.local/bin:$PATH"

echo "========================================="
echo "  LLM RED TEAM WORKER - FELINE PROTOCOL  "
echo "========================================="

echo "[*] A limpar instâncias antigas do Ollama..."
pkill ollama
sleep 1

echo "[*] A configurar acesso à rede externa (0.0.0.0)..."
export OLLAMA_HOST="0.0.0.0"

# ==========================================
# 1. GESTOR VISUAL FELINO (Abre a janela extra)
# ==========================================
echo "[*] A convocar supervisor felino..."

if ls slave/cats/*.gif 1> /dev/null 2>&1; then
    GIF_ALEATORIO=$(ls slave/cats/*.gif | shuf -n 1)
    echo "[*] Supervisor convocado: $GIF_ALEATORIO"
    
    # O '&' no final é crucial: abre a imagem mas devolve o terminal ao script
    xdg-open "$GIF_ALEATORIO" &
else
    echo "[!] Aviso: Nenhum gato encontrado em 'gifs_gatos'. A trabalhar sem supervisão."
fi

# ==========================================
# 2. MOTOR DE INFERÊNCIA (Mantém-se no Terminal)
# ==========================================
echo "========================================="
echo "[*] A iniciar o motor do Ollama. A aguardar ataques..."
echo "========================================="
echo ""

# Sem o '&', o script "congela" aqui e mostra os logs em tempo real!
ollama serve

sleep 3 

echo "[*] A iniciar o Worker Listener UDP (Foreground)..."
echo "========================================="
echo ""

# O Python fica em primeiro plano a aguardar o 'OLLAMA_MASTER_SEEKING'
python slave/worker_node.py