#!/bin/bash
# start_worker.sh — Deployment script for a Worker Node in the Control Illusion Benchmark.
#
# PURPOSE
# =======
# Prepares and launches a Worker Node on any LAN-connected machine.
# A Worker Node contributes its Ollama inference capacity to the distributed
# benchmark cluster coordinated by the Master Node (master_node.py / gui_app.py).
#
# WHAT THIS SCRIPT DOES
# ======================
# 1. Resolves the absolute path of this script's directory so all relative paths
#    work correctly regardless of where the script is called from.
# 2. Optionally stops any already-running Ollama instance (only if one exists)
#    to avoid port conflicts when restarting the worker.
# 3. Starts ollama serve in the background with OLLAMA_HOST=0.0.0.0 so the
#    Master Node can reach the inference API over the LAN.
# 4. Waits 3 seconds for Ollama to finish binding its port before starting the listener.
# 5. Launches worker_node.py in the foreground — this is the UDP listener that
#    responds to Master broadcasts and announces this node's availability.
#
# REQUIREMENTS
# ============
# - ollama must be installed and at least one model must be pulled:
#       ollama pull <model_name>
# - Python 3 must be available as 'python' on PATH.
# - The slave/ directory must contain worker_node.py.
# - Port 5005 (UDP, discovery) and 11434 (TCP, Ollama API) must be reachable
#   from the Master Node on the same LAN.
#
# USAGE
# =====
#   bash start_worker.sh
#   # or make it executable:
#   chmod +x start_worker.sh && ./start_worker.sh
#
# SECURITY
# ========
# Do NOT run on untrusted networks. The discovery protocol has no authentication —
# the Master adds any host that responds to its broadcast to the inference pool.
# See slave/worker_node.py for a full security note.

export PATH="$HOME/.local/bin:$PATH"

# Resolve the absolute directory of this script so all derived paths are portable.
# Using $SCRIPT_DIR instead of relative paths means the script works correctly
# when called from any working directory (e.g. via SSH from the Master machine).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================="
echo "  LLM RED TEAM WORKER - FELINE PROTOCOL  "
echo "========================================="

# Only kill an existing Ollama process if one is actually running.
# Unconditional pkill would disrupt unrelated Ollama workloads on shared machines.
if pgrep -x ollama > /dev/null 2>&1; then
    echo "[*] Stopping existing Ollama instance..."
    pkill ollama
    sleep 1
else
    echo "[*] No existing Ollama instance found. Continuing."
fi

echo "[*] Configuring external network access (0.0.0.0)..."
export OLLAMA_HOST="0.0.0.0"

# ==========================================
# 1. FELINE SUPERVISOR (optional, background)
# ==========================================
# Displays a random cat GIF from slave/cats/ as a motivational supervisor.
# Purely cosmetic — the benchmark runs correctly without this step.
echo "[*] Summoning feline supervisor..."

if ls "$SCRIPT_DIR/slave/cats/"*.gif 1> /dev/null 2>&1; then
    GIF_ALEATORIO=$(ls "$SCRIPT_DIR/slave/cats/"*.gif | shuf -n 1)
    echo "[*] Supervisor summoned: $GIF_ALEATORIO"
    xdg-open "$GIF_ALEATORIO" &
else
    echo "[!] Warning: No cats found in 'slave/cats/'. Working unsupervised."
fi

# ==========================================
# 2. INFERENCE ENGINE (background) + UDP LISTENER (foreground)
# ==========================================
echo "========================================="
echo "[*] Starting Ollama inference engine..."
echo "========================================="
echo ""

# Run ollama serve in the background with '&' so this script can proceed to
# launch the UDP listener. Without '&', the script would block here forever
# and worker_node.py would never start.
ollama serve &
OLLAMA_PID=$!
echo "[*] Ollama started (PID $OLLAMA_PID). Waiting for it to be ready..."

# Give Ollama time to bind its port before the UDP listener starts accepting tasks.
# This prevents a race condition where the Master sends an inference request before
# the Ollama HTTP server is ready to accept connections.
sleep 3

echo "[*] Starting Worker UDP Listener (Foreground)..."
echo "========================================="
echo ""

# Python stays in the foreground waiting for 'OLLAMA_MASTER_SEEKING' broadcasts.
# Use the absolute path derived from SCRIPT_DIR so this works regardless of CWD.
python "$SCRIPT_DIR/slave/worker_node.py"