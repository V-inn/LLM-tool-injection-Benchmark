#!/bin/bash
# start_worker.sh — Deployment script for a Worker Node in the Control Illusion Benchmark.
#
# PURPOSE
# =======
# Prepares and launches a Worker Node on any LAN-connected machine.
# A Worker Node contributes its Ollama inference capacity to the distributed
# benchmark cluster coordinated by the Master Node (master_node.py / the web UI).
#
# WHAT THIS SCRIPT DOES
# ======================
# 1. Resolves the absolute path of this script's directory so all relative paths
#    work correctly regardless of where the script is called from.
# 2. (--daemon) Optionally re-launches itself detached (setsid + nohup) so closing
#    the terminal / SSH session does NOT kill the worker. Logs go to worker.log.
# 3. Starts `ollama serve` under a restart-on-crash supervisor with
#    OLLAMA_HOST=0.0.0.0 so the Master can reach the inference API over the LAN,
#    and restarts it automatically if it exits (e.g. OOM).
# 4. Launches worker_node.py under its own restart loop — the UDP listener that
#    answers Master discovery AND broadcasts periodic OLLAMA_ALIVE heartbeats so
#    the Master's live-membership tracker can heal/rejoin this node mid-run.
#
# USAGE
# =====
#   bash start_worker.sh            # interactive (foreground); Ctrl-C to stop
#   bash start_worker.sh --daemon   # detached; survives terminal close, logs to worker.log
#
# For an always-on worker that also survives reboots, install the systemd unit:
#   sudo cp deploy/rbac-worker.service /etc/systemd/system/
#   sudo systemctl enable --now rbac-worker
# (edit the User= and WorkingDirectory= in the unit first — see deploy/rbac-worker.service).
#
# CONFIG (environment; forwarded to node.py)
# ==========================================
#   RBAC_UDP_PORT            UDP discovery/heartbeat port (default 5005; match the Master)
#   RBAC_HEARTBEAT_INTERVAL  seconds between heartbeats   (default 10; <= Master worker_stale_after)
#
# REQUIREMENTS
# ============
# - ollama installed with at least one model pulled: `ollama pull <model_name>`
# - python3 (stdlib only — the worker listener has no pip dependencies).
# - Ports RBAC_UDP_PORT (UDP) and 11434 (TCP, Ollama API) reachable from the Master.
#
# SECURITY
# ========
# Do NOT run on untrusted networks. The discovery/heartbeat protocol has no
# authentication — the Master adds any host that answers to the inference pool.
# See rbac_benchmark/worker/node.py for the full security note.

# Resolve the absolute directory of this script so all derived paths are portable
# (works when called from any working directory, e.g. via SSH from the Master).
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/worker.log"

# ------------------------------------------------------------------------------
# Optional self-detach: `--daemon` re-execs this script in a new session (setsid)
# with nohup and stdio redirected to worker.log, so SIGHUP from a closing terminal
# cannot reach it. The child sets RBAC_WORKER_DETACHED to avoid infinite re-exec.
# ------------------------------------------------------------------------------
DAEMONIZE=0
for arg in "$@"; do
    case "$arg" in
        --daemon|-d) DAEMONIZE=1 ;;
    esac
done

if [ "$DAEMONIZE" = "1" ] && [ -z "$RBAC_WORKER_DETACHED" ]; then
    export RBAC_WORKER_DETACHED=1
    echo "[*] Detaching worker into its own session; logs -> $LOG_FILE"
    setsid nohup bash "$0" "$@" > "$LOG_FILE" 2>&1 < /dev/null &
    echo "[*] Worker detached as PID $!. It will survive this terminal closing."
    echo "[*] Follow logs with:  tail -f \"$LOG_FILE\""
    exit 0
fi

echo "========================================="
echo "  LLM RED TEAM WORKER - FELINE PROTOCOL  "
echo "========================================="

echo "[*] Configuring external network access (0.0.0.0)..."
export OLLAMA_HOST="0.0.0.0"

# Keep at most ONE model resident in memory. The benchmark's shared task queue can
# hand this node a different model at model boundaries (and on retries), and Ollama's
# default keeps several models loaded simultaneously — on memory-tight machines that
# triggers the Linux OOM killer, which takes Ollama down mid-run.
export OLLAMA_MAX_LOADED_MODELS=1

# Discovery/heartbeat settings forwarded to node.py (defaults match the Master).
export RBAC_UDP_PORT="${RBAC_UDP_PORT:-5005}"
export RBAC_HEARTBEAT_INTERVAL="${RBAC_HEARTBEAT_INTERVAL:-10}"

# ==========================================
# 1. FELINE SUPERVISOR (optional, cosmetic — skipped when detached / headless)
# ==========================================
if [ -z "$RBAC_WORKER_DETACHED" ] && [ -n "$DISPLAY" ]; then
    echo "[*] Summoning feline supervisor..."
    CATS_DIR="$SCRIPT_DIR/rbac_benchmark/worker/cats"
    if ls "$CATS_DIR/"*.gif 1> /dev/null 2>&1; then
        GIF_ALEATORIO=$(ls "$CATS_DIR/"*.gif | shuf -n 1)
        echo "[*] Supervisor summoned: $GIF_ALEATORIO"
        xdg-open "$GIF_ALEATORIO" &
    else
        echo "[!] Warning: No cats found in '$CATS_DIR'. Working unsupervised."
    fi
fi

# ==========================================
# 2. Restart-on-crash supervision for Ollama + the UDP listener
# ==========================================
# Track child PIDs so a clean shutdown (Ctrl-C / SIGTERM) tears everything down
# instead of leaving orphaned Ollama/listener processes behind.
OLLAMA_SUP_PID=""
LISTENER_PID=""
SHUTTING_DOWN=0

cleanup() {
    SHUTTING_DOWN=1
    echo ""
    echo "[*] Shutting down worker..."
    [ -n "$LISTENER_PID" ]   && kill "$LISTENER_PID"   2>/dev/null
    if [ -n "$OLLAMA_SUP_PID" ]; then
        # Kill the supervisor subshell and the ollama child it spawned.
        pkill -P "$OLLAMA_SUP_PID" 2>/dev/null
        kill "$OLLAMA_SUP_PID" 2>/dev/null
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "========================================="
echo "[*] Starting Ollama inference engine (with auto-restart)..."
echo "========================================="

# Ollama supervisor: relaunch `ollama serve` whenever it exits (e.g. OOM kill),
# so a transient Ollama death does not permanently drop this worker from the run.
(
    while true; do
        ollama serve
        ec=$?
        echo "[!] Ollama serve exited (code $ec) — restarting in 2s..." >&2
        sleep 2
    done
) &
OLLAMA_SUP_PID=$!
echo "[*] Ollama supervisor started (PID $OLLAMA_SUP_PID)."

# Give Ollama time to bind its port before the listener starts accepting tasks.
# Prevents a race where the Master sends an inference request before the Ollama
# HTTP server is ready to accept connections.
sleep 3

echo "[*] Starting Worker UDP Listener + Heartbeat (with auto-restart)..."
echo "========================================="

# Listener supervisor loop. node.py is stdlib-only, so no pip install is needed.
# `wait` lets SIGINT/SIGTERM interrupt promptly and run cleanup().
while [ "$SHUTTING_DOWN" = "0" ]; do
    python3 "$SCRIPT_DIR/rbac_benchmark/worker/node.py" &
    LISTENER_PID=$!
    wait "$LISTENER_PID"
    ec=$?
    [ "$SHUTTING_DOWN" = "1" ] && break
    echo "[!] Worker listener exited (code $ec) — restarting in 2s..." >&2
    sleep 2
done
