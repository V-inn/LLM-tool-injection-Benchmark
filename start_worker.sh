#!/bin/bash
# start_worker.sh — STANDALONE Worker Node launcher for the Control Illusion Benchmark.
#
# SELF-CONTAINED: this single file is all you need on a worker machine. Copy it to
# a USB stick, drop it on any LAN box with `ollama` + `python3`, and run it — no
# repository checkout, no `pip install`, no internet. The worker daemon it runs is
# embedded below as a stdlib-only Python heredoc (written to a temp file at launch),
# so there is nothing else to fetch.
#
# WHAT IT DOES
# ============
# 1. (--daemon) Optionally re-launches itself detached (setsid + nohup) so closing
#    the terminal / SSH session does NOT kill the worker. Logs go to $RBAC_WORKER_LOG.
# 2. Starts `ollama serve` under a restart-on-crash supervisor with
#    OLLAMA_HOST=0.0.0.0 so the Master can reach the inference API over the LAN,
#    and restarts it automatically if it exits (e.g. OOM).
# 3. Runs the embedded worker daemon under its own restart loop — the UDP listener
#    that answers Master discovery ("OLLAMA_MASTER_SEEKING" -> "OLLAMA_READY") AND
#    broadcasts periodic "OLLAMA_ALIVE" heartbeats so the Master's live-membership
#    tracker can heal/rejoin this node mid-run.
#
# (The embedded daemon is a copy of rbac_benchmark/worker/node.py kept deliberately
#  dependency-free so this script can stand alone. Keep the two in sync if the
#  discovery/heartbeat protocol changes.)
#
# USAGE
# =====
#   bash start_worker.sh            # interactive (foreground); Ctrl-C to stop
#   bash start_worker.sh --daemon   # detached; survives terminal close, logs to a file
#
# For an always-on worker that also survives reboots, run with --daemon or wrap
# this script in your own systemd/service unit — optional and NOT needed for the
# standalone USB-stick workflow.
#
# CONFIG (environment)
# ====================
#   RBAC_UDP_PORT            UDP discovery/heartbeat port (default 5005; match the Master)
#   RBAC_HEARTBEAT_INTERVAL  seconds between heartbeats   (default 10; <= Master worker_stale_after)
#   RBAC_WORKER_LOG          --daemon log file            (default $TMPDIR/rbac_worker.log)
#
# REQUIREMENTS
# ============
# - ollama installed with at least one model pulled: `ollama pull <model_name>`
# - python3 (standard library only — the embedded worker has no dependencies).
# - Ports RBAC_UDP_PORT (UDP) and 11434 (TCP, Ollama API) reachable from the Master.
#
# SECURITY
# ========
# Do NOT run on untrusted networks. The discovery/heartbeat protocol has no
# authentication — the Master adds any host that answers to the inference pool.

# ------------------------------------------------------------------------------
# Optional self-detach: `--daemon` re-execs this script in a new session (setsid)
# with nohup and stdio redirected to a log, so SIGHUP from a closing terminal
# cannot reach it. The child sets RBAC_WORKER_DETACHED to avoid infinite re-exec.
# ------------------------------------------------------------------------------
TMPDIR_BASE="${TMPDIR:-/tmp}"
LOG_FILE="${RBAC_WORKER_LOG:-$TMPDIR_BASE/rbac_worker.log}"

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

# Discovery/heartbeat settings passed to the embedded daemon (defaults match the Master).
export RBAC_UDP_PORT="${RBAC_UDP_PORT:-5005}"
export RBAC_HEARTBEAT_INTERVAL="${RBAC_HEARTBEAT_INTERVAL:-10}"

# ------------------------------------------------------------------------------
# Materialise the embedded, stdlib-only worker daemon to a local temp file.
# Written to a writable temp dir (a USB stick may be mounted read-only), so this
# works no matter where the script itself lives.
# ------------------------------------------------------------------------------
WORKER_PY="$TMPDIR_BASE/rbac_worker_node_$$.py"
cat > "$WORKER_PY" <<'PYEOF'
"""Self-contained worker daemon (stdlib only) — embedded copy of
rbac_benchmark/worker/node.py. Answers UDP discovery and broadcasts OLLAMA_ALIVE
heartbeats so the Master's live-membership tracker can heal/rejoin this node."""
import os
import socket
import threading
import time

UDP_PORT = int(os.environ.get("RBAC_UDP_PORT", "5005"))
HEARTBEAT_INTERVAL = float(os.environ.get("RBAC_HEARTBEAT_INTERVAL", "10"))
BROADCAST_ADDR = "255.255.255.255"


def _heartbeat_loop(stop_event):
    hostname = socket.gethostname()
    message = f"OLLAMA_ALIVE:{hostname}".encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"[*] Heartbeat active — broadcasting OLLAMA_ALIVE every {HEARTBEAT_INTERVAL:.0f}s "
          f"to {BROADCAST_ADDR}:{UDP_PORT}")
    while not stop_event.is_set():
        try:
            sock.sendto(message, (BROADCAST_ADDR, UDP_PORT))
        except OSError as beat_error:
            print(f"[!] Heartbeat send failed, will retry: {beat_error}")
        stop_event.wait(HEARTBEAT_INTERVAL)
    sock.close()


def start_worker():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", UDP_PORT))

    stop_event = threading.Event()
    threading.Thread(target=_heartbeat_loop, args=(stop_event,), daemon=True).start()

    print(f"[*] Worker Node active — listening on UDP port {UDP_PORT}...")
    print("[*] Waiting for Master Controller broadcast...")
    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                message = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            except OSError as recv_error:
                print(f"[!] Socket error while receiving, ignoring: {recv_error}")
                continue

            if message == "OLLAMA_MASTER_SEEKING":
                print(f"[!] Discovery broadcast received from Master at: {addr[0]}")
                try:
                    sock.sendto(b"OLLAMA_READY", addr)
                except OSError as send_error:
                    print(f"[!] Failed to reply to {addr[0]}, ignoring: {send_error}")
                    continue
                time.sleep(1)
    finally:
        stop_event.set()
        sock.close()


if __name__ == "__main__":
    start_worker()
PYEOF

# ==========================================
# Feline supervisor (optional, cosmetic) — only if a cats/ dir sits next to this
# script; the standalone copy has none, so this is silently skipped.
# ==========================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CATS_DIR="$SCRIPT_DIR/cats"
[ -d "$SCRIPT_DIR/rbac_benchmark/worker/cats" ] && CATS_DIR="$SCRIPT_DIR/rbac_benchmark/worker/cats"
if [ -z "$RBAC_WORKER_DETACHED" ] && [ -n "$DISPLAY" ] && ls "$CATS_DIR/"*.gif 1> /dev/null 2>&1; then
    GIF_ALEATORIO=$(ls "$CATS_DIR/"*.gif | shuf -n 1)
    echo "[*] Supervisor summoned: $GIF_ALEATORIO"
    xdg-open "$GIF_ALEATORIO" &
fi

# ==========================================
# Restart-on-crash supervision for Ollama + the worker daemon
# ==========================================
OLLAMA_SUP_PID=""
LISTENER_PID=""
SHUTTING_DOWN=0

cleanup() {
    SHUTTING_DOWN=1
    echo ""
    echo "[*] Shutting down worker..."
    [ -n "$LISTENER_PID" ] && kill "$LISTENER_PID" 2>/dev/null
    if [ -n "$OLLAMA_SUP_PID" ]; then
        pkill -P "$OLLAMA_SUP_PID" 2>/dev/null
        kill "$OLLAMA_SUP_PID" 2>/dev/null
    fi
    rm -f "$WORKER_PY" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "========================================="
echo "[*] Starting Ollama inference engine (with auto-restart)..."
echo "========================================="

# Relaunch `ollama serve` whenever it exits (e.g. OOM kill) so a transient Ollama
# death does not permanently drop this worker from the run.
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
sleep 3

echo "[*] Starting Worker UDP Listener + Heartbeat (with auto-restart)..."
echo "========================================="

# Run the embedded daemon; restart it if it crashes. `wait` lets SIGINT/SIGTERM
# interrupt promptly and run cleanup().
while [ "$SHUTTING_DOWN" = "0" ]; do
    python3 "$WORKER_PY" &
    LISTENER_PID=$!
    wait "$LISTENER_PID"
    ec=$?
    [ "$SHUTTING_DOWN" = "1" ] && break
    echo "[!] Worker daemon exited (code $ec) — restarting in 2s..." >&2
    sleep 2
done
