"""
worker_node.py — UDP Discovery Responder + Heartbeat for the Control Illusion Benchmark.

PURPOSE
=======
This script runs on every machine that will contribute Ollama inference capacity
to a benchmark run. It listens on a UDP port for broadcast discovery messages from
the Master Node and replies to announce that this machine's Ollama instance is
available, AND it proactively broadcasts a periodic heartbeat so the Master's
live-membership tracker can tell — at any moment during a run — whether this node
is still alive.

DISCOVERY + LIVENESS PROTOCOL
=============================
Two complementary mechanisms, both over the same UDP port:

    1. Pull (discovery): the Master broadcasts "OLLAMA_MASTER_SEEKING" to
       255.255.255.255:5005. Each Worker replies "OLLAMA_READY" directly to the
       Master (unicast). The Master forms/refreshes the cluster from the replies.

    2. Push (heartbeat): every HEARTBEAT_INTERVAL seconds, each Worker broadcasts
       "OLLAMA_ALIVE:<hostname>" to 255.255.255.255:5005. The Master's heartbeat
       listener refreshes that worker's last-seen timestamp. A worker that goes
       silent past the Master's staleness window is evicted from dispatch (its
       in-flight work is re-queued to healthy nodes); when its heartbeats resume —
       e.g. after a restart — the Master auto-rejoins it. This is what lets a
       cluster HEAL and GROW mid-run without restarting the benchmark.

Both are backward compatible: an old Master that only broadcasts discovery still
works (it just ignores the heartbeats), and a Master with membership tracking
also tolerates an old worker that only answers discovery (it refreshes last-seen
from the discovery reply instead of the heartbeat).

After discovery, the Master sends inference to workers over the Ollama HTTP API
(port 11434). This script never touches that path — it only handles cluster
formation and liveness.

CONFIGURATION (environment overrides, so start_worker.sh can pass config values)
================================================================================
    RBAC_UDP_PORT            UDP discovery/heartbeat port   (default 5005)
    RBAC_HEARTBEAT_INTERVAL  seconds between heartbeats      (default 10)

Keep RBAC_UDP_PORT in sync with the Master's config.udp_discovery_port, and
RBAC_HEARTBEAT_INTERVAL at or below the Master's worker_stale_after.

SECURITY NOTE
=============
There is intentionally no authentication on this protocol. Any host on the LAN
that answers discovery or heartbeats will be added to the master's worker pool and
will receive inference workloads. This is an acceptable trade-off for a controlled
research environment (local cluster or trusted LAN). Do NOT run start_worker.sh on
an untrusted or public network.

DEPLOYMENT
==========
Use start_worker.sh (repo root) to launch this script together with the Ollama
inference engine. The script daemonizes both so closing the terminal does not kill
them, and restarts Ollama if it exits.
"""

import os
import socket
import threading
import time

UDP_PORT = int(os.environ.get("RBAC_UDP_PORT", "5005"))
HEARTBEAT_INTERVAL = float(os.environ.get("RBAC_HEARTBEAT_INTERVAL", "10"))
BROADCAST_ADDR = "255.255.255.255"


def _heartbeat_loop(stop_event: threading.Event):
    """
    Periodically broadcast "OLLAMA_ALIVE:<hostname>" so the Master's membership
    tracker keeps this node marked healthy. Runs on a daemon thread alongside the
    discovery listener. A transient network error (e.g. Wi-Fi renegotiation) is
    logged and retried on the next tick — a missed beat is tolerated by the
    Master's staleness window.
    """
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
        # Wait on the event so shutdown is prompt, but wake every interval to beat.
        stop_event.wait(HEARTBEAT_INTERVAL)

    sock.close()


def start_worker():
    """
    Binds a UDP socket and enters the discovery listen loop, with a background
    heartbeat thread running in parallel.

    Binds to 0.0.0.0 to listen on all network interfaces simultaneously —
    this ensures the worker is reachable whether the Master broadcasts over
    Wi-Fi, Ethernet, or any other interface present on the machine.

    The 1-second sleep after each reply prevents packet flooding if the Master
    broadcasts multiple times in quick succession (e.g. during a retry).
    """
    UDP_IP = "0.0.0.0"  # Listen on all network interfaces

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Allow a quick restart to re-bind the port without a TIME_WAIT stall.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((UDP_IP, UDP_PORT))

    # Start the heartbeat before we block on recvfrom so liveness begins immediately.
    stop_event = threading.Event()
    hb_thread = threading.Thread(target=_heartbeat_loop, args=(stop_event,), daemon=True)
    hb_thread.start()

    print(f"[*] Worker Node active — listening on UDP port {UDP_PORT}...")
    print("[*] Waiting for Master Controller broadcast...")

    try:
        while True:
            # Block until a UDP packet arrives on the discovery port.
            # Port 5005 overlaps common RTP/media traffic, so stray binary packets
            # from other LAN hosts are expected — never let one kill the listener.
            # Our own broadcast heartbeats also land here and are simply ignored
            # (they are not "OLLAMA_MASTER_SEEKING").
            try:
                data, addr = sock.recvfrom(1024)
                message = data.decode('utf-8')
            except UnicodeDecodeError:
                continue
            except OSError as recv_error:
                print(f"[!] Socket error while receiving, ignoring: {recv_error}")
                continue

            master_ip = addr[0]

            if message == "OLLAMA_MASTER_SEEKING":
                print(f"[!] Discovery broadcast received from Master at: {master_ip}")
                print(f"[+] Sending readiness confirmation...")

                # Reply directly to the Master's IP (unicast). The reply must contain
                # exactly "OLLAMA_READY" — the Master checks for this exact string.
                # A transient network drop (e.g. Wi-Fi renegotiation) raises OSError;
                # the master will simply retry discovery, so log and keep listening.
                response = "OLLAMA_READY"
                try:
                    sock.sendto(response.encode('utf-8'), addr)
                except OSError as send_error:
                    print(f"[!] Failed to reply to {master_ip}, ignoring: {send_error}")
                    continue

                # Brief delay to avoid reply packet flooding if the Master sends
                # multiple rapid broadcasts (e.g. during discovery retries).
                time.sleep(1)
    finally:
        stop_event.set()
        sock.close()


if __name__ == "__main__":
    start_worker()
