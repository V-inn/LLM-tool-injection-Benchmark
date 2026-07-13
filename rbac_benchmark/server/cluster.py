"""
server/cluster.py — live worker-node tracker for the sidebar.

The master's authoritative membership table lives inside the master SUBPROCESS
(orchestration.master_node), which the web server cannot read directly. So the
server maintains its OWN lightweight view by periodically broadcasting the same
UDP discovery probe the master uses and recording who answers. This works whether
or not a benchmark is running, so the sidebar shows the cluster forming as soon as
workers come online — exactly the "connect and watch the dots" view the UI wants.

Why an active probe rather than listening for the workers' OLLAMA_ALIVE heartbeat:
the master binds the discovery UDP port during a run, so a second listener on the
same host would collide. A broadcast probe uses an ephemeral source port instead —
no binding, no conflict — and reuses the worker daemon's existing discovery reply.

A node, once seen, stays in the list and flips to "offline" (red) when it stops
answering, rather than disappearing — so a worker dropping out is visible.
"""
from __future__ import annotations

import os
import socket
import threading
import time

# Discovery/heartbeat port — keep in sync with BenchmarkConfig.udp_discovery_port
# and the worker's RBAC_UDP_PORT (both default 5005).
UDP_PORT = int(os.environ.get("RBAC_UDP_PORT", "5005"))

# Probe cadence and the window after which a silent node is considered offline.
# STALE_AFTER spans a few probe cycles so one dropped UDP reply doesn't flap a
# healthy node to red.
_PROBE_TIMEOUT = 1.5
_POLL_INTERVAL = 4.0
STALE_AFTER = 13.0

# ip -> {"first_seen": float, "last_seen": float}
_nodes: dict[str, dict] = {}
_lock = threading.Lock()
_started = False


def _discover_once(port: int, timeout: float) -> list[str]:
    """Broadcast one discovery probe and collect the IPs that reply OLLAMA_READY.

    Uses an ephemeral source port (no bind), so it never collides with the
    master's heartbeat listener on the discovery port during a run.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)
    found: list[str] = []
    try:
        sock.sendto(b"OLLAMA_MASTER_SEEKING", ("255.255.255.255", port))
        while True:
            try:
                data, addr = sock.recvfrom(1024)
            except socket.timeout:
                break
            if data.decode("utf-8", "ignore") == "OLLAMA_READY" and addr[0] not in found:
                found.append(addr[0])
    except OSError:
        pass
    finally:
        sock.close()
    return found


def _poll_loop() -> None:
    while True:
        responders = _discover_once(UDP_PORT, _PROBE_TIMEOUT)
        now = time.time()
        with _lock:
            for ip in responders:
                node = _nodes.get(ip)
                if node is None:
                    _nodes[ip] = {"first_seen": now, "last_seen": now}
                else:
                    node["last_seen"] = now
        time.sleep(_POLL_INTERVAL)


def start() -> None:
    """Start the background discovery poller once (idempotent)."""
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_poll_loop, name="cluster-monitor", daemon=True).start()


def snapshot() -> list[dict]:
    """Current node view for the sidebar: sorted, each with a live healthy flag."""
    now = time.time()
    with _lock:
        return [
            {
                "ip": ip,
                "healthy": (now - node["last_seen"]) < STALE_AFTER,
                "last_seen": node["last_seen"],
                "first_seen": node["first_seen"],
            }
            for ip, node in sorted(_nodes.items())
        ]
