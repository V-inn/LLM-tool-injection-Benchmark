"""
worker_node.py — UDP Discovery Responder for the Control Illusion Benchmark.

PURPOSE
=======
This script runs on every machine that will contribute Ollama inference capacity
to a benchmark run. It listens on a UDP port for broadcast discovery messages from
the Master Node and replies to announce that this machine's Ollama instance is
available and ready to accept inference requests.

DISCOVERY PROTOCOL
==================
The handshake is deliberately minimal:
    1. Master broadcasts "OLLAMA_MASTER_SEEKING" to 255.255.255.255:5005.
    2. Each Worker that receives the broadcast replies with "OLLAMA_READY" directly
       to the Master's IP address (unicast, not broadcast).
    3. The Master collects all replies within its configured timeout window, deduplicates
       IPs, and forms the cluster worker list.

After discovery, the Master communicates with workers exclusively via the Ollama
HTTP API (port 11434). This script is only needed for the initial cluster formation;
it can remain running continuously without consuming meaningful resources.

SECURITY NOTE
=============
There is intentionally no authentication on this protocol. Any host on the LAN that
responds with "OLLAMA_READY" will be added to the master's worker pool and will receive
inference workloads. This is an acceptable trade-off for a controlled research
environment (local cluster or trusted LAN). Do NOT run start_worker.sh on an
untrusted or public network.

DEPLOYMENT
==========
Use start_worker.sh (repo root) to launch this script together with the Ollama
inference engine. The shell script handles:
    - Starting ollama serve in the background.
    - Setting OLLAMA_HOST=0.0.0.0 so the master can reach the inference endpoint.
    - Starting this listener in the foreground.
"""

import socket
import time


def start_worker():
    """
    Binds a UDP socket and enters the discovery listen loop.

    Binds to 0.0.0.0 to listen on all network interfaces simultaneously —
    this ensures the worker is reachable whether the Master broadcasts over
    Wi-Fi, Ethernet, or any other interface present on the machine.

    The 1-second sleep after each reply prevents packet flooding if the Master
    broadcasts multiple times in quick succession (e.g. during a retry).
    """
    UDP_IP = "0.0.0.0"  # Listen on all network interfaces
    UDP_PORT = 5005      # Must match config.udp_discovery_port on the Master

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"[*] Worker Node active — listening on UDP port {UDP_PORT}...")
    print("[*] Waiting for Master Controller broadcast...")

    while True:
        # Block until a UDP packet arrives on the discovery port.
        # Port 5005 overlaps common RTP/media traffic, so stray binary packets
        # from other LAN hosts are expected — never let one kill the listener.
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


if __name__ == "__main__":
    start_worker()