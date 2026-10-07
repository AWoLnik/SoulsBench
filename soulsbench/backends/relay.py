"""Reach Windows' 127.0.0.1 from WSL2 NAT networking, with no firewall rule and no admin rights.

Under WSL2's default NAT networking, WSL cannot connect to Windows' loopback, and anything bound to
the Windows side of the WSL adapter needs a firewall exception. WSL *can* start Windows programs,
though. For every TCP connection accepted on a WSL-local port, :class:`InteropRelay` starts a tiny
Python process on Windows (the bridge venv's interpreter) that connects to Windows'
``127.0.0.1:<port>`` and pumps bytes through its stdin/stdout. HTTP clients in WSL then talk to the
WSL-local port as if it were the bridge. Connections are kept alive, so this costs one process start
per connection, not per request.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
from functools import cache
from pathlib import Path

# Runs on Windows. Plain stdlib; stdin/stdout are binary at the fd level in Python 3.
RELAY_CODE = r"""
import os, socket, sys, threading
s = socket.create_connection((sys.argv[1], int(sys.argv[2])))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
def up():
    while True:
        data = os.read(0, 65536)
        if not data:
            break
        s.sendall(data)
    s.shutdown(socket.SHUT_WR)
threading.Thread(target=up, daemon=True).start()
while True:
    data = s.recv(65536)
    if not data:
        break
    while data:
        data = data[os.write(1, data):]
"""


@cache
def windows_bridge_python() -> str | None:
    """The bridge venv's python.exe as a WSL path, or None. ``SOULSBENCH_WINDOWS_PYTHON`` overrides."""
    override = os.environ.get("SOULSBENCH_WINDOWS_PYTHON")
    if override:
        return override
    try:
        out = subprocess.run(
            ["cmd.exe", "/c", "echo %LOCALAPPDATA%"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="/mnt/c" if Path("/mnt/c").is_dir() else None,
        )
        local = out.stdout.strip()
        if not local or "%" in local:
            return None
        wsl = subprocess.run(["wslpath", "-u", local], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    python = Path(wsl.stdout.strip()) / "soulsbench" / "venv" / "Scripts" / "python.exe"
    return str(python) if python.exists() else None


def _write_all(fd: int, data: bytes) -> None:
    while data:
        data = data[os.write(fd, data) :]


class InteropRelay:
    """A WSL-local TCP port that forwards every connection to ``host:port`` on Windows."""

    def __init__(self, python: str, port: int, host: str = "127.0.0.1") -> None:
        self._cmd = [python, "-c", RELAY_CODE, host, str(port)]
        self._server = socket.create_server(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self._server.getsockname()[1]}"
        threading.Thread(target=self._accept_loop, name="relay-accept", daemon=True).start()

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:  # closed
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            cwd="/mnt/c" if Path("/mnt/c").is_dir() else None,
        )

        def upstream() -> None:
            try:
                while data := conn.recv(65536):
                    _write_all(proc.stdin.fileno(), data)
            except OSError:
                pass
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        threading.Thread(target=upstream, daemon=True).start()
        try:
            while data := os.read(proc.stdout.fileno(), 65536):
                conn.sendall(data)
        except OSError:
            pass
        finally:
            conn.close()
            proc.kill()
            proc.wait()

    def close(self) -> None:
        self._server.close()
