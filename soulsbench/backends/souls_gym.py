"""WSL-side client for the Windows SoulsGym bridge.

The bridge runs as a native Windows process next to Dark Souls III and listens on Windows'
127.0.0.1. This client finds it, then speaks the JSON protocol in :mod:`soulsbench.schema` over HTTP.

Finding the bridge, in order:

1. an explicit URL (``--bridge-url`` / ``SOULSBENCH_BRIDGE_URL``),
2. ``http://127.0.0.1:<port>`` directly (WSL mirrored networking),
3. an interop relay into Windows' 127.0.0.1 (default NAT networking; no firewall rule or admin
   rights needed, see :mod:`soulsbench.backends.relay`),
4. ``http://<Windows host>:<port>`` via WSL's default gateway, only reachable if the bridge was
   started with ``--host <WSL adapter IP>`` and a firewall rule allows it.

``<port>`` is ``SOULSBENCH_BRIDGE_PORT`` or 47800.
"""

from __future__ import annotations

import os
import socket
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import ValidationError

from soulsbench.backends.base import EnvError
from soulsbench.backends.relay import InteropRelay, windows_bridge_python
from soulsbench.schema import (
    BridgeHealth,
    RecordStartRequest,
    RecordStopRequest,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)

DEFAULT_PORT = 47800
RESET_TIMEOUT_S = 240.0  # a reset may reload the game (death -> respawn -> fog gate)
STEP_TIMEOUT_S = 30.0
RELAY_PROBE_TIMEOUT_S = 15.0  # first relay connection starts a Windows Python process


class BridgeUnavailable(RuntimeError):
    pass


def wsl_windows_host_ip(route_file: Path = Path("/proc/net/route")) -> str | None:
    """The default gateway, which is the Windows host under WSL2 NAT networking."""
    try:
        lines = route_file.read_text().splitlines()[1:]
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        if len(fields) > 2 and fields[1] == "00000000":
            return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    return None


@dataclass
class BridgeConnection:
    url: str
    via: str  # explicit | direct | relay | wsl-gateway
    relay: InteropRelay | None = None

    def close(self) -> None:
        if self.relay is not None:
            self.relay.close()


def _probe(url: str, timeout_s: float) -> str | None:
    """None if the bridge at ``url`` answers /health, else a short reason."""
    try:
        httpx.get(f"{url}/health", timeout=timeout_s).raise_for_status()
        return None
    except httpx.HTTPError as e:
        return type(e).__name__


def connect_bridge(
    explicit: str | None = None,
    env: Mapping[str, str] | None = None,
    probe_timeout_s: float = 1.0,
) -> BridgeConnection:
    env = os.environ if env is None else env
    explicit = explicit or env.get("SOULSBENCH_BRIDGE_URL")
    if explicit:
        url = explicit.rstrip("/")
        if (err := _probe(url, probe_timeout_s)) is None:
            return BridgeConnection(url, "explicit")
        raise BridgeUnavailable(f"No SoulsBench bridge answered at {url} ({err}).")
    port = int(env.get("SOULSBENCH_BRIDGE_PORT", DEFAULT_PORT))
    tried = []
    direct = f"http://127.0.0.1:{port}"
    if (err := _probe(direct, probe_timeout_s)) is None:
        return BridgeConnection(direct, "direct")
    tried.append(f"{direct} ({err})")
    python = windows_bridge_python()
    if python:
        relay = InteropRelay(python, port)
        if (err := _probe(relay.url, RELAY_PROBE_TIMEOUT_S)) is None:
            return BridgeConnection(relay.url, "relay", relay)
        relay.close()
        tried.append(f"interop relay to Windows 127.0.0.1:{port} ({err})")
    else:
        tried.append("interop relay (no Windows bridge python; run windows/setup_bridge.ps1)")
    gateway = wsl_windows_host_ip()
    if gateway:
        url = f"http://{gateway}:{port}"
        if (err := _probe(url, probe_timeout_s)) is None:
            return BridgeConnection(url, "wsl-gateway")
        tried.append(f"{url} ({err})")
    raise BridgeUnavailable(
        "No SoulsBench bridge answered: " + "; ".join(tried) + ". Start it on Windows with "
        "windows\\run_bridge.ps1 or `python -m soulsbench.bridge start` (see README)."
    )


class SoulsGymBridgeEnv:
    """CombatEnv over HTTP to the Windows bridge."""

    name = "souls_gym"

    def __init__(self, bridge_url: str | None = None) -> None:
        self.connection = connect_bridge(bridge_url)
        self.url = self.connection.url
        self.via = self.connection.via
        self._client = httpx.Client(base_url=self.url)

    def health(self) -> BridgeHealth:
        return BridgeHealth.model_validate(self._call("GET", "/health", timeout=5.0))

    def spec(self) -> dict:
        return self._call("GET", "/spec", timeout=5.0)

    def reset(self, request: ResetRequest) -> ResetResponse:
        data = self._call("POST", "/reset", json=request.model_dump(), timeout=RESET_TIMEOUT_S)
        return self._parse(ResetResponse, data)

    def step(self, discrete_id: int) -> StepResponse:
        body = StepRequest(discrete_id=discrete_id).model_dump()
        return self._parse(
            StepResponse, self._call("POST", "/step", json=body, timeout=STEP_TIMEOUT_S)
        )

    def record_start(self, name: str, label: str, subdir: str, password: str) -> None:
        body = RecordStartRequest(name=name, label=label, subdir=subdir, password=password)
        self._call("POST", "/record/start", json=body.model_dump(), timeout=15.0)

    def record_stop(self, suffix: str) -> str | None:
        body = RecordStopRequest(suffix=suffix).model_dump()
        return self._call("POST", "/record/stop", json=body, timeout=30.0).get("path")

    def act(self, discrete_id: int) -> StepResponse:
        body = StepRequest(discrete_id=discrete_id).model_dump()
        return self._parse(
            StepResponse, self._call("POST", "/act", json=body, timeout=STEP_TIMEOUT_S)
        )

    def close(self) -> None:
        try:
            self._call("POST", "/close", json={}, timeout=30.0)
        except EnvError:
            pass
        finally:
            self._client.close()
            self.connection.close()

    def _call(self, method: str, path: str, *, timeout: float, json: dict | None = None) -> dict:
        try:
            resp = self._client.request(method, path, json=json, timeout=timeout)
        except httpx.HTTPError as e:
            raise EnvError(f"bridge {method} {path} failed: {type(e).__name__}: {e}") from None
        try:
            data = resp.json()
        except ValueError:
            raise EnvError(
                f"bridge {method} {path}: HTTP {resp.status_code}, non-JSON body"
            ) from None
        if resp.status_code != 200:
            raise EnvError(f"bridge {method} {path}: {data.get('error')}: {data.get('message')}")
        return data

    @staticmethod
    def _parse(model, data: dict):
        try:
            return model.model_validate(data)
        except ValidationError as e:
            raise EnvError(f"bridge sent an invalid {model.__name__}: {e.errors()[:2]}") from None
