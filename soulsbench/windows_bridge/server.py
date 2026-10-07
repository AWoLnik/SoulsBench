"""The bridge's HTTP server (stdlib only). Backend-agnostic: it serves any ``BridgeBackend``.

Endpoints (JSON bodies, see :mod:`soulsbench.schema`):

* ``GET  /health`` -> BridgeHealth
* ``GET  /spec``   -> action labels, scenarios, default SLA
* ``POST /reset``  ResetRequest -> ResetResponse
* ``POST /step``   StepRequest  -> StepResponse (paused mode: play one control period)
* ``POST /act``    StepRequest  -> StepResponse (realtime: apply now, return the state right after)
* ``POST /record/start`` RecordStartRequest, ``POST /record/stop`` RecordStopRequest (OBS)
* ``POST /close``  end of a harness session (release keys, keep the game paused)
* ``POST /shutdown`` ask the bridge process to exit cleanly (restores the game, see ``serve``)

All game access is serialized by one lock: the game is a single shared resource.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from soulsbench.scenarios import SCENARIOS
from soulsbench.schema import (
    ACTION_LABELS,
    DEFAULT_SLA,
    SCHEMA_VERSION,
    BridgeHealth,
    RecordStartRequest,
    RecordStopRequest,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from soulsbench.windows_bridge.errors import BridgeNotReady, BridgeStateError

logger = logging.getLogger(__name__)


class BridgeBackend(Protocol):
    backend: str

    def health(self) -> BridgeHealth: ...

    def reset(self, req: ResetRequest) -> ResetResponse: ...

    def step(self, req: StepRequest) -> StepResponse: ...

    def act(self, req: StepRequest) -> StepResponse: ...

    def record_start(self, req: RecordStartRequest) -> None: ...

    def record_stop(self, req: RecordStopRequest) -> str | None: ...

    def close(self) -> None: ...


def spec_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "action_labels": list(ACTION_LABELS),
        "n_actions": len(ACTION_LABELS),
        "scenarios": {name: s.model_dump() for name, s in SCENARIOS.items()},
        "default_sla": DEFAULT_SLA.model_dump(),
    }


def make_handler(
    backend: BridgeBackend, lock: threading.Lock, shutdown: threading.Event
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: one TCP connection for a whole episode
        server_version = "SoulsBenchBridge"

        def log_message(self, format: str, *args: Any) -> None:
            logger.debug("%s - %s", self.address_string(), format % args)

        def _send(self, status: int, payload: dict[str, Any] | BaseModel) -> None:
            if isinstance(payload, BaseModel):
                payload = payload.model_dump(mode="json")
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: HTTPStatus, error: str, message: str) -> None:
            self._send(status, {"error": error, "message": message})

        def _body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw or b"{}")

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(HTTPStatus.OK, backend.health())
            elif self.path == "/spec":
                self._send(HTTPStatus.OK, spec_payload())
            else:
                self._error(HTTPStatus.NOT_FOUND, "not_found", self.path)

        def do_POST(self) -> None:
            try:
                body = self._body()
            except ValueError as e:
                self._error(HTTPStatus.BAD_REQUEST, "bad_json", str(e))
                return
            if self.path == "/shutdown":  # no lock: must work while a long reset is running
                shutdown.set()
                self._send(HTTPStatus.OK, {"ok": True})
                return
            try:
                with lock:
                    if self.path == "/reset":
                        result = backend.reset(ResetRequest.model_validate(body))
                    elif self.path == "/step":
                        result = backend.step(StepRequest.model_validate(body))
                    elif self.path == "/act":
                        result = backend.act(StepRequest.model_validate(body))
                    elif self.path == "/record/start":
                        backend.record_start(RecordStartRequest.model_validate(body))
                        result = {"ok": True}
                    elif self.path == "/record/stop":
                        path = backend.record_stop(RecordStopRequest.model_validate(body))
                        result = {"ok": True, "path": path}
                    elif self.path == "/close":
                        backend.close()
                        result = {"ok": True}
                    else:
                        self._error(HTTPStatus.NOT_FOUND, "not_found", self.path)
                        return
            except ValidationError as e:
                self._error(HTTPStatus.BAD_REQUEST, "bad_request", str(e.errors()[:2]))
            except ValueError as e:
                self._error(HTTPStatus.BAD_REQUEST, "bad_request", str(e))
            except BridgeNotReady as e:
                self._error(HTTPStatus.SERVICE_UNAVAILABLE, "not_ready", str(e))
            except BridgeStateError as e:
                self._error(HTTPStatus.CONFLICT, "conflict", str(e))
            except Exception as e:  # game/SoulsGym failure: report it, keep serving
                logger.error("%s failed:\n%s", self.path, traceback.format_exc())
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, type(e).__name__, str(e))
            else:
                self._send(HTTPStatus.OK, result)

    return Handler


class BridgeServer:
    """One ThreadingHTTPServer per bind address, all sharing one backend and one lock."""

    def __init__(self, backend: BridgeBackend, hosts: list[str], port: int) -> None:
        self.backend = backend
        self.shutdown_requested = threading.Event()
        handler = make_handler(backend, threading.Lock(), self.shutdown_requested)
        self.servers = [ThreadingHTTPServer((host, port), handler) for host in hosts]
        for s in self.servers:
            s.daemon_threads = True
        self._threads: list[threading.Thread] = []

    @property
    def addresses(self) -> list[str]:
        return [f"http://{s.server_address[0]}:{s.server_address[1]}" for s in self.servers]

    def start(self) -> None:
        for s in self.servers:
            t = threading.Thread(
                target=s.serve_forever, name=f"bridge-{s.server_address}", daemon=True
            )
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        for s in self.servers:
            s.shutdown()
            s.server_close()
