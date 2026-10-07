"""HttpDecisionsAdapter: any model behind a base URL.

Wire protocol (owned by SoulsBench, identical for every HTTP model):

``POST {base_url}/decide``::

    {"schema_version": "soulsbench/v1", "episode_id": "...", "deadline_ms": 200,
     "action_labels": ["idle", "move_forward", ...16 labels...], "state": {CombatState}}

must answer ``200 {"discrete_id": <0..15>}`` (extra keys are ignored).

``POST {base_url}/reset`` with ``{"episode_id": "..."}`` is sent at episode start. It is optional: any
error (404 included) is ignored.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import ValidationError

from soulsbench.adapters.base import AdapterError, CombatAdapter, DecisionParseError
from soulsbench.schema import ACTION_LABELS, SCHEMA_VERSION, CombatState, Decision

logger = logging.getLogger(__name__)


class HttpDecisionsAdapter(CombatAdapter):
    decide_path = "/decide"

    def __init__(self, base_url: str, name: str = "http", api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(headers=headers)

    def preflight(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(f"{self.name}: base URL must be http(s)://..., got {self.base_url!r}")

    def reset(self, episode_id: str, seed: int | None = None) -> None:
        try:
            self._client.post(
                f"{self.base_url}/reset", json={"episode_id": episode_id}, timeout=2.0
            )
        except httpx.HTTPError as e:
            logger.debug("%s: /reset ignored (%s)", self.name, e)

    def encode_request(self, state: CombatState, deadline_ms: float) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": state.episode_id,
            "deadline_ms": deadline_ms,
            "action_labels": list(ACTION_LABELS),
            "state": state.model_dump(mode="json"),
        }

    def decode_response(self, payload: Any) -> Decision:
        try:
            return Decision.model_validate(payload)
        except ValidationError as e:
            raise DecisionParseError(f"invalid Decision: {e.errors()[:1]}") from None

    def decide(self, state: CombatState, deadline_ms: float) -> Decision:
        body = self.encode_request(state, deadline_ms)
        try:
            resp = self._client.post(
                f"{self.base_url}{self.decide_path}", json=body, timeout=deadline_ms / 1000
            )
        except httpx.TimeoutException:
            raise TimeoutError(f"no answer within {deadline_ms:g} ms") from None
        except httpx.HTTPError as e:
            raise AdapterError(f"{type(e).__name__}: {e}") from None
        if resp.status_code != 200:
            raise AdapterError(f"HTTP {resp.status_code}")
        try:
            payload = resp.json()
        except ValueError:
            raise DecisionParseError("response is not JSON") from None
        return self.decode_response(payload)

    def close(self) -> None:
        self._client.close()
