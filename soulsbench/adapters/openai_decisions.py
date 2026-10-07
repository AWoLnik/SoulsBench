"""OpenAI Decisions API (``gpt-6-luna``) behind the CombatAdapter contract.

Docs: https://developers.openai.com/api/docs/guides/decisions (public beta, 2026-10).
``POST https://api.openai.com/v1/decisions`` with ``Authorization: Bearer <key>``::

    {"model": "gpt-6-luna",
     "input": "<text>",
     "questions": [{"type": "choice", "name": "action", "instructions": "...",
                    "choices": [{"value": "<label>", "description": "..."}, ...]}]}

answers ``{"answers": [{"type": "choice", "name": "action", "choice": "<label>", ...}]}``.

It is the same kind of model as Jev (state + typed questions -> a choice), so it reuses the System
One adapter: the same prompts (tuned / r5 / base), state projection and offered actions, only
re-shaped into OpenAI's request. ``input`` takes text or messages, not an object, so the state is
sent as compact JSON text. Results are therefore directly comparable with jev, pplx, laya and kev.

Key: ``OPENAI_API_KEY`` or ``./openai_api_key.txt`` (gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from soulsbench.adapters.base import DecisionParseError
from soulsbench.adapters.jev import SystemOneAdapter
from soulsbench.schema import ACTION_LABELS, CombatState, Decision

OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MODEL = "gpt-6-luna"
KEY_FILE = Path("openai_api_key.txt")


class OpenAIDecisionsAdapter(SystemOneAdapter):
    def __init__(
        self,
        api_key: str | None,
        prompt: str = "tuned",
        base_url: str = OPENAI_BASE_URL,
        model: str = OPENAI_MODEL,
    ) -> None:
        super().__init__(
            "openai" if prompt == "tuned" else f"openai@{prompt}",
            base_url,
            model,
            api_key=api_key,
            prompt=prompt,
            needs_key=True,
            decide_path="/decisions",
        )

    def encode_request(self, state: CombatState, deadline_ms: float) -> dict[str, Any]:
        jev_style = super().encode_request(state, deadline_ms)
        question = jev_style["questions"]["action"]
        return {
            "model": self.model,
            "input": json.dumps(jev_style["state"], separators=(",", ":")),
            "questions": [
                {
                    "type": "choice",
                    "name": "action",
                    "instructions": question["instructions"],
                    "choices": [
                        {"value": label, "description": text}
                        for label, text in question["criteria"].items()
                    ],
                }
            ],
        }

    def decode_response(self, payload: Any) -> Decision:
        try:
            answer = next(a for a in payload["answers"] if a.get("name") == "action")
            return Decision(
                discrete_id=ACTION_LABELS.index(answer["choice"]), usage=payload.get("usage")
            )
        except (KeyError, TypeError, ValueError, StopIteration):
            raise DecisionParseError(f"unexpected openai response: {str(payload)[:200]}") from None
