"""Published API prices, for the cost columns of score cards.

USD per 1M tokens (input, output), checked 2026-10-06 against each provider's pricing:
Jev (TypeSafe), OpenAI Decisions (gpt-6-luna), Perplexity Decisions (pplx-decider-v1-27b),
Cloudflare Workers AI (clef, clef-flash). All of them bill input only. Locally served models
(laya, kev) and built-in baselines cost nothing per call.
"""

from __future__ import annotations

PRICES_PER_M: dict[str, tuple[float, float]] = {
    "jev": (0.042, 0.0),
    "openai": (0.10, 0.0),
    "pplx": (0.04, 0.0),
    "clef": (0.24, 0.0),
    "clef-flash": (0.09, 0.0),
}
LOCAL = frozenset({"laya", "kev", "idle", "random", "expert", "rules"})


def base_name(adapter: str) -> str:
    """'jev@r5' -> 'jev', 'rules:150' -> 'rules', 'systemone:x' -> 'x'."""
    name = adapter.split("@")[0]
    if name.startswith("systemone:"):
        return name.split(":", 1)[1]
    return name.split(":")[0]


def cost_usd(adapter: str, input_tokens: int, output_tokens: int) -> float | None:
    """Cost of these tokens at the adapter's published price; 0 for local, None if unknown."""
    name = base_name(adapter)
    if name in LOCAL or name.split("-")[0] in LOCAL:  # e.g. laya-typed (a local checkpoint)
        return 0.0
    if name not in PRICES_PER_M:
        return None
    pin, pout = PRICES_PER_M[name]
    return (input_tokens * pin + output_tokens * pout) / 1e6
