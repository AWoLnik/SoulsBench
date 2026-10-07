"""Adapter registry.

``make_adapter`` takes the ``--adapter`` spec:

* ``idle`` | ``random`` | ``expert``  built-in baselines
* ``rules`` | ``rules:<ms>``  the tuned System One prompt's rules run locally, no model (its
                   control); ``:<ms>`` delays each answer to match a model's latency
* ``http``         base URL from ``--http-base-url`` or ``SOULSBENCH_HTTP_BASE_URL``
* ``http:<name>``  base URL from ``--http-base-url`` or ``SOULSBENCH_<NAME>_BASE_URL``,
                   e.g. ``http:jev`` reads ``SOULSBENCH_JEV_BASE_URL``. Optional bearer token from
                   ``SOULSBENCH_<NAME>_API_KEY``.
* ``jev``          Jev (TypeSafe AI System One), native API; key from JEV_AI_API_KEY or ./jev_api_key.txt
* ``laya`` | ``kev``  locally served System One models (defaults: 127.0.0.1:8010 / :8009)
* ``clef`` | ``clef-flash``  Cloudflare Workers AI (CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID,
                   or ./cloudflare_api_key.txt + ./cloudflare_account_id.txt)
* ``pplx``         Perplexity Decisions API, pplx-decider-v1-27b (PERPLEXITY_API_KEY or
                   ./perplexity_api_key.txt)
* ``systemone:<name>``  any /v1/systemone server at SOULSBENCH_<NAME>_BASE_URL (+ _MODEL, _API_KEY)
* append ``@base`` for the original untuned prompt, ``@r5`` for the previous strategy (e.g. ``jev@r5``)
* ``openai``       OpenAI Decisions API, gpt-6-luna (OPENAI_API_KEY or ./openai_api_key.txt)
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

from soulsbench.adapters.base import AdapterError, CombatAdapter, DecisionParseError
from soulsbench.adapters.baselines import IdleAdapter, RandomAdapter, ScriptedExpertAdapter
from soulsbench.adapters.http import HttpDecisionsAdapter
from soulsbench.adapters.jev import (
    JEV_MODEL,
    JevAdapter,
    SystemOneAdapter,
    load_jev_key,
    load_secret,
)
from soulsbench.adapters.openai_decisions import KEY_FILE as OPENAI_KEY_FILE
from soulsbench.adapters.openai_decisions import OpenAIDecisionsAdapter
from soulsbench.adapters.rules import RulesAdapter

__all__ = [
    "AdapterError",
    "CombatAdapter",
    "DecisionParseError",
    "HttpDecisionsAdapter",
    "IdleAdapter",
    "JevAdapter",
    "SystemOneAdapter",
    "OpenAIDecisionsAdapter",
    "RandomAdapter",
    "RulesAdapter",
    "ScriptedExpertAdapter",
    "make_adapter",
]

_BUILTIN = {
    "idle": IdleAdapter,
    "random": RandomAdapter,
    "expert": ScriptedExpertAdapter,
    "rules": RulesAdapter,
}
# Locally served System One models: name -> (default base URL, model id). Env overrides both.
LOCAL_SYSTEMONE = {
    "laya": ("http://127.0.0.1:8010/v1", "english"),
    "kev": ("http://127.0.0.1:8009/v1", JEV_MODEL),
}


CLOUDFLARE_RUN_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run"
PERPLEXITY_URL = "https://api.perplexity.ai/v1"


def _hosted(kind: str, prompt: str, env: Mapping[str, str]) -> SystemOneAdapter:
    """Hosted APIs that reuse Jev's state + typed-questions contract under another path."""
    label = kind if prompt == "tuned" else f"{kind}@{prompt}"
    if kind == "pplx":
        key = load_secret(
            env, ("PERPLEXITY_API_KEY", "SOULSBENCH_PPLX_API_KEY"), Path("perplexity_api_key.txt")
        )
        return SystemOneAdapter(
            label,
            PERPLEXITY_URL,
            "pplx-decider-v1-27b",
            api_key=key,
            prompt=prompt,
            needs_key=True,
            decide_path="/decisions",
        )
    # Workers AI: the model is in the path and the answer comes wrapped in {"result": ...}.
    key = load_secret(env, ("CLOUDFLARE_API_TOKEN",), Path("cloudflare_api_key.txt"))
    account = load_secret(env, ("CLOUDFLARE_ACCOUNT_ID",), Path("cloudflare_account_id.txt"))
    if key and not account:
        raise ValueError(f"{kind}: set CLOUDFLARE_ACCOUNT_ID (or ./cloudflare_account_id.txt)")
    return SystemOneAdapter(
        label,
        CLOUDFLARE_RUN_URL.format(account=account or "<account>"),
        kind,
        api_key=key,
        prompt=prompt,
        needs_key=True,
        decide_path=f"/@cf/cloudflare/{kind}",
        send_model=False,
        unwrap="result",
    )


def _env_key(name: str, suffix: str) -> str:
    return f"SOULSBENCH_{re.sub(r'[^A-Za-z0-9]', '_', name).upper()}_{suffix}"


def make_adapter(
    spec: str, *, http_base_url: str | None = None, env: Mapping[str, str] | None = None
) -> CombatAdapter:
    env = os.environ if env is None else env
    kind, _, name = spec.strip().partition(":")
    if kind == "rules" and name:
        return RulesAdapter(delay_ms=float(name))  # rules:<ms>
    if kind in _BUILTIN and not name:
        return _BUILTIN[kind]()
    if kind == "http":
        label = name or "http"
        url = http_base_url or env.get(_env_key(label, "BASE_URL"))
        if not url:
            raise ValueError(
                f"adapter {spec!r} needs a base URL: pass --http-base-url or set "
                f"{_env_key(label, 'BASE_URL')}"
            )
        key = env.get(_env_key(label, "API_KEY"))
        return HttpDecisionsAdapter(url, name=f"http:{name}" if name else "http", api_key=key)
    kind, _, prompt = kind.partition("@")  # "<adapter>@base" = the untuned prompt
    prompt = prompt or "tuned"
    if kind == "jev" and not name:
        return JevAdapter(load_jev_key(env), prompt=prompt)
    if kind in ("clef", "clef-flash", "pplx") and not name:
        return _hosted(kind, prompt, env)
    if (kind in LOCAL_SYSTEMONE and not name) or (kind == "systemone" and name):
        label = name or kind
        default_url, default_model = LOCAL_SYSTEMONE.get(label, (None, JEV_MODEL))
        url = http_base_url or env.get(_env_key(label, "BASE_URL")) or default_url
        if not url:
            raise ValueError(f"adapter {spec!r} needs {_env_key(label, 'BASE_URL')}")
        return SystemOneAdapter(
            label if prompt == "tuned" else f"{label}@{prompt}",
            url,
            env.get(_env_key(label, "MODEL"), default_model),
            api_key=env.get(_env_key(label, "API_KEY")),
            prompt=prompt,
        )
    if kind == "openai" and not name:
        key = load_secret(env, ("OPENAI_API_KEY", "SOULSBENCH_OPENAI_API_KEY"), OPENAI_KEY_FILE)
        return OpenAIDecisionsAdapter(key, prompt=prompt)
    raise ValueError(
        f"unknown adapter {spec!r}; use idle, random, expert, rules, jev, laya, kev, clef, clef-flash, pplx, systemone:<name>, http[:name] or openai (System One adapters take @base or @r5)"
    )
