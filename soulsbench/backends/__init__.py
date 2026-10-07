"""Combat backends. ``souls_gym`` is live Dark Souls III through the Windows bridge."""

from __future__ import annotations

from soulsbench.backends.base import CombatEnv, EnvError

BACKENDS = ("souls_gym",)


def make_backend(name: str, *, bridge_url: str | None = None) -> CombatEnv:
    if name == "souls_gym":
        from soulsbench.backends.souls_gym import SoulsGymBridgeEnv

        return SoulsGymBridgeEnv(bridge_url)
    raise ValueError(f"unknown backend {name!r}; available: {', '.join(BACKENDS)}")


__all__ = ["BACKENDS", "CombatEnv", "EnvError", "make_backend"]
