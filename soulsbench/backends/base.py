"""What the harness needs from a combat backend."""

from __future__ import annotations

from typing import Protocol

from soulsbench.schema import BridgeHealth, ResetRequest, ResetResponse, StepResponse


class EnvError(Exception):
    """The backend (bridge, game) failed mid-episode. The episode is recorded as aborted."""


class CombatEnv(Protocol):
    name: str

    def health(self) -> BridgeHealth: ...

    def reset(self, request: ResetRequest) -> ResetResponse: ...

    def step(self, discrete_id: int) -> StepResponse:
        """Paused mode: play one control period with this action, then pause."""
        ...

    def act(self, discrete_id: int) -> StepResponse:
        """Realtime mode: apply this action now and return the state read right after."""
        ...

    def close(self) -> None: ...
