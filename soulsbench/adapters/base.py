"""The CombatAdapter contract every model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from soulsbench.schema import CombatState, Decision


class DecisionParseError(Exception):
    """The model answered, but the answer is not a valid Decision. The harness plays idle."""


class AdapterError(Exception):
    """The model could not be reached or failed. The harness plays idle."""


class CombatAdapter(ABC):
    """One model behind one interface.

    The harness calls :meth:`reset` once per episode and :meth:`decide` once per control tick. A
    decision that is late, malformed, raises, or names an illegal action is executed as idle.
    """

    name: str = "adapter"

    def preflight(self) -> None:
        """Raise if the adapter cannot run at all (missing URL, unimplemented mapping...)."""

    def reset(self, episode_id: str, seed: int | None = None) -> None:
        """Start a new episode. ``seed`` is the episode seed, for adapters that use randomness."""

    @abstractmethod
    def decide(self, state: CombatState, deadline_ms: float) -> Decision:
        """Pick one of the 16 actions for ``state`` within ``deadline_ms`` of wall-clock time."""

    def close(self) -> None:
        """Release resources (HTTP sessions...)."""
