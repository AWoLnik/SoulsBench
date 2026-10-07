"""Reference adapters: Idle, Random, ScriptedExpert."""

from __future__ import annotations

import random

from soulsbench.adapters.base import CombatAdapter
from soulsbench.schema import Action, CombatState, Decision


class IdleAdapter(CombatAdapter):
    """Never acts. Lower bound for survival and damage."""

    name = "idle"

    def decide(self, state: CombatState, deadline_ms: float) -> Decision:
        return Decision(discrete_id=Action.IDLE)


class RandomAdapter(CombatAdapter):
    """Uniform over the currently legal actions, seeded per episode."""

    name = "random"

    def __init__(self) -> None:
        self._rng = random.Random(0)

    def reset(self, episode_id: str, seed: int | None = None) -> None:
        self._rng = random.Random(seed if seed is not None else episode_id)

    def decide(self, state: CombatState, deadline_ms: float) -> Decision:
        legal = [i for i, ok in enumerate(state.legal_actions_mask) if ok]
        return Decision(discrete_id=self._rng.choice(legal))


class ScriptedExpertAdapter(CombatAdapter):
    """Heal under 40% HP, else close the distance and light attack.

    Deliberately simple: no dodging, no reading of telegraphs. It is the "does the plumbing work and
    does damage register" baseline, not a strong player.
    """

    name = "expert"
    heal_below_frac = 0.40
    attack_range_m = 2.5

    def decide(self, state: CombatState, deadline_ms: float) -> Decision:
        player = state.player
        wants_heal = player.hp < self.heal_below_frac * player.hp_max
        has_estus = player.estus is None or player.estus > 0
        if wants_heal and has_estus:
            return self._if_legal(state, Action.USE_ESTUS)
        if state.target.telegraph.distance > self.attack_range_m:
            return self._if_legal(state, Action.MOVE_FORWARD)
        return self._if_legal(state, Action.LIGHT_ATTACK)

    @staticmethod
    def _if_legal(state: CombatState, action: Action) -> Decision:
        return Decision(discrete_id=action if state.is_legal(action) else Action.IDLE)
