"""RulesAdapter: the tuned System One prompt's rules, executed locally with no model.

The tuned Jev prompt reads precomputed ``tactical`` fields and picks the first option whose
condition holds. This adapter applies exactly those conditions in the same order, instantly and
without timeouts. It is the control for that prompt: if it wins as often as a model does, the
model's result is the rules', and the gap between the two is what the model's latency and
mistakes cost.
"""

from __future__ import annotations

import time

from soulsbench.adapters.base import CombatAdapter
from soulsbench.adapters.jev import tactical
from soulsbench.schema import Action, CombatState, Decision


class RulesAdapter(CombatAdapter):
    """``delay_ms`` holds each answer back that long: the rules' timings assume a model's
    latency (an instant bot rolls ~0.2 s early), so ``rules:150`` is Jev's rules at Jev's typical
    latency with no timeouts or mistakes."""

    def __init__(self, delay_ms: float = 0.0) -> None:
        self.delay_s = delay_ms / 1000.0
        self.name = f"rules:{delay_ms:g}" if delay_ms else "rules"
        self.lag_s = 0.0

    def set_action_lag(self, lag_s: float) -> None:
        self.lag_s = lag_s

    def decide(self, state: CombatState, deadline_ms: float) -> Decision:
        action = choose(state, self.lag_s)
        if self.delay_s:
            time.sleep(self.delay_s)
        return Decision(discrete_id=action)


def choose(state: CombatState, lag_s: float) -> Action:
    """Mirror of ``TUNED_RUBRIC`` in :mod:`soulsbench.adapters.jev` (keep the two in sync)."""
    x = tactical(state, lag_s)
    sit, d = x["situation"], state.target.telegraph.distance
    if sit == "hit_incoming" and x["in_lunge_range"]:
        return Action.ROLL_LEFT
    if x["safe_to_drink"] and (x["low_hp"] or (sit == "transforming" and x["hurt"])):
        return Action.USE_ESTUS
    if (x["low_hp"] and not x["safe_to_drink"]) or (sit == "transforming" and d < 6):
        return Action.MOVE_BACK
    if (
        x["in_sword_range"]
        and x["can_attack"]
        and (sit in ("opening", "neutral", "slow_windup") or x["boss_one_hit_left"])
    ):
        return Action.LIGHT_ATTACK
    if not x["in_sword_range"] and (sit in ("neutral", "opening") or x["boss_one_hit_left"]):
        return Action.MOVE_FORWARD
    return Action.IDLE
