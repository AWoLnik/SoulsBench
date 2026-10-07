"""Benchmark scenarios. Both the WSL harness and the Windows bridge read this registry."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    boss: Literal["iudex"]
    target_name: str
    boss_phase: int
    boss_hp_pool: int
    max_time_s: float
    random_start_pose: bool
    pin_boss_hp: bool = True  # refill the boss every tick (phase lock); False = real HP, full fight
    description: str


_GUNDYR_P1_TEXT = (
    "Iudex Gundyr, phase 1, SoulsGym arena reset. Gundyr's in-game HP is refilled every tick so he "
    "never transforms; damage is tracked against his 1037 HP pool (SoulsGym convention). Win = 1037 "
    "damage dealt. The player's HP is real game HP (estus heals). Knight, SL9 base stats, starting "
    "gear."
)

SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario(
            name="gundyr_p1",
            boss="iudex",
            target_name="Iudex Gundyr",
            boss_phase=1,
            boss_hp_pool=1037,
            max_time_s=180.0,
            random_start_pose=False,
            description=_GUNDYR_P1_TEXT + " Fixed SoulsGym start pose.",
        ),
        Scenario(
            name="gundyr_p1_randstart",
            boss="iudex",
            target_name="Iudex Gundyr",
            boss_phase=1,
            boss_hp_pool=1037,
            max_time_s=180.0,
            random_start_pose=True,
            description=_GUNDYR_P1_TEXT + " Start pose drawn from SoulsGym's 11 poses by seed.",
        ),
        Scenario(
            name="gundyr_full",
            boss="iudex",
            target_name="Iudex Gundyr",
            boss_phase=1,
            boss_hp_pool=1037,
            max_time_s=300.0,
            random_start_pose=False,
            pin_boss_hp=False,
            description=(
                "Iudex Gundyr, the whole fight: real boss HP, he transforms into phase 2 at about "
                "half HP, and a win means he dies on screen. The player's HP is real (estus heals). "
                "Episodes after a transform or a kill start with a game reload (~20-30 s). Knight, "
                "SL9 base stats, starting gear, fixed SoulsGym start pose."
            ),
        ),
    )
}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(f"Unknown scenario {name!r}. Known: {sorted(SCENARIOS)}") from None
