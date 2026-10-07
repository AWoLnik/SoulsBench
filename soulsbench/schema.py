"""The shared SoulsBench contract.

Every model is benchmarked against exactly these types: one action space, one state, one decision
format, one SLA. Do not fork any of this per model. Adapters translate *to* this contract; they never
extend it.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "soulsbench/v1"
HISTORY_LEN = 8


class Action(IntEnum):
    """Discrete(16). Illegal or mid-animation actions are executed as IDLE."""

    IDLE = 0
    MOVE_FORWARD = 1
    MOVE_BACK = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    ROLL_FORWARD = 5
    ROLL_BACK = 6
    ROLL_LEFT = 7
    ROLL_RIGHT = 8
    LIGHT_ATTACK = 9
    HEAVY_ATTACK = 10
    BLOCK = 11
    PARRY = 12
    USE_ESTUS = 13
    BACKSTEP = 14
    TOGGLE_LOCK = 15

    @property
    def label(self) -> str:
        return self.name.lower()


N_ACTIONS = len(Action)
ACTION_LABELS: tuple[str, ...] = tuple(a.label for a in Action)


class SLA(BaseModel):
    """Timing contract for live DS3."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decide_budget_ms: float = Field(200.0, gt=0)
    control_period_ms: float = Field(
        250.0, gt=0, description="Paused mode only: game time each decision plays for."
    )
    realtime: bool = Field(
        True,
        description="True (default): the game runs continuously. Each answer is applied the "
        "moment it arrives and the next decision starts on the state read right after, so the "
        "model's latency is its reaction time. False: the game is paused while the model "
        "decides, then plays one control period (unrealistic reaction times; not comparable).",
    )
    min_action_interval_ms: float = Field(
        50.0,
        ge=0,
        description="Realtime only: actions are at least this far apart, so instant scripted "
        "baselines do not spam the game (3 frames at 60 fps). Slower models are not affected.",
    )


DEFAULT_SLA = SLA()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlayerState(_Strict):
    hp: int
    hp_max: int
    stamina: int
    stamina_max: int
    poise: float | None = Field(None, description="Current poise (super-armor durability).")
    poise_max: float | None = None
    estus: int | None = Field(None, description="Estus charges left; None if unreadable.")
    estus_max: int | None = None
    iframes: bool | None = Field(None, description="Invulnerability flag read from game memory.")
    parry_window: bool | None = Field(None, description="Parry-active flag read from game memory.")
    anim: str
    anim_id: int = Field(description="Stable SoulsGym animation id, -1 if unknown.")
    anim_time_s: float
    disabled: bool = Field(description="Locked in an animation: no move, roll, or attack possible.")
    locked_on: bool
    position: tuple[float, float, float]
    heading_rad: float


TelegraphKind = Literal["attack", "movement", "stagger", "unknown"]
TelegraphPhase = Literal[
    "neutral", "attack", "windup", "active", "recovery", "staggered", "unknown"
]


class Telegraph(_Strict):
    """What the boss is doing right now, and how far into it.

    ``phase`` is ``windup``/``active``/``recovery`` only when calibrated hit windows exist for the
    current attack (``timing_source == "calibrated"``). Without them an attack reports ``phase ==
    "attack"`` and ``hitbox_active is None`` rather than a guess.
    """

    anim: str
    anim_id: int
    kind: TelegraphKind
    progress: float | None = Field(None, ge=0.0, le=1.0)
    anim_time_s: float
    phase: TelegraphPhase
    hitbox_active: bool | None
    next_hit_in_s: float | None = Field(
        None,
        description="Seconds until the current attack's next calibrated hit window opens; "
        "0 while it is open, None when unknown or no hit is coming.",
    )
    timing_source: Literal["calibrated", "none", "n/a"]
    followup_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Observed chance that this attack chains straight into another attack "
        "(from logged runs); None when not an attack or never observed.",
    )
    distance: float = Field(description="Horizontal player-to-boss distance in meters.")
    angle_off_deg: float = Field(description="Unsigned angle between player facing and the boss.")


class TargetState(_Strict):
    name: str
    hp: int
    hp_max: int
    boss_phase: int
    position: tuple[float, float, float]
    heading_rad: float
    telegraph: Telegraph


class HistoryEntry(_Strict):
    tick: int
    t_s: float
    action: int = Field(description="Executed action id (after legality / fallback).")
    player_hp: int
    target_hp: int
    player_anim: str
    target_anim: str


class CombatState(_Strict):
    schema_version: Literal["soulsbench/v1"] = SCHEMA_VERSION
    episode_id: str
    scenario: str
    tick: int
    t_s: float = Field(description="Game time since episode start (paused time excluded).")
    player: PlayerState
    target: TargetState
    history: list[HistoryEntry] = Field(default_factory=list, max_length=HISTORY_LEN)
    legal_actions_mask: list[bool]

    @field_validator("legal_actions_mask")
    @classmethod
    def _check_mask(cls, mask: list[bool]) -> list[bool]:
        if len(mask) != N_ACTIONS:
            raise ValueError(f"legal_actions_mask must have {N_ACTIONS} entries, got {len(mask)}")
        if not mask[Action.IDLE]:
            raise ValueError("idle must always be legal")
        return mask

    def is_legal(self, discrete_id: int) -> bool:
        return 0 <= discrete_id < N_ACTIONS and self.legal_actions_mask[discrete_id]


class Usage(BaseModel):
    """Tokens one API call consumed, as the provider reported them."""

    model_config = ConfigDict(extra="ignore")

    input_tokens: int = 0
    output_tokens: int = 0


class Decision(BaseModel):
    """What an adapter returns. Extra keys from remote models are ignored, ``discrete_id`` is not."""

    model_config = ConfigDict(extra="ignore")

    discrete_id: int = Field(ge=0, lt=N_ACTIONS)
    usage: Usage | None = None  # hosted APIs: what this call cost, for the score card


# --- Bridge wire messages (WSL client <-> Windows bridge) -------------------------------------------

Outcome = Literal["running", "win", "death", "timeout"]


class ResetRequest(_Strict):
    episode_id: str
    scenario: str
    seed: int
    control_period_ms: float = Field(DEFAULT_SLA.control_period_ms, gt=0)
    realtime: bool = Field(
        False, description="Run the game continuously; drive it with /act instead of /step."
    )


class RecordStartRequest(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,119}$")  # a file name, nothing else
    label: str = Field("", max_length=200)  # on-screen text, if OBS has a "SoulsBench" source
    # one folder name under %USERPROFILE%\\Videos\\SoulsBench (no separators, no "..")
    subdir: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    password: str = ""  # obs-websocket password


class RecordStopRequest(_Strict):
    suffix: str = Field("done", pattern=r"^[A-Za-z0-9_-]{1,40}$")


class ResetResponse(_Strict):
    state: CombatState
    info: dict[str, Any] = Field(default_factory=dict)


class StepRequest(_Strict):
    discrete_id: int = Field(ge=0, lt=N_ACTIONS)


class StepInfo(_Strict):
    requested_action: int
    executed_action: int
    illegal: bool
    game_dt_s: float
    damage_dealt: int
    damage_taken: int
    healed: int
    step_wall_ms: float


class StepResponse(_Strict):
    state: CombatState
    terminated: bool
    truncated: bool
    outcome: Outcome
    info: StepInfo


class BridgeHealth(_Strict):
    ok: bool
    ready: bool = Field(description="Game attached, player in-game, env constructed.")
    bridge_version: str
    schema_version: str = SCHEMA_VERSION
    backend: str
    detail: dict[str, Any] = Field(default_factory=dict)


class BridgeError(_Strict):
    error: str
    message: str
