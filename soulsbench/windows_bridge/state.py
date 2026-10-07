"""Turn one raw game snapshot into a CombatState. Pure Python: no game, no soulsgym import.

Legality follows SoulsGym's own rule. Each player animation has three experimentally measured
times (attack, roll, move) after which that kind of input is accepted. Before that the game would
*queue* the input and fire it later, so those actions are illegal here (and executed as idle).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from soulsbench.scenarios import Scenario
from soulsbench.schema import (
    N_ACTIONS,
    Action,
    CombatState,
    HistoryEntry,
    PlayerState,
    TargetState,
    Telegraph,
)

# Unknown player animations (e.g. drinking estus) are treated as fully locked for their memory-read
# length when that length looks like a one-shot animation; longer ones are assumed to be loops.
UNKNOWN_ANIM_MAX_LOCK_S = 5.0
_PLAYER_GRAB_ANIMS = ("ThrowAtk", "ThrowDef")

MOVES = (Action.MOVE_FORWARD, Action.MOVE_BACK, Action.MOVE_LEFT, Action.MOVE_RIGHT)
ROLLS = (
    Action.ROLL_FORWARD,
    Action.ROLL_BACK,
    Action.ROLL_LEFT,
    Action.ROLL_RIGHT,
    Action.BACKSTEP,
)
STRIKES = (Action.LIGHT_ATTACK, Action.HEAVY_ATTACK, Action.PARRY)


@dataclass(frozen=True)
class AttackWindows:
    """Seconds into an attack animation during which its hitbox was observed to be live."""

    windows: tuple[tuple[float, float], ...]
    samples: int = 0


@dataclass(frozen=True)
class AnimTables:
    # name -> (stable id, (attack_ready_s, roll_ready_s, move_ready_s))
    player: Mapping[str, tuple[int, tuple[float, float, float]]]
    # name -> (stable id, SoulsGym category: "attacks" | "movement" | "misc")
    boss: Mapping[str, tuple[int, str]]
    attack_windows: Mapping[str, AttackWindows] = field(default_factory=dict)
    # attack name -> observed chance it chains straight into another attack
    followups: Mapping[str, float] = field(default_factory=dict)


@dataclass
class RawSnapshot:
    """Everything read from the game for one tick. Optional fields are None when unreadable."""

    player_hp: int
    player_hp_max: int
    player_sp: int
    player_sp_max: int
    player_anim: str
    player_anim_time_s: float  # SoulsGym-tracked duration of the current animation
    player_pos: tuple[float, float, float]
    player_heading: float
    locked_on: bool
    boss_hp: int
    boss_hp_max: int
    boss_phase: int
    boss_anim: str  # SoulsGym's name (attack ids recovered through hit-reaction overlays)
    boss_anim_time_s: float  # SoulsGym-tracked duration
    boss_pos: tuple[float, float, float]
    boss_heading: float
    player_anim_mem_time_s: float | None = None
    player_anim_max_s: float | None = None
    player_poise: float | None = None
    player_poise_max: float | None = None
    player_iframes: bool | None = None
    player_parry_window: bool | None = None
    estus: int | None = None
    estus_max: int | None = None
    boss_anim_raw: str | None = None
    boss_anim_mem_time_s: float | None = None
    boss_anim_max_s: float | None = None


def _lock_times(snap: RawSnapshot, tables: AnimTables) -> tuple[tuple[float, float, float], float]:
    """(attack, roll, move) ready times and the clock to compare them against."""
    known = tables.player.get(snap.player_anim)
    if known is not None:
        return known[1], snap.player_anim_time_s
    length = snap.player_anim_max_s
    if length is not None and 0.0 < length <= UNKNOWN_ANIM_MAX_LOCK_S:
        clock = snap.player_anim_mem_time_s
        return (length, length, length), clock if clock is not None else snap.player_anim_time_s
    return (0.0, 0.0, 0.0), snap.player_anim_time_s


def legal_mask(snap: RawSnapshot, tables: AnimTables) -> tuple[list[bool], bool]:
    """Return (legal_actions_mask, disabled)."""
    (attack_t, roll_t, move_t), t = _lock_times(snap, tables)
    has_sp = snap.player_sp > 0
    move_ok = t >= move_t
    roll_ok = has_sp and t >= roll_t
    strike_ok = has_sp and t >= attack_t
    item_ok = t >= attack_t
    mask = [False] * N_ACTIONS
    mask[Action.IDLE] = True
    for a in MOVES:
        mask[a] = move_ok
    for a in ROLLS:
        mask[a] = roll_ok
    for a in STRIKES:
        mask[a] = strike_ok
    mask[Action.BLOCK] = move_ok
    mask[Action.USE_ESTUS] = item_ok and (snap.estus is None or snap.estus > 0)
    # SoulsGym keeps the camera locked on the boss (there is no camera action), so the only useful
    # lock press is re-acquiring a lost lock. Releasing it would just be undone next tick.
    mask[Action.TOGGLE_LOCK] = not snap.locked_on and snap.player_anim not in _PLAYER_GRAB_ANIMS
    disabled = t < min(attack_t, roll_t, move_t)
    return mask, disabled


def _phase_from_windows(t: float, windows: Iterable[tuple[float, float]]) -> tuple[str, bool]:
    ordered = sorted(windows)
    if any(lo <= t <= hi for lo, hi in ordered):
        return "active", True
    if t > ordered[-1][1]:
        return "recovery", False
    return "windup", False


def _next_hit_in(t: float, windows: Iterable[tuple[float, float]]) -> float | None:
    for lo, hi in sorted(windows):
        if lo <= t <= hi:
            return 0.0
        if t < lo:
            return round(lo - t, 3)
    return None


def _forward(heading: float) -> tuple[float, float]:
    # SoulsGym poses are [x, y, z, a] with z up. Both start poses in its coordinates.yaml (player
    # facing Gundyr, Gundyr facing the player) agree with forward = (-sin a, -cos a).
    return -math.sin(heading), -math.cos(heading)


def telegraph(snap: RawSnapshot, tables: AnimTables) -> Telegraph:
    entry = tables.boss.get(snap.boss_anim)
    category = entry[1] if entry else None
    if snap.boss_anim == "ThrowDef" or category == "misc":
        kind = "stagger"
    elif category == "attacks":
        kind = "attack"
    elif category == "movement":
        kind = "movement"
    else:
        kind = "unknown"

    t = snap.boss_anim_time_s
    next_hit = None
    if kind == "attack":
        # Phase 2 reuses phase-1 animation names with different timings: prefer "p2:<anim>".
        windows = None
        if snap.boss_phase == 2:
            windows = tables.attack_windows.get(f"p2:{snap.boss_anim}")
        windows = windows or tables.attack_windows.get(snap.boss_anim)
        if windows and windows.windows:
            phase, hitbox, source = (*_phase_from_windows(t, windows.windows), "calibrated")
            next_hit = _next_hit_in(t, windows.windows)
        else:
            phase, hitbox, source = "attack", None, "none"
    elif kind == "movement":
        phase, hitbox, source = "neutral", False, "n/a"
    elif kind == "stagger":
        phase, hitbox, source = "staggered", False, "n/a"
    else:
        phase, hitbox, source = "unknown", None, "n/a"

    progress = None
    if (
        snap.boss_anim_raw == snap.boss_anim
        and snap.boss_anim_max_s
        and snap.boss_anim_max_s > 0
        and snap.boss_anim_mem_time_s is not None
    ):
        progress = min(max(snap.boss_anim_mem_time_s / snap.boss_anim_max_s, 0.0), 1.0)

    dx = snap.boss_pos[0] - snap.player_pos[0]
    dy = snap.boss_pos[1] - snap.player_pos[1]
    distance = math.hypot(dx, dy)
    fx, fy = _forward(snap.player_heading)
    if distance > 1e-6:
        cos_off = max(-1.0, min(1.0, (fx * dx + fy * dy) / distance))
        angle_off = math.degrees(math.acos(cos_off))
    else:
        angle_off = 0.0

    return Telegraph(
        anim=snap.boss_anim,
        anim_id=entry[0] if entry else -1,
        kind=kind,
        progress=progress,
        anim_time_s=t,
        phase=phase,
        hitbox_active=hitbox,
        next_hit_in_s=next_hit,
        timing_source=source,
        followup_p=tables.followups.get(snap.boss_anim) if kind == "attack" else None,
        distance=distance,
        angle_off_deg=angle_off,
    )


def build_state(
    snap: RawSnapshot,
    tables: AnimTables,
    *,
    episode_id: str,
    scenario: Scenario,
    tick: int,
    t_s: float,
    history: Iterable[HistoryEntry],
) -> CombatState:
    mask, disabled = legal_mask(snap, tables)
    player_entry = tables.player.get(snap.player_anim)
    player = PlayerState(
        hp=snap.player_hp,
        hp_max=snap.player_hp_max,
        stamina=snap.player_sp,
        stamina_max=snap.player_sp_max,
        poise=snap.player_poise,
        poise_max=snap.player_poise_max,
        estus=snap.estus,
        estus_max=snap.estus_max,
        iframes=snap.player_iframes,
        parry_window=snap.player_parry_window,
        anim=snap.player_anim,
        anim_id=player_entry[0] if player_entry else -1,
        anim_time_s=snap.player_anim_time_s,
        disabled=disabled,
        locked_on=snap.locked_on,
        position=snap.player_pos,
        heading_rad=snap.player_heading,
    )
    target = TargetState(
        name=scenario.target_name,
        hp=snap.boss_hp,
        hp_max=snap.boss_hp_max,
        boss_phase=snap.boss_phase,
        position=snap.boss_pos,
        heading_rad=snap.boss_heading,
        telegraph=telegraph(snap, tables),
    )
    return CombatState(
        episode_id=episode_id,
        scenario=scenario.name,
        tick=tick,
        t_s=t_s,
        player=player,
        target=target,
        history=list(history),
        legal_actions_mask=mask,
    )


# --- attack-window data file ----------------------------------------------------------------------

ATTACK_WINDOWS_SCHEMA = "soulsbench/attack_windows/v1"
DEFAULT_ATTACK_WINDOWS = Path(__file__).parent / "data" / "attack_windows.json"


def load_attack_windows(boss: str, path: Path = DEFAULT_ATTACK_WINDOWS) -> dict[str, AttackWindows]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get("schema") != ATTACK_WINDOWS_SCHEMA:
        raise ValueError(f"{path}: expected schema {ATTACK_WINDOWS_SCHEMA}")
    return {
        anim: AttackWindows(
            tuple((float(a), float(b)) for a, b in w["windows"]), int(w.get("samples", 0))
        )
        for anim, w in data.get("bosses", {}).get(boss, {}).items()
    }


def load_followups(boss: str, path: Path = DEFAULT_ATTACK_WINDOWS) -> dict[str, float]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {a: float(p) for a, p in data.get("followups", {}).get(boss, {}).items()}


# --- estus -------------------------------------------------------------------------------------

DRINK_ANIMS = frozenset({"ItemDrinkStart", "ItemDrinking"})


@dataclass
class EstusTracker:
    """Estus charges left in the episode: memory at the start, minus the drinks the bridge saw.

    DS3's inventory quantity for the Estus Flask drops on every drink except the last one: with one
    charge left it stays at 1 while the HUD shows an empty flask (observed live on 1.15.2). So each
    drinking animation counts once, when either its heal lands or the memory quantity drops, and the
    reported count is ``min(start - drinks, memory)``.
    """

    start: int | None
    cap: int | None
    used: int = 0
    _last_mem: int | None = None
    _in_drink: bool = False
    _counted: bool = False
    _grace: int = 0

    @classmethod
    def begin(cls, mem_count: int | None, cap: int | None) -> EstusTracker:
        return cls(start=mem_count, cap=cap, _last_mem=mem_count)

    def update(self, player_anim: str, healed: int, mem_count: int | None) -> None:
        mem_drop = (
            mem_count is not None and self._last_mem is not None and mem_count < self._last_mem
        )
        if mem_count is not None:
            self._last_mem = mem_count
        drinking = player_anim in DRINK_ANIMS
        if drinking and not self._in_drink:
            self._in_drink, self._counted, self._grace = True, False, 1
        if self._in_drink and not self._counted and (healed > 0 or mem_drop):
            self.used += 1
            self._counted = True
        if self._in_drink and not drinking:  # drink over; give a late heal one more tick
            if self._counted or self._grace <= 0:
                self._in_drink = False
            else:
                self._grace -= 1

    def remaining(self, mem_count: int | None) -> int | None:
        if self.start is None:
            return mem_count
        left = self.start - self.used
        if mem_count is not None:
            left = min(left, mem_count)
        return max(left, 0)
