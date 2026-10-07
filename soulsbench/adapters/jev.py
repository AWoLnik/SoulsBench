"""System One decision models (Jev and API-compatible ones) behind the CombatAdapter contract.

Jev (TypeSafe AI) defined the ``POST /v1/systemone`` API; Laya and Kev serve the same contract
locally, so one adapter drives all three. ``prompt="tuned"`` is the strategy prompt iterated on Jev
(paused 5/10 wins on gundyr_p1); ``prompt="base"`` is the original, untuned one. Benchmark other
models with both, so a prompt tuned on Jev is not mistaken for a better model.

API: ``POST https://api.typesafe.ai/v1/systemone`` with ``Authorization: Bearer <key>``, model
``jev-latest``, a ``state`` (object) and named typed ``questions`` (docs.typesafe.ai/api). We ask
exactly one ``choice`` question whose options are the currently legal action labels, so Jev can
only pick something the harness will execute. The answer label maps back to its ``discrete_id``.

Latency is the constraint: measured ~140-200 ms per call from this machine against a 200 ms decide
budget. So the state sent is a compact projection of the CombatState (same information a model
needs, fewer tokens), not the full JSON.

Key lookup: ``JEV_AI_API_KEY`` or ``SOULSBENCH_JEV_API_KEY``, else ``./jev_api_key.txt`` (gitignored).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from soulsbench.adapters.base import DecisionParseError
from soulsbench.adapters.http import HttpDecisionsAdapter
from soulsbench.schema import ACTION_LABELS, CombatState, Decision

JEV_BASE_URL = "https://api.typesafe.ai/v1"
JEV_MODEL = "jev-latest"
KEY_FILE = Path("jev_api_key.txt")

BASE_INSTRUCTIONS = (
    "You control a knight fighting Iudex Gundyr in Dark Souls III, one decision per 0.25 s. Pick the "
    "single best action now: avoid being hit during the boss's attacks (dodge or wait), punish "
    "openings with attacks when in range, and heal when HP is low and it is safe."
)
BASE_RUBRIC = {
    "idle": "Do nothing this tick; wait out an animation or a telegraph.",
    "move_forward": "Walk toward the boss (closes distance).",
    "move_back": "Walk away from the boss.",
    "move_left": "Strafe left around the boss.",
    "move_right": "Strafe right around the boss.",
    "roll_forward": "Dodge roll toward/through the boss; invulnerable briefly, costs stamina.",
    "roll_back": "Dodge roll away from the boss; invulnerable briefly, costs stamina.",
    "roll_left": "Dodge roll to the left; invulnerable briefly, costs stamina.",
    "roll_right": "Dodge roll to the right; invulnerable briefly, costs stamina.",
    "light_attack": "Fast sword swing; deals damage if the boss is within ~2.5 m.",
    "heavy_attack": "Slow strong swing; more damage, long recovery.",
    "block": "Raise the shield; reduces damage, drains stamina on hit.",
    "parry": "Shield parry; only safe against a well-timed boss swing.",
    "use_estus": "Drink estus to heal ~250 HP; long animation, vulnerable while drinking.",
    "backstep": "Quick hop backward; short invulnerability.",
    "toggle_lock": "Re-acquire lock-on to the boss.",
}

ACTION_RUBRIC = {
    "idle": "Wait.",
    "move_forward": "Walk toward the boss.",
    "move_back": "Walk away from the boss.",
    "move_left": "Strafe left.",
    "move_right": "Strafe right.",
    "roll_forward": "Roll toward the boss (brief invulnerability).",
    "roll_back": "Roll away (brief invulnerability).",
    "roll_left": "Roll left (brief invulnerability).",
    "roll_right": "Roll right (brief invulnerability).",
    "light_attack": "Quick sword hit, ~52 damage, needs distance <= 2.5 m.",
    "heavy_attack": "Slow strong hit, long recovery.",
    "block": "Shield up.",
    "parry": "Shield parry.",
    "use_estus": "Heal 250 HP; slow, vulnerable while drinking.",
    "backstep": "Hop back (brief invulnerability).",
    "toggle_lock": "Re-acquire lock-on.",
}


def instructions(lag_s: float = 0.0) -> str:
    """Pick the action whose condition matches. Every threshold is precomputed into a yes/no
    field (see :func:`tactical`), and each option's criterion states when to choose it: v4 with
    numeric thresholds had Jev walk forward on half the ticks it should attack and on a third
    of the ticks it should dodge."""
    lag_note = f" Your action lands {lag_s:g} s after this state." if lag_s else ""
    return (
        "Knight vs Iudex Gundyr (Dark Souls III), one action per 0.25 s; win = kill him without "
        "dying." + lag_note + " Read the 'tactical' fields and pick the FIRST option, in the order "
        "dodge, heal, retreat, attack, approach, wait, whose condition is true."
    )


TUNED_RUBRIC = {
    # Left, not forward: forward rolls end under or behind him, the lock-on forces a QuickTurn
    # and the next swing lands during it (~26% of damage taken in true realtime).
    "roll_left": "Dodge (check first): tactical.situation is hit_incoming and "
    "tactical.in_lunge_range is true.",
    "use_estus": "Heal: tactical.safe_to_drink is true and (tactical.low_hp is true, or "
    "tactical.situation is transforming and tactical.hurt is true).",
    "move_back": "Retreat: (tactical.low_hp is true and tactical.safe_to_drink is false), or "
    "tactical.situation is transforming and you are within 6 m.",
    "light_attack": "Attack: tactical.in_sword_range is true, tactical.can_attack is true, and "
    "(tactical.situation is opening, neutral or slow_windup, or tactical.boss_one_hit_left is "
    "true).",
    "move_forward": "Approach: you are not in sword range, and tactical.situation is neutral "
    "or opening or tactical.boss_one_hit_left is true.",
    "idle": "Wait: none of the conditions above is true (e.g. an attack is starting or his "
    "string continues).",
}
TUNED_OFFER = frozenset(TUNED_RUBRIC)


def instructions_r5(lag_s: float = 0.0) -> str:
    """Previous strategy prompt (5/10 paused, 1/10 realtime full fight). ``lag_s`` is how long after the state the action
    lands (realtime: one tick); the timing thresholds move earlier by that much."""
    roll, attack = 0.3 + lag_s, 1.0 + lag_s
    lag_note = (
        f"Your action takes effect {lag_s:g} s after this state, so act early. " if lag_s else ""
    )
    return (
        "Knight vs Iudex Gundyr (Dark Souls III), one action per 0.25 s; win = kill him. "
        + lag_note
        + f"Rules in order: 1) distance < 4.5 and (boss.next_hit_in_s <= {roll:g}, or boss.phase "
        "is 'attack', meaning an attack with unknown timing): roll left or right now. "
        "2) hp_percent < 40, estus_left > 0 and (distance > 5 or no hit within 1.5 s): "
        f"use_estus. 3) distance <= 2.5 and (next_hit_in_s is null or > {attack:g}): light_attack. "
        f"4) next_hit_in_s between {roll:g} and {attack:g} and distance < 4.5: stay still (idle) "
        "and wait to roll. 5) otherwise move_forward toward the boss. Never walk backward."
    )


INSTRUCTIONS = instructions()
# "<adapter>@<prompt>": tuned = current strategy, r5 = the previous one, base = untuned.
PROMPTS = {
    "tuned": instructions,
    "r5": instructions_r5,
    "base": lambda lag_s=0.0: BASE_INSTRUCTIONS,
}


REALTIME_OFFER = frozenset(
    {
        "idle",
        "move_forward",
        "move_left",
        "move_right",
        "roll_forward",
        "roll_back",
        "roll_left",
        "roll_right",
        "light_attack",
        "use_estus",
    }
)


def load_secret(env, names: tuple[str, ...], file: Path) -> str | None:
    """First of the env vars ``names`` that is set, else the contents of ``file`` (gitignored)."""
    for name in names:
        if env.get(name):
            return env[name].strip()
    return file.read_text().strip() if file.exists() else None


def load_jev_key(env: dict[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env
    return load_secret(env, ("JEV_AI_API_KEY", "SOULSBENCH_JEV_API_KEY"), KEY_FILE)


STRIKE_IDS = frozenset(ACTION_LABELS.index(a) for a in ("light_attack", "heavy_attack"))
TRANSFORM_ANIM = "Attack1500"  # Gundyr's phase-2 transformation (see windows_bridge.live_env)
LIGHT_ATTACK_DAMAGE = 52  # the starting Knight's straight sword, measured live
FOLLOWUP_RISKY = 0.35  # a recovery whose attack chains at least this often is not an opening


def situation(state: CombatState, lag_s: float = 0.0) -> str:
    """The boss telegraph reduced to one label, with the realtime lag folded in."""
    tel = state.target.telegraph
    if tel.anim == TRANSFORM_ANIM:
        return "transforming"  # ~6 s with a blast at the end; SoulsGym files it under "misc"
    if tel.phase == "staggered":
        return "opening"
    # Windows are calibrated from realtime hit times (padded ~0.08 s), so they open ~0.1 s before
    # the real hit. Decide when the hit is one answer latency (lag_s) + 0.1 s away: the roll goes
    # in just before the hit and its invulnerability covers it.
    if tel.phase == "attack" or (
        tel.next_hit_in_s is not None and tel.next_hit_in_s <= lag_s + 0.1
    ):
        return "hit_incoming"  # uncalibrated attacks count as danger
    if tel.phase == "windup" and tel.next_hit_in_s is not None and tel.next_hit_in_s > lag_s + 1.0:
        # A light attack lands ~0.4 s after input and can be rolled out of after ~0.8 s: room
        # for one swing before the hit (47 s of these in sword range went unused in 5 fights).
        return "slow_windup"
    if tel.phase in ("windup", "active"):
        return "attack_starting"
    if tel.phase == "recovery":
        risky = tel.followup_p is None or tel.followup_p >= FOLLOWUP_RISKY
        return "string_continues" if risky else "opening"
    return "neutral"


def seconds_since_my_attack(state: CombatState) -> float | None:
    """Time since the player's last executed attack (None if none in the history window).
    Realtime fallbacks turn requests made mid-swing into idles, so counting attacks in a row
    does not work; time since the last swing does."""
    if state.player.anim.startswith("Attack"):
        return 0.0
    for h in reversed(state.history):
        if h.action in STRIKE_IDS:
            return round(state.t_s - h.t_s, 2)
    return None


def compact_state(state: CombatState) -> dict[str, Any]:
    """The state the r5 and base prompts see."""
    p, t = state.player, state.target
    tel = t.telegraph
    out: dict[str, Any] = {
        "time_s": round(state.t_s, 2),
        "player": {
            "hp": f"{p.hp}/{p.hp_max}",
            "hp_percent": round(100 * p.hp / max(p.hp_max, 1)),
            "stamina": f"{p.stamina}/{p.stamina_max}",
            "estus_left": p.estus,
            "animation": p.anim,
            "locked_in_animation": p.disabled,
            "invulnerable": p.iframes,
        },
        "boss": {
            "hp": f"{t.hp}/{t.hp_max}",
            "animation": tel.anim,
            "kind": tel.kind,
            "phase": tel.phase,
            "animation_progress": None if tel.progress is None else round(tel.progress, 2),
            "hitbox_active": tel.hitbox_active,
            "next_hit_in_s": tel.next_hit_in_s,
            "distance_m": round(tel.distance, 2),
            "angle_off_deg": round(tel.angle_off_deg),
        },
        "recent": [
            {
                "action": ACTION_LABELS[h.action],
                "player_hp": h.player_hp,
                "boss_anim": h.target_anim,
            }
            for h in state.history[-3:]
        ],
    }
    return out


def tuned_state(state: CombatState, lag_s: float) -> dict[str, Any]:
    """The tuned prompt only reads the tactical fields; a small state keeps Jev under budget
    (a timeout on a dodge tick is a guaranteed hit)."""
    p = state.player
    return {
        "hp_percent": round(100 * p.hp / max(p.hp_max, 1)),
        "estus_left": p.estus,
        "boss_hp_percent": round(100 * state.target.hp / max(state.target.hp_max, 1)),
        "tactical": tactical(state, lag_s),
    }


def tactical(state: CombatState, lag_s: float) -> dict[str, Any]:
    """The tuned prompt's conditions, precomputed from the telegraph and player state."""
    p, d = state.player, state.target.telegraph.distance
    since = seconds_since_my_attack(state)
    sit = situation(state, lag_s)
    return {
        "situation": sit,
        "in_lunge_range": d < 7,  # his lunges close ~3 m: dodges at 5.4 m were needed
        "boss_close": d < 5,
        "in_sword_range": d <= 2.5,
        "boss_far": d >= 8,
        # one swing at a time in neutral; in a real opening a second swing fits (v9 survived
        # 84 s on average but dealt only ~6 hits per fight)
        "can_attack": since is None or since >= (0.6 if sit == "opening" else 1.0),
        # 45% died with full flasks: two of his hits take ~240 of 454 HP
        "low_hp": 100 * p.hp < 55 * max(p.hp_max, 1) and (p.estus is None or p.estus > 0),
        "hurt": 100 * p.hp < 90 * max(p.hp_max, 1) and (p.estus is None or p.estus > 0),
        # one light attack (~52) kills: worth trading for. Two fights sat 15 s at 4 HP because
        # phase-2 Gundyr is never neutral.
        "boss_one_hit_left": 0 < state.target.hp <= LIGHT_ATTACK_DAMAGE,
        # Phase-2 Gundyr is rarely neutral: 6 of 13 deaths still had estus after 10-25 s at low HP.
        "safe_to_drink": sit == "opening"
        or (sit == "transforming" and d >= 6)
        or (sit == "neutral" and d >= 6)
        or (sit == "slow_windup" and d >= 5)
        or (d >= 8 and sit != "hit_incoming"),
    }


class SystemOneAdapter(HttpDecisionsAdapter):
    decide_path = "/systemone"

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str = JEV_MODEL,
        api_key: str | None = None,
        prompt: str = "tuned",
        needs_key: bool = False,
        decide_path: str = "/systemone",
        send_model: bool = True,
        unwrap: str | None = None,
    ):
        if prompt not in PROMPTS:
            raise ValueError(f"prompt must be one of {sorted(PROMPTS)}, got {prompt!r}")
        super().__init__(base_url, name=name, api_key=api_key)
        self.model = model
        self.prompt = prompt
        self._needs_key, self._has_key = needs_key, bool(api_key)
        self.instructions = BASE_INSTRUCTIONS if prompt == "base" else PROMPTS[prompt]()
        self.rubric = BASE_RUBRIC if prompt == "base" else ACTION_RUBRIC
        self.offered: frozenset[str] | None = None  # None = offer every legal action
        if prompt == "tuned":
            self.rubric, self.offered = TUNED_RUBRIC, TUNED_OFFER
        self.lag_s = 0.0
        # Hosted clones of the API differ only in path, model field and an envelope key.
        self.decide_path, self._send_model, self._unwrap = decide_path, send_model, unwrap

    def set_action_lag(self, lag_s: float) -> None:
        """Realtime mode: actions land one tick after the state the model saw."""
        if self.prompt == "base":
            return  # the untuned baseline stays untuned
        self.instructions = PROMPTS[self.prompt](lag_s)
        self.lag_s = lag_s
        if self.prompt == "tuned":
            self.rubric, self.offered = TUNED_RUBRIC, TUNED_OFFER
        elif lag_s > 0:
            # Realtime runs showed Jev idling and parrying where its rules say otherwise; with a
            # tick of lag both are losing moves, so they are simply not offered.
            self.offered = REALTIME_OFFER

    def preflight(self) -> None:
        super().preflight()
        if self._needs_key and not self._has_key:
            raise ValueError(f"{self.name}: no API key")

    def reset(self, episode_id: str, seed: int | None = None) -> None:
        pass  # stateless API

    def decide(self, state: CombatState, deadline_ms: float) -> Decision:
        legal = [i for i, ok in enumerate(state.legal_actions_mask) if ok]
        # Locked in an animation: nothing to ask, unless realtime, where the answer lands a tick
        # later and the lock may be over by then.
        if len(legal) == 1 and self.lag_s == 0:
            return Decision(discrete_id=legal[0])
        return super().decide(state, deadline_ms)

    def encode_request(self, state: CombatState, deadline_ms: float) -> dict[str, Any]:
        legal = [ACTION_LABELS[i] for i, ok in enumerate(state.legal_actions_mask) if ok]
        if self.offered is not None and self.lag_s > 0:
            # Realtime: the mask describes a state one tick old. Offer the full set; the harness
            # re-checks legality against the state the action actually lands in.
            legal = [a for a in ACTION_LABELS if a in self.offered]
        elif self.offered is not None:
            legal = [a for a in legal if a in self.offered] or ["idle"]
        body: dict[str, Any] = {"model": self.model} if self._send_model else {}
        return body | {
            "state": tuned_state(state, self.lag_s)
            if self.prompt == "tuned"
            else compact_state(state),
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": self.instructions,
                    "criteria": {label: self.rubric[label] for label in legal},
                }
            },
        }

    def decode_response(self, payload: Any) -> Decision:
        try:
            if self._unwrap:
                payload = payload[self._unwrap]
            label = payload["answers"]["action"]["choice"]
            return Decision(discrete_id=ACTION_LABELS.index(label), usage=payload.get("usage"))
        except (KeyError, TypeError, ValueError):
            raise DecisionParseError(
                f"unexpected {self.name} response: {str(payload)[:200]}"
            ) from None


class JevAdapter(SystemOneAdapter):
    """Jev on TypeSafe's hosted API."""

    def __init__(self, api_key: str | None, prompt: str = "tuned"):
        super().__init__(
            "jev" if prompt == "tuned" else f"jev@{prompt}",
            JEV_BASE_URL,
            JEV_MODEL,
            api_key=api_key,
            prompt=prompt,
            needs_key=True,
        )

    def preflight(self) -> None:
        if not self._has_key:
            raise ValueError("jev: no API key (set JEV_AI_API_KEY or put it in ./jev_api_key.txt)")
