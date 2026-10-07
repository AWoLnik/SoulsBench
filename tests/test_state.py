import json
import math

import pytest

from soulsbench.scenarios import get_scenario
from soulsbench.schema import Action
from soulsbench.windows_bridge.state import (
    ATTACK_WINDOWS_SCHEMA,
    MOVES,
    ROLLS,
    STRIKES,
    AnimTables,
    AttackWindows,
    EstusTracker,
    build_state,
    legal_mask,
    load_attack_windows,
    telegraph,
)
from tests.fakes import TABLES, snapshot


def legal(mask) -> set[Action]:
    return {Action(i) for i, ok in enumerate(mask) if ok}


def test_neutral_player_can_do_everything_but_relock():
    mask, disabled = legal_mask(snapshot(), TABLES)
    assert legal(mask) == set(Action) - {Action.TOGGLE_LOCK}
    assert not disabled


def test_mid_roll_is_disabled():
    mask, disabled = legal_mask(
        snapshot(player_anim="RollingMedium", player_anim_time_s=0.3), TABLES
    )
    assert legal(mask) == {Action.IDLE}
    assert disabled


def test_attack_recovery_allows_combo_before_movement():
    # AttackRightLight1 timings: attack 0.63, roll 0.80, move 1.03
    mask, disabled = legal_mask(
        snapshot(player_anim="AttackRightLight1", player_anim_time_s=0.7), TABLES
    )
    assert legal(mask) == {Action.IDLE, Action.USE_ESTUS, *STRIKES}
    assert not disabled
    mask, _ = legal_mask(snapshot(player_anim="AttackRightLight1", player_anim_time_s=0.9), TABLES)
    assert set(ROLLS) <= legal(mask) and not set(MOVES) & legal(mask)


def test_no_stamina_blocks_rolls_and_strikes_only():
    mask, _ = legal_mask(snapshot(player_sp=0), TABLES)
    assert not (set(ROLLS) | set(STRIKES)) & legal(mask)
    assert {Action.BLOCK, Action.USE_ESTUS, *MOVES} <= legal(mask)


def test_estus_gating():
    assert not legal_mask(snapshot(estus=0), TABLES)[0][Action.USE_ESTUS]
    assert legal_mask(snapshot(estus=None), TABLES)[0][Action.USE_ESTUS]  # unreadable: game decides


def test_toggle_lock_only_reacquires():
    assert not legal_mask(snapshot(locked_on=True), TABLES)[0][Action.TOGGLE_LOCK]
    assert legal_mask(snapshot(locked_on=False), TABLES)[0][Action.TOGGLE_LOCK]


def test_unknown_animation_locks_for_its_memory_length():
    drinking = dict(player_anim="ItemDrinkSomething", player_anim_max_s=2.0)
    mask, disabled = legal_mask(snapshot(**drinking, player_anim_mem_time_s=0.5), TABLES)
    assert legal(mask) == {Action.IDLE} and disabled
    mask, disabled = legal_mask(snapshot(**drinking, player_anim_mem_time_s=2.05), TABLES)
    assert Action.LIGHT_ATTACK in legal(mask) and not disabled
    looping = snapshot(player_anim="GuardOn", player_anim_max_s=30.0, player_anim_mem_time_s=0.1)
    assert not legal_mask(looping, TABLES)[1]
    assert not legal_mask(snapshot(player_anim="Mystery", player_anim_max_s=None), TABLES)[1]


def test_telegraph_without_calibration_does_not_guess():
    tel = telegraph(snapshot(boss_anim="Attack3000", boss_anim_time_s=0.4), TABLES)
    assert (tel.kind, tel.phase, tel.hitbox_active, tel.timing_source) == (
        "attack",
        "attack",
        None,
        "none",
    )
    assert tel.anim_id == 0


@pytest.mark.parametrize(
    ("t", "phase", "hit"),
    [
        (0.2, "windup", False),
        (0.9, "active", True),
        (1.25, "windup", False),
        (1.6, "active", True),
        (2.5, "recovery", False),
    ],
)
def test_telegraph_with_calibrated_windows(t, phase, hit):
    tables = AnimTables(
        player=TABLES.player,
        boss=TABLES.boss,
        attack_windows={"Attack3000": AttackWindows(((0.8, 1.1), (1.5, 1.8)), samples=7)},
    )
    tel = telegraph(snapshot(boss_anim="Attack3000", boss_anim_time_s=t), tables)
    assert (tel.phase, tel.hitbox_active, tel.timing_source) == (phase, hit, "calibrated")


def test_telegraph_non_attacks():
    walk = telegraph(snapshot(boss_anim="WalkFrontBattle"), TABLES)
    assert (walk.kind, walk.phase, walk.hitbox_active) == ("movement", "neutral", False)
    broken = telegraph(snapshot(boss_anim="SABreak"), TABLES)
    assert (broken.kind, broken.phase, broken.hitbox_active) == ("stagger", "staggered", False)
    mystery = telegraph(snapshot(boss_anim="Attack9999"), TABLES)
    assert (mystery.kind, mystery.anim_id, mystery.hitbox_active) == ("unknown", -1, None)


def test_distance_and_angle():
    ahead = telegraph(snapshot(boss_pos=(0.0, 5.0, 0.0)), TABLES)  # player at origin facing +y
    assert ahead.distance == pytest.approx(5.0)
    assert ahead.angle_off_deg == pytest.approx(0.0, abs=1e-4)
    side = telegraph(snapshot(boss_pos=(3.0, 0.0, 1.0)), TABLES)
    assert side.distance == pytest.approx(3.0)  # horizontal only
    assert side.angle_off_deg == pytest.approx(90.0, abs=1e-4)
    behind = telegraph(snapshot(boss_pos=(0.0, -2.0, 0.0)), TABLES)
    assert behind.angle_off_deg == pytest.approx(180.0, abs=1e-4)


def test_progress_only_when_memory_matches_the_animation():
    s = dict(boss_anim="Attack3000", boss_anim_mem_time_s=0.5, boss_anim_max_s=2.0)
    assert telegraph(snapshot(**s, boss_anim_raw="Attack3000"), TABLES).progress == pytest.approx(
        0.25
    )
    assert telegraph(snapshot(**s, boss_anim_raw="SABlend_Add_Front"), TABLES).progress is None
    assert telegraph(snapshot(**s, boss_anim_raw=None), TABLES).progress is None


def test_build_state_fields():
    state = build_state(
        snapshot(player_iframes=True, player_poise=12.5, player_poise_max=40.0),
        TABLES,
        episode_id="ep",
        scenario=get_scenario("gundyr_p1"),
        tick=3,
        t_s=0.75,
        history=(),
    )
    assert state.player.iframes is True
    assert state.player.anim_id == 0
    assert state.target.name == "Iudex Gundyr"
    assert state.target.boss_phase == 1
    assert math.isclose(state.target.telegraph.distance, 5.0)


def test_attack_windows_file(tmp_path):
    bundled = load_attack_windows("iudex")  # calibrated from the first live runs
    assert all(w.windows and w.samples >= 3 for w in bundled.values())
    path = tmp_path / "w.json"
    path.write_text(
        json.dumps(
            {
                "schema": ATTACK_WINDOWS_SCHEMA,
                "bosses": {"iudex": {"Attack3000": {"windows": [[0.8, 1.1]], "samples": 4}}},
            }
        )
    )
    assert load_attack_windows("iudex", path) == {"Attack3000": AttackWindows(((0.8, 1.1),), 4)}
    path.write_text(json.dumps({"schema": "nope"}))
    with pytest.raises(ValueError):
        load_attack_windows("iudex", path)


def _replay(tracker, ticks):
    out = []
    for anim, healed, mem in ticks:
        tracker.update(anim, healed, mem)
        out.append(tracker.remaining(mem))
    return out


def test_estus_tracker_replays_the_live_episode():
    """Ticks 73-77, 109-113, 139-143 of the first live run: memory stuck at 1 after the 3rd drink."""
    t = EstusTracker.begin(3, 3)
    drink = lambda mem_after: [  # noqa: E731
        ("ItemDrinkStart", 0, mem_after + 1),
        ("ItemDrinkStart", 0, mem_after + 1),
        ("ItemDrinking", 0, mem_after + 1),
        ("ItemDrinking", 250, mem_after),
        ("DamageMiddle", 0, mem_after),
    ]
    assert _replay(t, drink(2))[-1] == 2
    assert _replay(t, drink(1))[-1] == 1
    last = [
        ("ItemDrinkStart", 0, 1),
        ("ItemDrinking", 0, 1),
        ("ItemDrinking", 250, 1),
        ("Idle", 0, 1),
    ]
    assert _replay(t, last)[-1] == 0  # memory still says 1; the HUD shows an empty flask
    assert t.used == 3


def test_estus_tracker_edge_cases():
    late = EstusTracker.begin(3, 3)  # heal registers on the tick the animation ends
    assert _replay(late, [("ItemDrinking", 0, 3), ("DamageSmall", 250, 3), ("Idle", 0, 3)])[-1] == 2
    cut = EstusTracker.begin(3, 3)  # hit before the sip: nothing consumed
    assert _replay(cut, [("ItemDrinkStart", 0, 3), ("DamageLarge", 0, 3), ("Idle", 0, 3)])[-1] == 3
    full_hp = EstusTracker.begin(3, 3)  # drinking at full HP: no heal, but memory drops
    assert (
        _replay(full_hp, [("ItemDrinking", 0, 3), ("ItemDrinking", 0, 2), ("Idle", 0, 2)])[-1] == 2
    )
    unknown = EstusTracker.begin(None, None)
    assert unknown.remaining(None) is None and unknown.remaining(2) == 2


def test_next_hit_countdown():
    tables = AnimTables(
        player=TABLES.player,
        boss=TABLES.boss,
        attack_windows={"Attack3000": AttackWindows(((0.8, 1.1), (1.5, 1.8)), samples=7)},
    )

    def hit(t):
        return telegraph(snapshot(boss_anim="Attack3000", boss_anim_time_s=t), tables).next_hit_in_s

    assert hit(0.3) == pytest.approx(0.5)
    assert hit(0.9) == 0.0
    assert hit(1.2) == pytest.approx(0.3)
    assert hit(2.0) is None
    assert telegraph(snapshot(boss_anim="WalkFrontBattle"), tables).next_hit_in_s is None


def test_phase2_windows_override_phase1():
    tables = AnimTables(
        player=TABLES.player,
        boss=TABLES.boss,
        attack_windows={
            "Attack3000": AttackWindows(((0.8, 1.1),), 5),
            "p2:Attack3000": AttackWindows(((0.3, 0.5),), 5),
        },
    )
    p1 = telegraph(snapshot(boss_anim="Attack3000", boss_anim_time_s=0.4), tables)
    p2 = telegraph(snapshot(boss_anim="Attack3000", boss_anim_time_s=0.4, boss_phase=2), tables)
    assert (p1.phase, p2.phase) == ("windup", "active")
