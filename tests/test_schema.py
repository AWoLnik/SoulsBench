import pytest
from pydantic import ValidationError

from soulsbench.scenarios import get_scenario
from soulsbench.schema import ACTION_LABELS, N_ACTIONS, Action, CombatState, Decision
from soulsbench.windows_bridge.state import build_state
from tests.fakes import TABLES, snapshot


def _state(**overrides) -> CombatState:
    return build_state(
        snapshot(**overrides),
        TABLES,
        episode_id="ep",
        scenario=get_scenario("gundyr_p1"),
        tick=0,
        t_s=0.0,
        history=(),
    )


def test_action_space_is_the_contract():
    assert N_ACTIONS == 16
    assert ACTION_LABELS == (
        "idle",
        "move_forward",
        "move_back",
        "move_left",
        "move_right",
        "roll_forward",
        "roll_back",
        "roll_left",
        "roll_right",
        "light_attack",
        "heavy_attack",
        "block",
        "parry",
        "use_estus",
        "backstep",
        "toggle_lock",
    )
    assert Action.TOGGLE_LOCK.label == "toggle_lock"


def test_state_json_roundtrip():
    state = _state()
    again = CombatState.model_validate_json(state.model_dump_json())
    assert again == state
    assert state.schema_version == "soulsbench/v1"


def test_state_rejects_unknown_fields_and_bad_masks():
    data = _state().model_dump()
    with pytest.raises(ValidationError):
        CombatState.model_validate({**data, "surprise": 1})
    with pytest.raises(ValidationError):
        CombatState.model_validate({**data, "legal_actions_mask": [True] * 15})
    with pytest.raises(ValidationError):
        CombatState.model_validate({**data, "legal_actions_mask": [False] + [True] * 15})


def test_decision_bounds_and_extra_keys():
    assert Decision.model_validate({"discrete_id": 9, "why": "close"}).discrete_id == 9
    for bad in ({"discrete_id": 16}, {"discrete_id": -1}, {}, {"discrete_id": "nine"}):
        with pytest.raises(ValidationError):
            Decision.model_validate(bad)
