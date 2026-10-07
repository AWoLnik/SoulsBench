import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from soulsbench.adapters import (
    AdapterError,
    DecisionParseError,
    HttpDecisionsAdapter,
    OpenAIDecisionsAdapter,
    RandomAdapter,
    ScriptedExpertAdapter,
    make_adapter,
)
from soulsbench.harness import DecisionGuard
from soulsbench.scenarios import get_scenario
from soulsbench.schema import SLA, Action
from soulsbench.windows_bridge.state import build_state
from tests.fakes import TABLES, snapshot


def _state(**overrides):
    return build_state(
        snapshot(**overrides),
        TABLES,
        episode_id="ep-1",
        scenario=get_scenario("gundyr_p1"),
        tick=0,
        t_s=0.0,
        history=(),
    )


def test_registry():
    assert make_adapter("idle", env={}).name == "idle"
    assert make_adapter("random", env={}).name == "random"
    assert make_adapter("expert", env={}).name == "expert"
    with pytest.raises(ValueError, match="SOULSBENCH_HTTP_BASE_URL"):
        make_adapter("http", env={})
    jev = make_adapter(
        "http:jev",
        env={"SOULSBENCH_JEV_BASE_URL": "http://jev:9000/", "SOULSBENCH_JEV_API_KEY": "k"},
    )
    assert isinstance(jev, HttpDecisionsAdapter)
    assert (jev.name, jev.base_url) == ("http:jev", "http://jev:9000")
    assert jev._client.headers["Authorization"] == "Bearer k"
    assert make_adapter("http:laya", http_base_url="http://x", env={}).base_url == "http://x"
    with pytest.raises(ValueError):
        make_adapter("nope", env={})


def test_openai_decisions_request_and_response(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no key file from the repo
    adapter = make_adapter("openai", env={"OPENAI_API_KEY": "k"})
    assert isinstance(adapter, OpenAIDecisionsAdapter)
    assert adapter.base_url + adapter.decide_path == "https://api.openai.com/v1/decisions"
    adapter.set_action_lag(0.2)
    body = adapter.encode_request(_state(), 200)
    assert body["model"] == "gpt-6-luna" and isinstance(body["input"], str)
    assert json.loads(body["input"])["tactical"]["situation"] == "neutral"
    (q,) = body["questions"]
    assert q["type"] == "choice" and q["name"] == "action"
    assert {c["value"] for c in q["choices"]} == {
        "roll_left",
        "use_estus",
        "move_back",
        "light_attack",
        "move_forward",
        "idle",
    }
    reply = {"answers": [{"type": "choice", "name": "action", "choice": "roll_left"}]}
    assert adapter.decode_response(reply).discrete_id == Action.ROLL_LEFT
    with pytest.raises(DecisionParseError):
        adapter.decode_response({"answers": [{"name": "other", "choice": "idle"}]})
    with pytest.raises(ValueError, match="no API key"):
        make_adapter("openai", env={}).preflight()
    assert make_adapter("openai@r5", env={"OPENAI_API_KEY": "k"}).name == "openai@r5"


def test_random_is_legal_and_seeded():
    state = _state(player_anim="AttackRightLight1", player_anim_time_s=0.7)
    a, b = RandomAdapter(), RandomAdapter()
    a.reset("e", seed=7)
    b.reset("e", seed=7)
    picks_a = [a.decide(state, 200).discrete_id for _ in range(50)]
    picks_b = [b.decide(state, 200).discrete_id for _ in range(50)]
    assert picks_a == picks_b
    assert all(state.is_legal(p) for p in picks_a)
    assert len(set(picks_a)) > 1


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (dict(player_hp=100), Action.USE_ESTUS),  # 22% HP, estus left
        (dict(player_hp=100, estus=0, boss_pos=(0.0, 1.0, 0.0)), Action.LIGHT_ATTACK),  # no estus
        (dict(boss_pos=(0.0, 8.0, 0.0)), Action.MOVE_FORWARD),  # far
        (dict(boss_pos=(0.0, 2.0, 0.0)), Action.LIGHT_ATTACK),  # in range
        (
            dict(boss_pos=(0.0, 2.0, 0.0), player_anim="RollingMedium", player_anim_time_s=0.1),
            Action.IDLE,
        ),
    ],
)
def test_scripted_expert(overrides, expected):
    assert ScriptedExpertAdapter().decide(_state(**overrides), 200).discrete_id == expected


class _ModelServer:
    """A stand-in remote model for the HTTP adapter tests."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                status, payload = outer.reply(self.path, body)
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def model_server():
    servers = []

    def make(reply):
        s = _ModelServer(reply)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def test_http_adapter_happy_path(model_server):
    server = model_server(lambda path, body: (200, {"discrete_id": 9, "note": "ignored"}))
    adapter = HttpDecisionsAdapter(server.url, name="http:test")
    adapter.reset("ep-1", seed=0)
    decision = adapter.decide(_state(), 200)
    assert decision.discrete_id == 9
    paths = [p for p, _ in server.requests]
    assert paths == ["/reset", "/decide"]
    body = server.requests[1][1]
    assert body["deadline_ms"] == 200
    assert body["action_labels"][13] == "use_estus"
    assert body["state"]["legal_actions_mask"][0] is True
    assert body["state"]["episode_id"] == "ep-1"
    adapter.close()


def test_http_adapter_failures(model_server):
    bad_json = HttpDecisionsAdapter(model_server(lambda p, b: (200, b"not json")).url)
    with pytest.raises(DecisionParseError):
        bad_json.decide(_state(), 200)
    bad_id = HttpDecisionsAdapter(model_server(lambda p, b: (200, {"discrete_id": 42})).url)
    with pytest.raises(DecisionParseError):
        bad_id.decide(_state(), 200)
    server_error = HttpDecisionsAdapter(model_server(lambda p, b: (500, {"error": "x"})).url)
    with pytest.raises(AdapterError):
        server_error.decide(_state(), 200)
    with pytest.raises(AdapterError):
        HttpDecisionsAdapter("http://127.0.0.1:9").decide(_state(), 200)  # nothing listening


def test_slow_http_model_times_out_under_the_guard(model_server):
    def slow(path, body):
        time.sleep(0.4)
        return 200, {"discrete_id": 9}

    adapter = HttpDecisionsAdapter(model_server(slow).url)
    guard = DecisionGuard(SLA(decide_budget_ms=100, control_period_ms=250))
    out = guard.decide(adapter, _state())
    guard.close()
    assert out.status == "timeout" and out.executed == Action.IDLE


def test_jev_request_maps_legal_actions_and_back():
    from soulsbench.adapters import JevAdapter

    mid_swing = _state(player_anim="AttackRightLight1", player_anim_time_s=0.7)
    tuned = JevAdapter("k").encode_request(mid_swing, 200)
    assert set(tuned["questions"]["action"]["criteria"]) == {"idle", "light_attack", "use_estus"}
    assert tuned["state"]["tactical"]["situation"] == "neutral"
    a = JevAdapter("k", prompt="r5")
    body = a.encode_request(mid_swing, 200)
    assert body["model"] == "jev-latest" and body["questions"]["action"]["type"] == "choice"
    assert set(body["questions"]["action"]["criteria"]) == {
        "idle",
        "light_attack",
        "heavy_attack",
        "parry",
        "use_estus",
    }  # only what the harness would execute
    assert (
        a.decode_response({"answers": {"action": {"choice": "parry"}}}).discrete_id == Action.PARRY
    )
    with pytest.raises(DecisionParseError):
        a.decode_response({"answers": {}})
    with pytest.raises(ValueError):
        JevAdapter(None).preflight()


def test_hosted_decision_apis(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no key files from the repo
    env = {"CLOUDFLARE_API_TOKEN": "cf", "CLOUDFLARE_ACCOUNT_ID": "acct", "PERPLEXITY_API_KEY": "p"}
    clef = make_adapter("clef-flash", env=env)
    assert clef.base_url + clef.decide_path == (
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash"
    )
    body = clef.encode_request(_state(), 200)
    assert "model" not in body and body["questions"]["action"]["type"] == "choice"
    wrapped = {"result": {"answers": {"action": {"choice": "roll_left"}}}, "success": True}
    assert clef.decode_response(wrapped).discrete_id == Action.ROLL_LEFT
    pplx = make_adapter("pplx@base", env=env)
    assert pplx.base_url + pplx.decide_path == "https://api.perplexity.ai/v1/decisions"
    assert pplx.encode_request(_state(), 200)["model"] == "pplx-decider-v1-27b"
    assert pplx.name == "pplx@base"
    with pytest.raises(ValueError, match="no API key"):
        make_adapter("pplx", env={}).preflight()
    with pytest.raises(ValueError, match="CLOUDFLARE_ACCOUNT_ID"):
        make_adapter("clef", env={"CLOUDFLARE_API_TOKEN": "cf"})


def test_rules_adapter_matches_jev_on_logged_situations():
    from soulsbench.adapters import RulesAdapter

    r = RulesAdapter()
    r.set_action_lag(0.2)
    far = _state(boss_pos=(0.0, 9.0, 0.0))
    assert r.decide(far, 200).discrete_id == Action.MOVE_FORWARD
    close = _state(boss_pos=(0.0, 2.0, 0.0))
    assert r.decide(close, 200).discrete_id == Action.LIGHT_ATTACK
    swing = _state(boss_pos=(0.0, 2.0, 0.0), boss_anim="Attack3000", boss_anim_time_s=0.5)
    assert make_adapter("rules", env={}).name == "rules"
    assert make_adapter("rules:150", env={}).delay_s == 0.15
    assert r.decide(swing, 200).discrete_id == Action.ROLL_LEFT  # uncalibrated attack: dodge
