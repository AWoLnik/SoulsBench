"""Bridge server + WSL client over real HTTP, with a test-double backend instead of the game."""

import sys

import httpx
import pytest

from soulsbench.adapters import ScriptedExpertAdapter
from soulsbench.backends.base import EnvError
from soulsbench.backends.relay import InteropRelay
from soulsbench.backends.souls_gym import (
    BridgeUnavailable,
    SoulsGymBridgeEnv,
    connect_bridge,
    wsl_windows_host_ip,
)
from soulsbench.harness import DecisionGuard, run_episode
from soulsbench.scenarios import get_scenario
from soulsbench.schema import DEFAULT_SLA, ResetRequest
from soulsbench.windows_bridge.server import BridgeServer
from tests.fakes import FakeBackend


@pytest.fixture
def server():
    srv = BridgeServer(FakeBackend(), ["127.0.0.1"], 0)
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def bridge(server):
    return server.backend, server.addresses[0]


def test_health_spec_reset_step(bridge):
    backend, url = bridge
    env = SoulsGymBridgeEnv(url)
    assert env.health().ready
    spec = env.spec()
    assert spec["n_actions"] == 16 and "gundyr_p1" in spec["scenarios"]
    state = env.reset(ResetRequest(episode_id="e", scenario="gundyr_p1", seed=0)).state
    assert state.tick == 0
    resp = env.step(9)
    assert resp.state.tick == 1 and resp.info.executed_action == 9 and resp.info.damage_dealt == 100
    env.close()
    assert backend.closed == 1


def test_errors_map_to_env_errors(bridge):
    _, url = bridge
    env = SoulsGymBridgeEnv(url)
    with pytest.raises(EnvError, match="conflict"):
        env.step(0)  # no episode yet
    with pytest.raises(EnvError, match="bad_request"):
        env.reset(ResetRequest(episode_id="e", scenario="no_such_scenario", seed=0))
    r = httpx.post(f"{url}/step", json={"discrete_id": 99})
    assert r.status_code == 400 and r.json()["error"] == "bad_request"
    assert httpx.get(f"{url}/nope").status_code == 404


def test_full_episode_over_http(bridge):
    _, url = bridge
    env = SoulsGymBridgeEnv(url)
    guard = DecisionGuard(DEFAULT_SLA)
    m = run_episode(
        env, ScriptedExpertAdapter(), guard, run_id="r", scenario=get_scenario("gundyr_p1"), seed=0
    )
    guard.close()
    assert m.outcome == "win"
    assert m.step_rtt_ms_p95 < 100


def test_shutdown_endpoint(server):
    assert not server.shutdown_requested.is_set()
    assert httpx.post(f"{server.addresses[0]}/shutdown", json={}).json() == {"ok": True}
    assert server.shutdown_requested.is_set()


def test_gateway_parsing(tmp_path):
    route = tmp_path / "route"
    route.write_text(
        "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n"
        "eth0\t00000000\t011019AC\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"
        "eth0\t001019AC\t00000000\t0001\t0\t0\t0\t00F0FFFF\t0\t0\t0\n"
    )
    assert wsl_windows_host_ip(route) == "172.25.16.1"


def test_connect_explicit(server):
    conn = connect_bridge(server.addresses[0] + "/", env={})
    assert (conn.url, conn.via) == (server.addresses[0], "explicit")
    with pytest.raises(BridgeUnavailable):
        connect_bridge("http://127.0.0.1:9", env={})


def test_full_episode_through_the_interop_relay(server):
    """The relay pumps bytes through a child process's stdin/stdout. Linux Python stands in for the
    Windows interpreter here; on a real machine it is the bridge venv's python.exe."""
    port = int(server.addresses[0].rsplit(":", 1)[1])
    relay = InteropRelay(sys.executable, port)
    env = SoulsGymBridgeEnv(relay.url)
    guard = DecisionGuard(DEFAULT_SLA)
    m = run_episode(
        env, ScriptedExpertAdapter(), guard, run_id="r", scenario=get_scenario("gundyr_p1"), seed=0
    )
    guard.close()
    env.close()
    relay.close()
    assert m.outcome == "win" and m.ticks > 10
