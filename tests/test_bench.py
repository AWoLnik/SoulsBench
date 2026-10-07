import json

from soulsbench.bench import Bench, parse_args, parse_seeds
from soulsbench.schema import SLA
from tests.fakes import FakeCombatEnv


class RecordingEnv(FakeCombatEnv):
    def __init__(self):
        super().__init__()
        self.started, self.stopped = [], []

    def record_start(self, name, label, subdir, password):
        self.started.append((name, subdir))

    def record_stop(self, suffix):
        self.stopped.append(suffix)
        return f"C:/v/{self.started[-1][0]}_{suffix}.mkv"


def _bench(tmp_path, *extra):
    sla = SLA(decide_budget_ms=50, min_action_interval_ms=0)
    args = parse_args(
        [
            "--name",
            "t",
            "--adapters",
            "expert,idle",
            "--seeds",
            "3",
            "--scenario",
            "gundyr_p1",
            "--out",
            str(tmp_path),
            "--no-log-states",
            *extra,
        ]
    )
    return Bench(args, sla)


def test_interleaves_rotates_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr("soulsbench.bench.time.sleep", lambda s: None)
    env = RecordingEnv()
    assert _bench(tmp_path, "--record-seeds", "1").run(env) == 0
    root = tmp_path / "bench_t"
    order = [
        (json.loads(line)["seed"], name)
        for name in ("expert", "idle")
        for line in (root / name / "episodes.jsonl").read_text().splitlines()
    ]
    assert sorted(order) == [(s, n) for s in range(3) for n in ("expert", "idle")]
    assert [n for n, _ in env.started] == ["idle_seed001", "expert_seed001"]  # rotated round 1
    assert env.stopped == ["death", "win"]
    assert "| expert |" in (root / "comparison.md").read_text()
    # a second run with the same plan has nothing left to do
    assert _bench(tmp_path).run(RecordingEnv()) == 0
    assert len((root / "idle" / "episodes.jsonl").read_text().splitlines()) == 3


def test_resume_refuses_a_different_plan(tmp_path):
    import pytest

    _bench(tmp_path).check_plan()
    other = _bench(tmp_path)
    other.seeds = [0, 1]
    with pytest.raises(SystemExit, match="different plan"):
        other.check_plan()


def test_parse_seeds():
    assert parse_seeds("0-2,7") == {0, 1, 2, 7} and parse_seeds("") == set()


def test_record_best_keeps_one_video_per_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr("soulsbench.bench.time.sleep", lambda s: None)
    videos = tmp_path / "Videos" / "SoulsBench" / "t"
    videos.mkdir(parents=True)

    class FileEnv(RecordingEnv):
        def record_stop(self, suffix):
            path = videos / f"{self.started[-1][0]}_{suffix}.mp4"
            path.write_bytes(b"x")
            return str(path)

    assert _bench(tmp_path, "--record-best").run(FileEnv()) == 0
    kept = sorted(p.name for p in videos.iterdir())
    assert len(kept) == 2  # one per adapter, the rest deleted as the bench went
    assert any(n.startswith("expert_") and n.endswith("_win.mp4") for n in kept)
    best = json.loads((tmp_path / "bench_t" / "best_videos.json").read_text())
    assert set(best) == {"expert", "idle"}


def test_unresponsive_adapter_episode_is_aborted_and_stops(tmp_path, monkeypatch):
    from soulsbench.harness import DecisionOutcome

    monkeypatch.setattr("soulsbench.bench.time.sleep", lambda s: None)
    monkeypatch.setattr(
        "soulsbench.harness.DecisionGuard.finish",
        lambda self, pending, apply_to: DecisionOutcome(None, 0, "timeout", 50.0),
    )
    assert _bench(tmp_path).run(FakeCombatEnv()) == 3
    (line,) = (tmp_path / "bench_t" / "expert" / "episodes.jsonl").read_text().splitlines()
    ep = json.loads(line)
    assert ep["outcome"] == "aborted" and "unresponsive" in ep["error"]
