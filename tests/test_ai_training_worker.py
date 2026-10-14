"""The worker owns the engine headlessly: config validation, stepping, metrics
publication, lifecycle, and the control surface the GUI drives.

It must import and run with no Qt present, because "Charts only" and "Headless"
run it in a bare interpreter.
"""

import importlib
import json
import sys
import threading
import time

import pytest

from src.api.ai_training_engine import TrainingConfig, UnknownAlgorithmError
from src.api.ai_training_worker import (
    TrainingWorker,
    default_anchor_layout,
    load_worker_config,
)


def _config_dict(**overrides):
    base = {
        "algorithm": "Trilateration",
        "dt": 0.005,
        "movement_pattern": "Random Walk",
        "movement_speed": 1.0,
        "measurement_source": "uwb",
        "num_agents": 2,
        "allow_gpu": False,
    }
    base.update(overrides)
    return base


def _worker(tmp_path, **overrides):
    run_dir = tmp_path / f"run-{overrides.get('num_agents', 2)}-{len(list(tmp_path.iterdir()))}"
    return TrainingWorker(
        config=TrainingConfig.from_dict(_config_dict(**overrides)),
        run_dir=str(run_dir),
    )


def test_the_worker_module_does_not_pull_in_qt():
    for name in [m for m in sys.modules if m.startswith(("PyQt5", "PySide"))]:
        del sys.modules[name]

    module = importlib.import_module("src.api.ai_training_worker")
    importlib.reload(module)
    assert not [m for m in sys.modules if m.startswith(("PyQt5", "PySide"))]


def test_config_round_trips_through_json(tmp_path):
    path = tmp_path / "worker.json"
    path.write_text(
        json.dumps(_config_dict(algorithm="Duty-Cycled UWB-IMU NA-AEKF", num_agents=4)),
        encoding="utf-8",
    )
    config = load_worker_config(path)
    assert config.algorithm == "Duty-Cycled UWB-IMU NA-AEKF"
    assert config.num_agents == 4


def test_a_missing_config_file_is_not_fatal(capsys):
    config = load_worker_config("does-not-exist.json")
    assert config.algorithm == TrainingConfig().algorithm
    assert config.movement_pattern == "Random Walk"
    assert "not found" in capsys.readouterr().out


def test_a_malformed_config_file_is_an_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError):
        load_worker_config(path)


def test_unknown_config_keys_are_reported_not_ignored_silently(tmp_path, capsys):
    path = tmp_path / "typo.json"
    payload = _config_dict()
    payload["dtt"] = 0.01  # typo for dt
    path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_worker_config(path)
    assert config.dt == 0.005
    assert "dtt" in capsys.readouterr().out


@pytest.mark.parametrize(
    "override, error",
    [
        ({"algorithm": "No Such Filter"}, UnknownAlgorithmError),
        ({"movement_pattern": "random"}, LookupError),
        ({"dt": 0.0}, ValueError),
        ({"num_agents": 0}, ValueError),
    ],
)
def test_bad_config_fails_at_startup_not_mid_run(override, error):
    """Hours of training must not be lost to a typo discovered at step 900k."""
    with pytest.raises(error):
        TrainingWorker(config=TrainingConfig.from_dict(_config_dict(**override)),
                       run_dir="unused")


def test_the_default_anchor_layout_is_a_usable_rectangle():
    anchors = default_anchor_layout(4)
    assert len(anchors) == 4
    xs = {round(a.position.x, 6) for a in anchors}
    ys = {round(a.position.y, 6) for a in anchors}
    assert len(xs) == 2 and len(ys) == 2, "anchors must form a rectangle, not a line"
    assert len(default_anchor_layout(7)) == 7
    with pytest.raises(ValueError):
        default_anchor_layout(0)


def test_tick_advances_one_step_and_returns_a_state_frame(tmp_path):
    worker = _worker(tmp_path, num_agents=2)
    frame = worker.tick()
    assert frame["type"] == "state"
    assert frame["step_id"] == 0
    assert frame["protocol_version"] == 3
    assert len(frame["agents"]) == 2
    assert worker.status["step"] == 1
    worker.close()


def test_the_state_frame_is_json_serialisable(tmp_path):
    """It crosses a socket and the JSONL sink, so numpy types must be gone."""
    worker = _worker(tmp_path, num_agents=2)
    frame = worker.tick()
    json.dumps(frame)
    worker.close()


def test_tick_publishes_one_metrics_record_per_agent_per_step(tmp_path):
    worker = _worker(tmp_path, num_agents=2)
    for _ in range(3):
        worker.tick()
    records = worker.recorder.tail()
    assert len(records) == 6
    assert [r["step"] for r in records] == [0, 0, 1, 1, 2, 2]
    assert {r["agent_id"] for r in records} == {0, 1}
    for record in records:
        assert "error" in record
        assert "step_energy_uJ" in record
    worker.close()


def test_reward_is_recorded_once_the_client_supplies_it(tmp_path):
    worker = _worker(tmp_path, num_agents=1)
    worker.tick()
    worker.set_reward(step_reward=1.5, cumulative_reward=3.0)
    worker.tick()
    records = worker.recorder.tail()
    assert records[0]["reward_step"] is None
    assert records[1]["reward_step"] == 1.5
    assert records[1]["reward_cumulative"] == 3.0
    worker.close()


def test_a_submitted_action_is_used_and_never_replayed(tmp_path):
    """The frame echoes the action that drove it, so convergence cannot mask it."""
    worker = _worker(tmp_path, num_agents=1, algorithm="Duty-Cycled UWB-IMU NA-AEKF")
    worker.submit_action(0, {"filter": "Duty-Cycled UWB-IMU NA-AEKF", "anchors": [0, 1],
                             "measurement_source": "uwb", "t_imu": 0.5, "t_uwb": 0.25})
    frame = worker.tick()
    assert frame["actions"][0]["anchors"] == [0, 1]
    assert frame["actions"][0]["t_imu"] == 0.5
    assert frame["actions"][0]["t_uwb"] == 0.25

    # Queue is drained: the next step falls back to the default full-anchor action.
    frame = worker.tick()
    assert frame["actions"][0]["anchors"] == [0, 1, 2, 3]
    assert "t_imu" not in frame["actions"][0], "stale action was replayed"
    worker.close()


def test_pause_holds_the_step_counter_and_resume_continues(tmp_path):
    worker = _worker(tmp_path)
    worker.tick()
    worker.pause()
    assert worker.status["phase"] == "paused"
    worker.tick()
    assert worker.status["step"] == 1, "a paused worker must not advance"
    worker.resume()
    worker.tick()
    assert worker.status["step"] == 2
    worker.close()


def test_reset_returns_to_step_zero_and_clears_agent_state(tmp_path):
    worker = _worker(tmp_path, num_agents=2)
    worker.tick()
    worker.tick()
    worker.reset()
    assert worker.status["step"] == 0
    assert worker.status["phase"] == "running"
    assert all(agent.state is None for agent in worker.engine.agents)
    frame = worker.tick()
    assert frame["step_id"] == 0
    worker.close()


def test_a_stopped_worker_does_not_advance(tmp_path):
    worker = _worker(tmp_path)
    worker.tick()
    worker.stop()
    assert worker.status["phase"] == "stopped"
    before = worker.status["step"]
    worker.tick()
    assert worker.status["step"] == before
    worker.resume()
    assert worker.status["phase"] == "stopped", "stop must not be undone by resume"
    worker.close()


def test_run_forever_returns_promptly_when_stopped(tmp_path):
    """A GUI close must not leave an orphaned thread simulating for hours."""
    worker = _worker(tmp_path)
    thread = worker.run_in_thread()
    deadline = time.time() + 5.0
    while worker.status["step"] < 5 and time.time() < deadline:
        time.sleep(0.01)
    assert worker.status["step"] >= 5

    worker.stop()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "run_forever ignored stop()"
    worker.close()


def test_run_forever_honours_max_steps(tmp_path):
    worker = _worker(tmp_path)
    assert worker.run_forever(max_steps=7) == 7
    assert worker.status["step"] == 7
    worker.close()


def test_status_reports_phase_and_health(tmp_path):
    worker = _worker(tmp_path)
    worker.tick()
    status = worker.status
    assert status["phase"] == "running"
    assert status["step"] == 1
    assert status["num_agents"] == worker.config.num_agents
    assert status["algorithm"] == "Trilateration"
    assert status["num_anchors"] == 4
    assert status["uptime_s"] >= 0.0
    worker.close()


def test_the_dashboard_serves_the_workers_own_metrics(tmp_path):
    import urllib.request

    worker = TrainingWorker(
        config=TrainingConfig.from_dict(_config_dict(num_agents=1)),
        run_dir=str(tmp_path / "run"),
        serve_http=True,
        http_port=0,
    )
    try:
        port = worker.http.start()
        worker.tick()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/metrics", timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        assert body["tail"], "dashboard served no metrics"
        assert body["tail"][0]["step"] == 0

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=5) as resp:
            status = json.loads(resp.read().decode("utf-8"))
        assert status["phase"] == "running"
        assert status["records"] >= 1
    finally:
        worker.close()


def test_the_cli_runs_a_bounded_number_of_steps(tmp_path):
    from src.api.ai_training_worker import main

    config_path = tmp_path / "cli.json"
    config_path.write_text(json.dumps(_config_dict(num_agents=1, allow_gpu=False)),
                           encoding="utf-8")
    code = main([
        "--config", str(config_path),
        "--num-anchors", "4",
        "--run-dir", str(tmp_path / "cli-run"),
        "--max-steps", "3",
    ])
    assert code == 0
    lines = (tmp_path / "cli-run" / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
