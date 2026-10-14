"""The worker must bridge the external RL client to its Qt-free engine."""

import json
import socket
import threading
import time

from src.api.ai_training_engine import TrainingConfig
from src.api.ai_training_worker import TrainingWorker


def _connect(port):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=2.0)
            sock.settimeout(5.0)
            return sock, sock.makefile("r", encoding="utf-8")
        except OSError:
            time.sleep(0.02)
    raise AssertionError("worker gym server did not accept a connection")


def _read(reader):
    line = reader.readline()
    assert line, "worker closed the gym connection"
    return json.loads(line)


def test_worker_sends_observation_and_applies_agent_action(tmp_path):
    worker = TrainingWorker(
        config=TrainingConfig(
            algorithm="Trilateration",
            movement_pattern="Random Walk",
            measurement_source="uwb",
        ),
        run_dir=str(tmp_path),
        serve_http=True,
        http_port=0,
        serve_frames=True,
        frames_port=0,
        serve_gym=True,
        gym_port=0,
    )
    try:
        endpoints = worker.start_services()
        sock, reader = _connect(endpoints["gym_port"])
        try:
            hello = _read(reader)
            assert hello["type"] == "hello"
            assert hello["protocol_version"] == 3

            runner = threading.Thread(target=worker.run_forever, daemon=True)
            runner.start()
            states = _read(reader)
            assert isinstance(states, list) and len(states) == 1
            state = states[0]
            assert state["step"] == 0
            assert state["step_id"] == 0
            assert isinstance(state["measurements"]["uwb_ranges"], list)
            assert "precision" in state

            sock.sendall((json.dumps({
                "action": [[0, 1, 2]],
                "metrics": {"step_rewards": [0.75], "cumulative_rewards": [4.0]},
            }) + "\n").encode("utf-8"))
            next_states = _read(reader)
            assert next_states[0]["step"] == 1
            assert next_states[0]["step_id"] == 1
            assert worker.engine.agents[0].last_action["anchors"] == [0, 1, 2]
            assert worker.recorder.tail()[-1]["reward_step"] == 0.75
            assert worker.recorder.tail()[-1]["reward_cumulative"] == 4.0
        finally:
            reader.close()
            sock.close()
    finally:
        worker.stop()
        worker.close()
