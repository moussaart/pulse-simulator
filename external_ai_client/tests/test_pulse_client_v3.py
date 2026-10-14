"""The packaged client must consume v3 handshake and state contracts."""

import json
import socket
import threading

import pytest

from pulse_ai_client.client import PulseClient, ProtocolError
from pulse_ai_client.env import PulseRLEnv
from pulse_ai_client.state import PulseState
from agent_switch_mo.env_wrapper import DEFAULT_FILTER_ALGORITHM, MacroSwitchWrapper


def _listen():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    return server, server.getsockname()[1]


def _line(sock):
    reader = sock.makefile("r", encoding="utf-8")
    line = reader.readline()
    reader.close()
    return json.loads(line)


def test_state_parser_accepts_training_engine_v3_observation():
    state = PulseState.from_dict({
        "step_id": 12,
        "protocol_version": 3,
        "timestamp": 0.06,
        "tag_position_gt": [1.0, 2.0],
        "tag_position_estimated": [1.1, 1.9],
        "anchor_positions": [[0, 0, 0], [3, 0, 0]],
        "anchor_ids": ["A0", "A1"],
        "measurements": {"source": "uwb", "uwb_ranges": [2.2, 2.5], "true_distances": [2.0, 2.3]},
        "algorithm": {"name": "Duty-Cycled UWB-IMU NA-AEKF"},
        "precision": {"localization_error": 0.14, "prev_localization_error": 0.2},
        "energy": {"total_power_mW": 4.0, "step_energy_uJ": 20.0, "cumulative_energy_J": 0.0001},
        "duty_cycle": None,
        "reward": {"step_reward": 0.5, "cumulative_reward": 3.0},
    })
    assert state.step == 12
    assert state.step_id == 12
    assert state.protocol_version == 3
    assert state.measurements.uwb_ranges == [2.2, 2.5]
    assert state.algorithm.name == "Duty-Cycled UWB-IMU NA-AEKF"
    assert state.reward["step_reward"] == 0.5
    assert state.duty_cycle.cycle_length == 0.0


def test_switch_wrapper_uses_the_registered_duty_cycled_algorithm_name():
    assert DEFAULT_FILTER_ALGORITHM == "Duty-Cycled UWB-IMU NA-AEKF"
    env = PulseRLEnv(num_anchors=4)
    wrapper = MacroSwitchWrapper(env)
    env.client.algorithms = ["Trilateration", DEFAULT_FILTER_ALGORITHM]
    assert wrapper._resolve_algorithm() == DEFAULT_FILTER_ALGORITHM
    env.client.algorithms = ["Trilateration"]
    assert wrapper._resolve_algorithm() == DEFAULT_FILTER_ALGORITHM


def test_client_consumes_hello_before_returning_the_first_state():
    server, port = _listen()
    def serve():
        conn, _ = server.accept()
        with conn:
            conn.sendall((json.dumps({
                "type": "hello", "protocol_version": 3, "num_agents": 1,
                "algorithms": ["Trilateration"],
            }) + "\n").encode())
            conn.sendall((json.dumps([{
                "step_id": 4, "protocol_version": 3,
                "tag_position_gt": [1.0, 2.0],
                "measurements": {"source": "uwb", "uwb_ranges": [1.0]},
                "algorithm": {"name": "Trilateration"},
            }]) + "\n").encode())
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = PulseClient(host="127.0.0.1", port=port, timeout=2.0)
    try:
        client.connect(retries=1)
        states = client.receive_state()
        assert len(states) == 1
        assert states[0].step == 4
        assert client.handshake["num_agents"] == 1
    finally:
        client.close()
        thread.join(timeout=2.0)


def test_client_refuses_incompatible_protocol_version():
    server, port = _listen()
    def serve():
        conn, _ = server.accept()
        with conn:
            conn.sendall(b'{"type":"hello","protocol_version":2}\n')
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = PulseClient(host="127.0.0.1", port=port, timeout=2.0)
    try:
        with pytest.raises(ProtocolError, match="protocol version"):
            client.connect(retries=1)
    finally:
        client.close()
        thread.join(timeout=2.0)


def test_client_surfaces_server_error_frames():
    server, port = _listen()
    def serve():
        conn, _ = server.accept()
        with conn:
            conn.sendall(b'{"type":"hello","protocol_version":3}\n')
            conn.sendall(b'{"type":"error","code":"bad_frame","message":"bad"}\n')
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    client = PulseClient(host="127.0.0.1", port=port, timeout=2.0)
    try:
        client.connect(retries=1)
        with pytest.raises(ProtocolError, match="bad_frame"):
            client.receive_state()
    finally:
        client.close()
        thread.join(timeout=2.0)


def test_environment_reset_sends_server_reset_after_initial_connection():
    server, port = _listen()
    commands = []
    def serve():
        conn, _ = server.accept()
        with conn:
            conn.sendall(b'{"type":"hello","protocol_version":3}\n')
            conn.sendall((json.dumps([{"step_id": 0, "protocol_version": 3}]) + "\n").encode())
            reader = conn.makefile("r", encoding="utf-8")
            commands.append(json.loads(reader.readline()))
            conn.sendall(b'{"type":"ack","command":"reset"}\n')
            conn.sendall((json.dumps([{"step_id": 0, "protocol_version": 3}]) + "\n").encode())
            reader.close()
        server.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    env = PulseRLEnv(host="127.0.0.1", port=port, num_anchors=4)
    try:
        env.reset()
        _, info = env.reset()
        assert commands == [{"command": "reset"}]
        assert info["states"][0].step == 0
    finally:
        env.close()
        thread.join(timeout=2.0)


def test_environment_does_not_hide_protocol_errors_as_disconnects():
    class FailingClient:
        def send_action(self, *args, **kwargs):
            pass

        def receive_state(self):
            raise ProtocolError("server error bad_frame: bad action")

    env = PulseRLEnv(num_anchors=4)
    env.client = FailingClient()
    env._connected = True
    with pytest.raises(ProtocolError, match="bad_frame"):
        env.step([1, 0, 0, 0])
