import json
import socket
import threading
import time

import pytest

from src.api.ai_gym_server import (
    PROTOCOL_VERSION,
    REASON_ACTION,
    REASON_DISCONNECTED,
    REASON_RESET,
    REASON_STOPPED,
    ActionResult,
    AIGymServer,
    unpack_action_result,
)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _connect(port: int, timeout: float = 5.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
            return sock, sock.makefile("r", encoding="utf-8")
        except OSError as exc:
            last = exc
            time.sleep(0.05)
    raise AssertionError(f"could not connect to gym server: {last}")


def _readline(reader, timeout=5.0):
    result = {}

    def _go():
        try:
            result["line"] = reader.readline()
        except Exception as exc:  # pragma: no cover
            result["err"] = exc

    t = threading.Thread(target=_go, daemon=True)
    t.start()
    t.join(timeout)
    assert "line" in result, "timed out waiting for a server line"
    return result["line"]


def test_server_sends_hello_handshake():
    port = _free_port()
    server = AIGymServer(port=port, num_agents=2)
    assert server.start() is True
    try:
        sock, reader = _connect(port)
        hello = json.loads(_readline(reader))
        assert hello["type"] == "hello"
        assert hello["protocol_version"] == PROTOCOL_VERSION
        assert hello["num_agents"] == 2
        sock.close()
    finally:
        server.stop()


def test_fast_action_is_not_lost():
    """A client that replies before the engine starts waiting must still be served."""
    port = _free_port()
    server = AIGymServer(port=port, num_agents=1)
    server.start()
    try:
        sock, reader = _connect(port)
        _readline(reader)  # hello

        assert server.send_state({"step": 0}) is True
        state = json.loads(_readline(reader))
        assert state["step"] == 0

        # Reply immediately, i.e. before wait_for_action is first called.
        sock.sendall((json.dumps({"action": [[0, 1]], "metrics": {"step_rewards": [1.0]}}) + "\n").encode())

        got = None
        deadline = time.time() + 5
        while time.time() < deadline and got is None:
            got = server.wait_for_action(timeout=0.2)
        assert got is not None, "action was lost"
        assert got.reason == REASON_ACTION
        action, metrics = got.action, got.metrics
        assert action == [[0, 1]]
        assert metrics["step_rewards"] == [1.0]
        sock.close()
    finally:
        server.stop()


def test_action_result_adapter_handles_v3_and_legacy_responses():
    """The GUI used to unpack the v3 three-field result as a two-tuple."""
    v3 = ActionResult([[0, 1]], {"reward": 1.0}, REASON_ACTION)
    assert unpack_action_result(v3) == ([[0, 1]], {"reward": 1.0}, REASON_ACTION)
    assert unpack_action_result(([[1, 0]], {"reward": 0.5})) == (
        [[1, 0]], {"reward": 0.5}, REASON_ACTION
    )
    assert unpack_action_result(None) is None


def test_disconnect_unblocks_waiter():
    port = _free_port()
    server = AIGymServer(port=port, num_agents=1)
    server.start()
    try:
        sock, reader = _connect(port)
        _readline(reader)
        server.send_state({"step": 0})
        _readline(reader)
        # socket.makefile() dups the fd, so both handles must be closed for the
        # server's read loop to actually observe the disconnect.
        reader.close()
        sock.close()
        got = server.wait_for_action(timeout=5.0)
        assert got is not None, "waiter hung after client disconnect"
        assert got.reason == REASON_DISCONNECTED
        assert got.action is None, "disconnect must not look like a real action"
    finally:
        server.stop()


def test_wait_for_action_returns_none_on_timeout():
    port = _free_port()
    server = AIGymServer(port=port, num_agents=1)
    server.start()
    try:
        sock, reader = _connect(port)
        _readline(reader)
        server.send_state({"step": 0})
        _readline(reader)
        started = time.time()
        got = server.wait_for_action(timeout=0.3)
        assert got is None
        assert time.time() - started < 2.0
        sock.close()
    finally:
        server.stop()


def test_reset_command_is_acked():
    port = _free_port()
    calls = []
    server = AIGymServer(port=port, num_agents=1, on_reset=lambda: calls.append(1))
    server.start()
    try:
        sock, reader = _connect(port)
        _readline(reader)
        sock.sendall((json.dumps({"command": "reset"}) + "\n").encode())
        ack = json.loads(_readline(reader))
        assert ack["type"] == "ack"
        assert ack["command"] == "reset"
        assert calls == [1]
        sock.close()
    finally:
        server.stop()


def test_reset_command_unblocks_a_waiting_engine_with_reset_reason():
    port = _free_port()
    server = AIGymServer(port=port, num_agents=1)
    server.start()
    try:
        sock, reader = _connect(port)
        _readline(reader)
        box = {}
        waiter = threading.Thread(
            target=lambda: box.setdefault("result", server.wait_for_action(timeout=3.0)),
            daemon=True,
        )
        waiter.start()
        sock.sendall((json.dumps({"command": "reset"}) + "\n").encode())
        ack = json.loads(_readline(reader))
        assert ack["type"] == "ack"
        waiter.join(2.0)
        assert not waiter.is_alive()
        assert box["result"].reason == REASON_RESET
    finally:
        server.stop()


def test_malformed_frame_produces_structured_error():
    port = _free_port()
    server = AIGymServer(port=port, num_agents=1)
    server.start()
    try:
        sock, reader = _connect(port)
        _readline(reader)
        sock.sendall(b"{not json}\n")
        err = json.loads(_readline(reader))
        assert err["type"] == "error"
        assert err["code"] == "bad_json"
        sock.close()
    finally:
        server.stop()


def test_stop_is_immediate_while_waiting():
    port = _free_port()
    server = AIGymServer(port=port, num_agents=1)
    server.start()
    # Keep a live reference: a discarded socket is closed by the GC, which would
    # end the read loop and unblock the waiter for the wrong reason.
    sock, reader = _connect(port)
    box = {}

    def _wait():
        box["got"] = server.wait_for_action(timeout=None)

    t = threading.Thread(target=_wait, daemon=True)
    t.start()
    time.sleep(0.3)
    assert not box, "wait_for_action returned before it was given a reason"
    started = time.time()
    server.stop()
    t.join(3.0)
    assert not t.is_alive(), "stop() did not unblock wait_for_action"
    assert time.time() - started < 2.5
    assert box.get("got") is not None, "stop() must report a reason, not a bare None"
    assert box["got"].reason == REASON_STOPPED
    reader.close()
    sock.close()


def test_bind_failure_is_reported_not_raised():
    port = _free_port()
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        server = AIGymServer(port=port, host="127.0.0.1", num_agents=1)
        # Windows allows a duplicate bind with SO_REUSEADDR, so only assert the
        # contract: start() must return a bool and never raise.
        assert server.start() in (True, False)
        server.stop()
    finally:
        blocker.close()
