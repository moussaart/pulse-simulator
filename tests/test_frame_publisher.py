"""Frame transport: push state frames from the worker to any number of viewers.

The GUI supervisor is a second, passive consumer of the simulation. It cannot use
:class:`AIGymServer`, because that socket belongs to the RL agent and accepts
exactly one client; a viewer competing for it would silently break training. So
frames go out over their own connection-per-subscriber channel.

Two properties matter at 200 Hz and are what these tests pin down:
  * one slow or dead viewer must not stall the simulation, and
  * a viewer that joins late must not be handed stale frames.
"""

import json
import socket
import tempfile
import threading
import time

from src.api.frame_publisher import FramePublisher

_buffers = {}


def _subscribe(port, timeout=5.0):
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    sock.sendall(json.dumps({"command": "subscribe", "kind": "frame"}).encode() + b"\n")
    sock.settimeout(timeout)
    ready = _read_frame(sock, timeout)
    assert ready["type"] == "ready"
    _buffers[sock] = b""
    return sock


def _read_frame(sock, timeout=5.0):
    sock.settimeout(timeout)
    buf = _buffers.get(sock, b"")
    while b"\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            raise AssertionError("connection closed before a frame arrived")
        buf += chunk
    line, rest = buf.split(b"\n", 1)
    _buffers[sock] = rest
    return json.loads(line.decode("utf-8"))


def test_a_subscriber_receives_published_frames():
    with FramePublisher(port=0) as pub:
        sock = _subscribe(pub.port)
        try:
            pub.publish({"type": "state", "step": 1})
            assert _read_frame(sock)["step"] == 1
            pub.publish({"type": "state", "step": 2})
            assert _read_frame(sock)["step"] == 2
        finally:
            sock.close()


def test_every_subscriber_gets_the_same_frames():
    with FramePublisher(port=0) as pub:
        a, b = _subscribe(pub.port), _subscribe(pub.port)
        try:
            pub.publish({"type": "state", "step": 7})
            assert _read_frame(a)["step"] == 7
            assert _read_frame(b)["step"] == 7
        finally:
            a.close()
            b.close()


def test_frames_published_before_a_subscriber_joins_are_not_replayed():
    """A late viewer must start from now, not from a stale backlog."""
    with FramePublisher(port=0) as pub:
        pub.publish({"type": "state", "step": 1})
        sock = _subscribe(pub.port)
        try:
            pub.publish({"type": "state", "step": 2})
            assert _read_frame(sock)["step"] == 2
        finally:
            sock.close()


def test_a_subscriber_that_never_reads_does_not_stall_the_publisher():
    """The simulation must not block on a frozen GUI."""
    with FramePublisher(port=0) as pub:
        slow = _subscribe(pub.port)
        try:
            # Far more frames than a socket buffer would hold. If publish() waited
            # for the slow reader, this would block until the timeout.
            started = time.monotonic()
            for step in range(20000):
                pub.publish({"type": "state", "step": step})
            assert time.monotonic() - started < 5.0
        finally:
            slow.close()


def test_a_dead_subscriber_is_dropped_and_does_not_break_publishing():
    with FramePublisher(port=0) as pub:
        dead = _subscribe(pub.port)
        alive = _subscribe(pub.port)
        dead.close()
        # Give the publisher a moment to notice the peer went away.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            pub.publish({"type": "state", "step": 1})
            if pub.subscriber_count == 1:
                break
            time.sleep(0.01)
        assert pub.subscriber_count == 1
        for _ in range(50):
            pub.publish({"type": "state", "step": 2})
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if _read_frame(alive)["step"] == 2:
                break
        else:
            raise AssertionError("subscriber did not receive the latest frame")
        alive.close()


def _drain_available(sock):
    try:
        sock.settimeout(0.01)
        chunk = sock.recv(65536)
    except (socket.timeout, OSError):
        return None
    return chunk


def test_an_unknown_subscribe_command_is_refused():
    with FramePublisher(port=0) as pub:
        sock = socket.create_connection(("127.0.0.1", pub.port), timeout=5.0)
        try:
            sock.sendall(json.dumps({"command": "drop-tables"}).encode() + b"\n")
            reply = _read_frame(sock)
            assert reply["type"] == "error"
        finally:
            sock.close()


def test_a_raw_connection_that_sends_nothing_is_reclaimed():
    """Half-open connections must not leak a thread each."""
    with FramePublisher(port=0) as pub:
        idle = socket.create_connection(("127.0.0.1", pub.port), timeout=5.0)
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                pub.publish({"type": "state", "step": 1})
                time.sleep(0.05)
                if _drain_available(idle) is not None:
                    break
            else:
                raise AssertionError("idle connection was never closed")
        finally:
            idle.close()


def test_publishing_without_a_subscriber_is_harmless():
    with FramePublisher(port=0) as pub:
        pub.publish({"type": "state", "step": 1})
        pub.publish({"type": "state", "step": 2})
