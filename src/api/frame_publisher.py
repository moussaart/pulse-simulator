"""Push state frames from a training worker to any number of viewers.

:class:`~src.api.ai_gym_server.AIGymServer` is the RL action channel and accepts a
single client, which belongs to the agent. A GUI viewer is a second, passive
consumer, so frames get their own connection-per-subscriber channel.

:meth:`FramePublisher.publish` is non-blocking by construction: it appends to a
bounded per-subscriber queue and a writer thread drains it. That matters at 200 Hz
because a viewer which stops reading (a frozen or minimised GUI) would otherwise
fill its socket buffer and block ``sendall`` — stalling the simulation and
reproducing the very failure this component exists to remove. When a viewer falls
behind it loses its oldest frames and catches up, rather than holding up the run.
"""

from __future__ import annotations

import collections
import json
import os
import select
import socket
import threading
from typing import Any, Deque, Dict, List, Optional, Set

__all__ = ["FramePublisher", "PROTOCOL_VERSION"]

#: Version of the frame wire format, so a viewer can refuse an old worker.
PROTOCOL_VERSION = 1

#: How often a blocked socket call is retried. Kept short so stop() is prompt.
_POLL_INTERVAL = 0.25

#: Frames buffered per subscriber before the oldest are dropped. Small on
#: purpose: a stale position estimate is worth less than a fresh one, and the
#: viewer redraws from whatever arrives last.
_QUEUE_DEPTH = 8

#: Seconds a subscriber may have data queued with no successful send before it is
#: treated as gone. Guards against a peer whose socket never errors out.
_STALE_AFTER = 2.0


class _Subscriber:
    """One viewer: a bounded queue plus the thread that drains it."""

    __slots__ = ("sock", "queue", "lock", "dropped", "alive", "_thread", "_last_send")

    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.queue: Deque[bytes] = collections.deque(maxlen=_QUEUE_DEPTH)
        self.lock = threading.Lock()
        self.dropped = 0
        self.alive = True
        self._thread: Optional[threading.Thread] = None
        self._last_send = 0.0

    def offer(self, line: bytes) -> None:
        """Queue a frame. Never blocks; drops the oldest when full."""
        with self.lock:
            if len(self.queue) == self.queue.maxlen:
                self.dropped += 1
            self.queue.append(line)

    def _take(self) -> Optional[bytes]:
        with self.lock:
            if self.queue:
                return self.queue.popleft()
        return None

    def run_writer(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            line = self._take()
            if line is None:
                stop_event.wait(0.01)
                continue
            try:
                self.sock.sendall(line)
                self._last_send = 0.0
            except OSError:
                self.alive = False
                return


class FramePublisher:
    """Serve newline-delimited JSON frames to every connected subscriber."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.requested_port = int(port)
        self.port = int(port)
        self._lock = threading.Lock()
        self._subscribers: Set[_Subscriber] = set()
        self._sock: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> int:
        if self._sock is not None:
            return self.port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.host, self.requested_port))
        sock.listen(16)
        sock.settimeout(_POLL_INTERVAL)
        self._sock = sock
        self.port = sock.getsockname()[1]
        self._stop_event.clear()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="frame-publisher-accept", daemon=True
        )
        self._accept_thread.start()
        return self.port

    def stop(self) -> None:
        self._stop_event.set()
        sock, self._sock = self._sock, None
        if sock is not None:
            self._shutdown(sock)
        with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for sub in subscribers:
            self._shutdown(sub.sock)
        for sub in subscribers:
            if sub._thread is not None:
                sub._thread.join(timeout=2.0)
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
            self._accept_thread = None

    def __enter__(self) -> "FramePublisher":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    @staticmethod
    def _shutdown(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return sum(1 for s in self._subscribers if s.alive)

    # ── accepting ──────────────────────────────────────────────────────────

    def _accept_loop(self) -> None:
        while not self._stop_event.is_set():
            sock = self._sock
            if sock is None:
                break
            try:
                client, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client.settimeout(_POLL_INTERVAL)
            threading.Thread(
                target=self._client_loop, args=(client,),
                name="frame-publisher-client", daemon=True,
            ).start()

    def _client_loop(self, sock: socket.socket) -> None:
        """Hand-shake, register the subscriber, then run its writer thread.

        The subscriber is registered only after a valid ``subscribe``, and is
        acknowledged before the caller is expected to publish anything, so a
        viewer never misses the first frame through a registration race.
        """
        sub: Optional[_Subscriber] = None
        try:
            if not self._handshake(sock):
                return
            sub = _Subscriber(sock)
            with self._lock:
                self._subscribers.add(sub)
            self._send_line(sock, {
                "type": "ready",
                "protocol_version": PROTOCOL_VERSION,
                "pid": os.getpid(),
            })
            sock.settimeout(_POLL_INTERVAL)
            sub._thread = threading.Thread(
                target=sub.run_writer, args=(self._stop_event,),
                name="frame-publisher-writer", daemon=True,
            )
            sub._thread.start()
            # Keep the subscriber registered until it closes or the publisher
            # stops. Monitoring the receive side also detects a quiet peer that
            # disconnects while there are no frames in its outbound queue.
            while not self._stop_event.is_set() and sub.alive:
                try:
                    readable, _, _ = select.select([sock], [], [], _POLL_INTERVAL)
                except (OSError, ValueError):
                    break
                if readable:
                    try:
                        if not sock.recv(1):
                            break
                        # This is a one-way channel; post-handshake client data is
                        # a protocol violation and closes the connection.
                        break
                    except socket.timeout:
                        continue
                    except OSError:
                        break
        finally:
            if sub is not None:
                with self._lock:
                    self._subscribers.discard(sub)
                sub.alive = False
                self._shutdown(sock)
            else:
                self._shutdown(sock)

    def _handshake(self, sock: socket.socket) -> bool:
        """Read one subscribe frame. Returns False if the peer is not a viewer."""
        buf = b""
        idle = 0
        while not self._stop_event.is_set():
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                idle += 1
                # Reclaim a connection that never identifies itself.
                if idle > 8:
                    return False
                continue
            except OSError:
                return False
            if not chunk:
                return False
            buf += chunk
            if b"\n" not in buf:
                continue
            line, _ = buf.split(b"\n", 1)
            try:
                request = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_line(sock, {"type": "error", "error": "bad_json"})
                return False
            if not isinstance(request, dict) or request.get("command") != "subscribe":
                self._send_line(
                    sock, {"type": "error", "error": "expected a subscribe command"}
                )
                return False
            return True
        return False

    @staticmethod
    def _send_line(sock: socket.socket, payload: Dict[str, Any]) -> None:
        try:
            sock.sendall((json.dumps(payload, default=float) + "\n").encode("utf-8"))
        except OSError:
            pass

    # ── publishing ─────────────────────────────────────────────────────────

    def publish(self, frame: Any) -> int:
        """Queue a frame for every subscriber; returns how many accepted it.

        Non-blocking: this only appends to bounded deques. Frames are never
        replayed to a late joiner.
        """
        with self._lock:
            subscribers = list(self._subscribers)
        if not subscribers:
            return 0
        line = (json.dumps(frame, default=float) + "\n").encode("utf-8")
        delivered = 0
        for sub in subscribers:
            if sub.alive:
                sub.offer(line)
                delivered += 1
        return delivered
