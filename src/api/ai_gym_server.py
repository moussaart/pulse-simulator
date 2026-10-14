"""Bidirectional TCP server exposing the training engine as a Gymnasium backend.

Protocol v3 changes relative to v2:
    * a ``hello`` handshake frame carrying the protocol version, agent count and
      the registered algorithm names, so a client can validate before stepping;
    * a ``queue.Queue`` action mailbox instead of a clearable ``Event``, which
      removes the lost-action race where a fast reply arrived before the engine
      started waiting and was then wiped by ``Event.clear()``;
    * disconnect delivers an explicit sentinel so neither peer can hang;
    * ``{"command": "reset"}`` is acknowledged and forwarded to the engine;
    * malformed frames produce structured ``{"type": "error"}`` frames.
"""

from __future__ import annotations

import json
import queue
import selectors
import socket
import threading
import time
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional

PROTOCOL_VERSION = 3

_MAX_LINE_BYTES = 8 * 1024 * 1024

#: Outcome of :meth:`AIGymServer.wait_for_action`. ``reason`` distinguishes the
#: three cases that used to collapse into a bare ``None``, so a caller can tell
#: "no reply yet" from "the peer is gone" from "the server was stopped".
ActionResult = namedtuple("ActionResult", "action metrics reason")

REASON_ACTION = "action"
REASON_DISCONNECTED = "disconnected"
REASON_STOPPED = "stopped"
REASON_RESET = "reset"


def unpack_action_result(result: Any):
    """Return ``(action, metrics, reason)`` for v3 and legacy results.

    ``None`` remains the timeout sentinel. The named fields are preferred for
    v3 because ``ActionResult`` deliberately has three values; treating it as
    the old ``(action, metrics)`` tuple raises ``ValueError`` in GUI slots.
    """
    if result is None:
        return None
    if all(hasattr(result, name) for name in ("action", "metrics", "reason")):
        return result.action, result.metrics, result.reason
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[0], result[1], REASON_ACTION
    raise TypeError("unsupported action result: %r" % (result,))


class _Sentinel:
    """Marker pushed onto the mailbox when the peer is gone."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<DISCONNECTED>"


DISCONNECTED = _Sentinel()
RESET_REQUESTED = _Sentinel()


class AIGymServer:
    """TCP server that sends ``state`` and waits for ``action``."""

    def __init__(
        self,
        port: int = 5555,
        host: str = "0.0.0.0",
        num_agents: int = 1,
        algorithms: Optional[List[str]] = None,
        on_reset: Optional[Callable[[], None]] = None,
    ):
        self.port = int(port)
        self.host = host
        self.num_agents = int(num_agents)
        self.algorithms = list(algorithms or [])
        self.on_reset = on_reset

        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.client_address = None

        self.is_running = False
        self._server_thread: Optional[threading.Thread] = None

        self._mailbox: "queue.Queue[Any]" = queue.Queue(maxsize=1)
        self._connected = False
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._last_error: Optional[Dict[str, Any]] = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def protocol_version(self) -> int:
        return PROTOCOL_VERSION

    @property
    def last_error(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return dict(self._last_error) if self._last_error else None

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the accept loop in a background thread.

        Returns True once the listening socket is bound, so callers can report
        a port clash immediately instead of failing later at the first step.
        """
        if self.is_running:
            return True

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            sock.listen(1)
            sock.settimeout(0.5)
        except OSError as exc:
            self._set_error("bind_failed", f"could not bind port {self.port}: {exc}")
            print(f"[AIGymServer] Bind failed on port {self.port}: {exc}")
            return False

        self.server_socket = sock
        # Port 0 is useful for isolated worker processes and tests; retain the
        # kernel-assigned port so readiness metadata can advertise it.
        self.port = int(sock.getsockname()[1])
        self.is_running = True
        self._drain_mailbox()
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        print(f"[AIGymServer] Listening on {self.host}:{self.port} (protocol v{PROTOCOL_VERSION})")
        return True

    def stop(self) -> None:
        """Stop accepting, close sockets, and unblock every waiter immediately."""
        self.is_running = False

        client = self.client_socket
        self.client_socket = None
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass

        server = self.server_socket
        self.server_socket = None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass

        with self._lock:
            self._connected = False
        self._drain_mailbox()
        self._push(DISCONNECTED)
        print("[AIGymServer] Stopped.")

    @property
    def _stop_requested(self) -> bool:
        return not self.is_running

    # ── Mailbox ─────────────────────────────────────────────────────────────

    def _drain_mailbox(self) -> None:
        while True:
            try:
                self._mailbox.get_nowait()
            except queue.Empty:
                return

    def _push(self, item: Any) -> None:
        """Best-effort push that never blocks the reader thread."""
        try:
            self._mailbox.put_nowait(item)
        except queue.Full:
            try:
                self._mailbox.get_nowait()
            except queue.Empty:
                pass
            try:
                self._mailbox.put_nowait(item)
            except queue.Full:
                pass

    # ── Internals ───────────────────────────────────────────────────────────

    def _set_error(self, code: str, message: str) -> None:
        with self._lock:
            self._last_error = {"code": code, "message": message}

    def _run_server(self) -> None:
        while self.is_running and self.server_socket is not None:
            try:
                client_sock, addr = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                if self.is_running:
                    print("[AIGymServer] Accept error")
                break

            if not self.is_running:
                try:
                    client_sock.close()
                except OSError:
                    pass
                break

            print(f"[AIGymServer] Client connected from {addr}")
            previous = self.client_socket
            self.client_socket = client_sock
            self.client_address = addr
            if previous is not None:
                try:
                    previous.close()
                except OSError:
                    pass
            with self._lock:
                self._connected = True
            self._drain_mailbox()

            self._send_hello()
            self._read_loop(client_sock)

            with self._lock:
                self._connected = False
            if self.client_socket is client_sock:
                self.client_socket = None
            self._push(DISCONNECTED)
            print("[AIGymServer] Client disconnected.")

    def _send_hello(self) -> None:
        self._send_json({
            "type": "hello",
            "protocol_version": PROTOCOL_VERSION,
            "num_agents": self.num_agents,
            "algorithms": self.algorithms,
        })

    def _send_json(self, payload: Dict[str, Any]) -> bool:
        client = self.client_socket
        if client is None:
            return False
        try:
            message = (json.dumps(payload, default=float) + "\n").encode("utf-8")
            with self._send_lock:
                client.sendall(message)
            return True
        except OSError as exc:
            self._set_error("send_failed", str(exc))
            return False

    def _read_loop(self, sock: socket.socket) -> None:
        """Read newline-delimited frames without blocking shutdown."""
        sel = selectors.DefaultSelector()
        try:
            sel.register(sock, selectors.EVENT_READ)
        except (KeyError, ValueError):
            return

        buffer = b""
        sock.settimeout(0.2)
        try:
            while self.is_running:
                events = sel.select(timeout=0.2)
                if not events:
                    continue
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk
                if len(buffer) > _MAX_LINE_BYTES:
                    self._send_error("line_too_long", f"frame exceeded {_MAX_LINE_BYTES} bytes")
                    break
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    self._handle_frame(line)
                    if not self.is_running:
                        return
        finally:
            try:
                sel.unregister(sock)
            except (KeyError, ValueError):
                pass
            sel.close()
            try:
                sock.close()
            except OSError:
                pass

    def _send_error(self, code: str, message: str) -> None:
        self._set_error(code, message)
        self._send_json({"type": "error", "code": code, "message": message})

    def _handle_frame(self, line: bytes) -> None:
        try:
            data = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_error("bad_json", f"could not parse frame: {exc}")
            return
        if not isinstance(data, dict):
            self._send_error("bad_frame", "frame must be a JSON object")
            return

        command = data.get("command")
        if command is not None:
            if command == "reset":
                if self.on_reset is not None:
                    try:
                        self.on_reset()
                    except Exception as exc:  # noqa: BLE001 - reported to the client
                        self._send_error("reset_failed", str(exc))
                        return
                self._drain_mailbox()
                # Wake an engine currently blocked waiting for the next action;
                # it must publish the reset state instead of applying a stale
                # pre-reset action.
                self._push(RESET_REQUESTED)
                self._send_json({"type": "ack", "command": "reset"})
            else:
                self._send_error("unknown_command", f"unsupported command {command!r}")
            return

        metrics = data.get("metrics")
        if "action" in data:
            self._push((data["action"], metrics))
        elif "anchor_mask" in data:
            mask = data["anchor_mask"]
            if not isinstance(mask, list):
                self._send_error("bad_frame", "anchor_mask must be a list")
                return
            self._push(([i for i, v in enumerate(mask) if v], metrics))
        elif metrics is None:
            self._send_error("bad_frame", "frame has no action, anchor_mask or command")

    # ── Public API ──────────────────────────────────────────────────────────

    def send_state(self, state_data: Any) -> bool:
        """Send one state (or a list of per-agent states) to the client."""
        if not self.is_running or self.client_socket is None:
            return False
        if isinstance(state_data, list):
            for item in state_data:
                if isinstance(item, dict):
                    item["protocol_version"] = PROTOCOL_VERSION
        elif isinstance(state_data, dict):
            state_data["protocol_version"] = PROTOCOL_VERSION
        return self._send_json(state_data)

    def wait_for_action(self, timeout: Optional[float] = None) -> Optional[ActionResult]:
        """Block for the client's reply.

        Returns an :class:`ActionResult`, or ``None`` when ``timeout`` elapses
        with no reply. ``reason`` is one of :data:`REASON_ACTION`,
        :data:`REASON_DISCONNECTED` or :data:`REASON_STOPPED`, so a caller never
        has to guess whether a bare ``None`` meant "not yet" or "peer gone".
        """
        if not self.is_running and self.client_socket is None:
            return ActionResult(None, None, REASON_STOPPED)

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
            else:
                remaining = 0.2
            try:
                item = self._mailbox.get(timeout=remaining)
            except queue.Empty:
                continue
            if item is DISCONNECTED:
                reason = REASON_STOPPED if self._stop_requested else REASON_DISCONNECTED
                return ActionResult(None, None, reason)
            if item is RESET_REQUESTED:
                return ActionResult(None, None, REASON_RESET)
            action, metrics = item
            return ActionResult(action, metrics, REASON_ACTION)
