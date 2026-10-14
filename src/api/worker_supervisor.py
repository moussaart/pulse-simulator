"""Launch and supervise a Qt-free AI training worker process.

The GUI is a viewer/controller: it starts the worker, consumes its passive frame
stream, and uses the worker's loopback HTTP endpoint for lifecycle commands and
liveness. This module deliberately imports no Qt and is independently testable.
"""

from __future__ import annotations

import collections
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Sequence

from src.api.ai_training_engine import TrainingConfig
from src.api.ai_training_worker import WORKER_FLAG
from src.api.frame_publisher import PROTOCOL_VERSION as FRAME_PROTOCOL_VERSION

__all__ = ["WorkerStartupError", "WorkerSupervisor"]


class WorkerStartupError(RuntimeError):
    """The subprocess exited or failed to become ready."""


class WorkerSupervisor:
    """Own a worker subprocess and expose frames, status, and lifecycle control."""

    def __init__(
        self,
        command: Optional[Sequence[str]] = None,
        startup_timeout: float = 20.0,
        frame_queue_limit: int = 32,
    ) -> None:
        self.command = list(command) if command is not None else None
        self.startup_timeout = float(startup_timeout)
        self.frame_queue_limit = max(1, int(frame_queue_limit))
        self.process: Optional[subprocess.Popen] = None
        self.endpoints: Optional[Dict[str, Any]] = None
        self._temp_dir: Optional[str] = None
        self._frame_socket: Optional[socket.socket] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._reader_stop = threading.Event()
        self._frame_condition = threading.Condition()
        self._frames: Deque[Dict[str, Any]] = collections.deque()
        self._logs: Deque[str] = collections.deque(maxlen=200)
        self._stdout_thread: Optional[threading.Thread] = None
        self._closed = False

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def recent_logs(self):
        return list(self._logs)

    def _default_command(self) -> list:
        if getattr(sys, "frozen", False):
            return [sys.executable, WORKER_FLAG]
        root = Path(__file__).resolve().parents[2]
        return [sys.executable, str(root / "main.py"), WORKER_FLAG]

    def _command_for(
        self, config_path: str, run_dir: str, ready_path: str, gym_port: int
    ) -> list:
        base = list(self.command) if self.command is not None else self._default_command()
        return base + [
            "--config", config_path,
            "--run-dir", run_dir,
            "--http", "--port", "0",
            "--frames", "--frames-port", "0",
            "--gym", "--gym-port", str(int(gym_port)),
            "--ready-file", ready_path,
        ]

    def start(
        self,
        config: Optional[TrainingConfig] = None,
        run_dir: Optional[str] = None,
        gym_port: int = 5555,
    ) -> Dict[str, Any]:
        """Start worker, await its ready file, and connect to its frame stream."""
        if self.running:
            raise RuntimeError("worker is already running")
        self._closed = False
        self._reader_stop.clear()
        self._frames.clear()
        self._logs.clear()
        self._temp_dir = tempfile.mkdtemp(prefix="pulse-worker-supervisor-")
        cfg = config or TrainingConfig()
        config_path = os.path.join(self._temp_dir, "worker-config.json")
        ready_path = os.path.join(self._temp_dir, "worker-ready.json")
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(cfg.to_dict(), fh)
        chosen_run_dir = os.path.abspath(run_dir or os.path.join(self._temp_dir, "run"))
        os.makedirs(chosen_run_dir, exist_ok=True)

        kwargs = {
            "cwd": str(Path(__file__).resolve().parents[2]),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "text": True,
            "bufsize": 1,
        }
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.process = subprocess.Popen(
                self._command_for(config_path, chosen_run_dir, ready_path, gym_port), **kwargs
            )
            self._stdout_thread = threading.Thread(
                target=self._drain_stdout, name="pulse-worker-stdout", daemon=True
            )
            self._stdout_thread.start()
            self.endpoints = self._wait_ready(ready_path)
            self._connect_frames(int(self.endpoints["frames_port"]))
            return dict(self.endpoints)
        except Exception:
            self.stop()
            raise

    def _drain_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        try:
            for line in process.stdout:
                self._logs.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def _wait_ready(self, ready_path: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            process = self.process
            if process is None:
                raise WorkerStartupError("worker process was not created")
            code = process.poll()
            if code is not None:
                detail = "\n".join(self.recent_logs[-20:])
                raise WorkerStartupError(
                    "worker exited during startup (code %s)%s"
                    % (code, ("\n" + detail) if detail else "")
                )
            try:
                with open(ready_path, "r", encoding="utf-8") as fh:
                    ready = json.load(fh)
                if not isinstance(ready, dict):
                    raise ValueError("ready file is not a JSON object")
                if not isinstance(ready.get("http_port"), int):
                    raise ValueError("worker did not report an HTTP control port")
                if not isinstance(ready.get("frames_port"), int):
                    raise ValueError("worker did not report a frame port")
                return ready
            except FileNotFoundError:
                pass
            except (ValueError, json.JSONDecodeError) as exc:
                raise WorkerStartupError("invalid worker ready file: %s" % exc) from exc
            time.sleep(0.025)
        detail = "\n".join(self.recent_logs[-20:])
        raise WorkerStartupError(
            "worker did not become ready within %.1fs%s"
            % (self.startup_timeout, ("\n" + detail) if detail else "")
        )

    def _connect_frames(self, port: int) -> None:
        sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        sock.settimeout(0.5)
        sock.sendall(b'{"command":"subscribe","kind":"frame"}\n')
        ready = self._recv_json_line(sock, timeout=5.0)
        if ready.get("type") != "ready":
            sock.close()
            raise WorkerStartupError("frame endpoint rejected subscription: %r" % ready)
        if ready.get("protocol_version") != FRAME_PROTOCOL_VERSION:
            sock.close()
            raise WorkerStartupError("unsupported frame protocol: %r" % ready)
        self._frame_socket = sock
        self._reader_thread = threading.Thread(
            target=self._read_frames, name="pulse-worker-frames", daemon=True
        )
        self._reader_thread.start()

    @staticmethod
    def _recv_json_line(sock: socket.socket, timeout: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        buf = b""
        while time.monotonic() < deadline:
            sock.settimeout(max(0.01, min(0.5, deadline - time.monotonic())))
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise WorkerStartupError("worker closed the frame stream")
            buf += chunk
            if b"\n" not in buf:
                continue
            line, _ = buf.split(b"\n", 1)
            result = json.loads(line.decode("utf-8"))
            if not isinstance(result, dict):
                raise WorkerStartupError("frame stream sent a non-object")
            return result
        raise WorkerStartupError("timed out during frame subscription")

    def _read_frames(self) -> None:
        sock = self._frame_socket
        if sock is None:
            return
        buf = b""
        while not self._reader_stop.is_set():
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line:
                    continue
                try:
                    frame = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._logs.append("[supervisor] malformed frame discarded")
                    continue
                if not isinstance(frame, dict) or frame.get("type") != "state":
                    continue
                with self._frame_condition:
                    if len(self._frames) >= self.frame_queue_limit:
                        self._frames.popleft()
                    self._frames.append(frame)
                    self._frame_condition.notify_all()

    def wait_for_frame(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for the next available frame, or raise ``TimeoutError``."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        with self._frame_condition:
            while not self._frames:
                if self.process is not None and self.process.poll() is not None:
                    raise WorkerStartupError("worker exited before another frame arrived")
                if deadline is None:
                    self._frame_condition.wait(0.5)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for worker frame")
                    self._frame_condition.wait(min(remaining, 0.5))
            return self._frames.popleft()

    def drain_frames(self) -> list:
        """Return and clear all queued frames; suited to a GUI timer callback."""
        with self._frame_condition:
            result = list(self._frames)
            self._frames.clear()
            return result

    def status(self) -> Dict[str, Any]:
        if self.endpoints is None:
            raise RuntimeError("worker has not started")
        url = "http://127.0.0.1:%d/api/status" % self.endpoints["http_port"]
        with urllib.request.urlopen(url, timeout=3.0) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("worker returned malformed status")
        return value

    def control(self, command: str) -> Dict[str, Any]:
        if command not in ("pause", "resume", "reset", "stop"):
            raise ValueError("unsupported worker command: %r" % command)
        if self.endpoints is None:
            raise RuntimeError("worker has not started")
        url = "http://127.0.0.1:%d/api/control" % self.endpoints["http_port"]
        request = urllib.request.Request(
            url,
            data=json.dumps({"command": command}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            raise RuntimeError(payload.get("error", "worker control failed")) from exc
        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "worker control failed"))
        return payload.get("result", {})

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the worker to stop, then terminate only if graceful shutdown fails."""
        if self._closed and self.process is None:
            return
        self._closed = True
        process = self.process
        if process is not None and process.poll() is None and self.endpoints is not None:
            try:
                self.control("stop")
            except Exception:
                pass
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        self._reader_stop.set()
        sock, self._frame_socket = self._frame_socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        self.process = None
        self.endpoints = None
        if self._temp_dir is not None:
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
