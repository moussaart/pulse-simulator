"""Append-only training metrics sink plus a loopback-only HTTP dashboard.

The recorder is the source of truth for the independent dashboard: it survives
the PULSE GUI being frozen or closed, because it is owned by the training
backend process rather than by the Qt event loop.
"""

from __future__ import annotations

import glob
import json
import math
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PULSE Training Monitor</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#14161a; color:#e6e6e6;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; }
  header { padding:16px 24px; border-bottom:1px solid #2a2e35; display:flex;
           align-items:baseline; gap:16px; flex-wrap:wrap; }
  h1 { font-size:1.125rem; margin:0; font-weight:600; letter-spacing:.01em; }
  .status { font-size:.8125rem; color:#9aa4b2; }
  .status b { color:#e6e6e6; font-weight:600; }
  .grid { display:grid; gap:16px; padding:24px;
          grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); }
  .card { background:#1b1e24; border:1px solid #2a2e35; border-radius:10px; padding:16px; }
  .card h2 { font-size:.875rem; margin:0 0 12px; font-weight:600; color:#c8ced8; }
  canvas { width:100%; height:220px; display:block; }
  .legend { font-size:.75rem; color:#8b95a3; margin-top:8px; }
  .stale { color:#f0a35e; }
  .dead  { color:#e06c6c; }
</style>
</head>
<body>
<header>
  <h1>PULSE Training Monitor</h1>
  <div class="status" id="status">connecting&hellip;</div>
</header>
<div class="grid">
  <div class="card"><h2>Reward</h2><canvas id="reward"></canvas>
    <div class="legend">per-agent step reward and cumulative reward</div></div>
  <div class="card"><h2>Localization error</h2><canvas id="error"></canvas>
    <div class="legend">per-agent mean position error (m)</div></div>
  <div class="card"><h2>Energy</h2><canvas id="energy"></canvas>
    <div class="legend">per-agent step energy and cumulative energy (mJ)</div></div>
</div>
<script>
const SERIES = [
  { id: "reward", x: "time", fields: ["reward_step", "reward_cumulative"] },
  { id: "error",  x: "time", fields: ["error"] },
  { id: "energy", x: "time", fields: ["step_energy_mJ", "cumulative_energy_mJ"] }
];
const PALETTE = ["#4c9aff", "#4ec9a0", "#f0a35e", "#c78bf0", "#e06c6c", "#8ed14f"];
let last = null;

function draw(canvasId, series) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !last || !last.length) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = 220;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const xs = last.map(r => r[series.x] ?? 0);
  let lo = Infinity, hi = -Infinity;
  const perAgent = {};
  for (const f of series.fields) {
    perAgent[f] = last.map(r => Array.isArray(r[f]) ? r[f] : [r[f]]);
    for (const arr of perAgent[f]) for (const v of arr) {
      if (typeof v === "number" && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
  }
  if (!isFinite(lo) || !isFinite(hi)) return;
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;

  const x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
  const sx = v => (x1 - x0 < 1e-9) ? 0 : (v - x0) / (x1 - x0) * (w - 8) + 4;
  const sy = v => h - 8 - (v - lo) / (hi - lo) * (h - 24);

  ctx.strokeStyle = "#2a2e35"; ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = 8 + (h - 24) * i / 3;
    ctx.beginPath(); ctx.moveTo(4, y); ctx.lineTo(w - 4, y); ctx.stroke();
  }

  for (const f of series.fields) {
    perAgent[f].forEach((arr, ai) => {
      ctx.strokeStyle = PALETTE[(ai + (f === series.fields[1] ? 3 : 0)) % PALETTE.length];
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      let started = false;
      arr.forEach((v, i) => {
        if (typeof v !== "number" || !isFinite(v)) { started = false; return; }
        const X = sx(xs[i]), Y = sy(v);
        if (!started) { ctx.moveTo(X, Y); started = true; } else { ctx.lineTo(X, Y); }
      });
      ctx.stroke();
    });
  }
}

async function refresh() {
  try {
    const [m, s] = await Promise.all([
      fetch("/api/metrics", { cache: "no-store" }).then(r => r.json()),
      fetch("/api/status", { cache: "no-store" }).then(r => r.json())
    ]);
    last = m.tail;
    const el = document.getElementById("status");
    const age = Math.max(0, Date.now() / 1000 - (s.last_wall_time || 0));
    el.className = "status" + (s.phase !== "running" ? " dead" : age > 15 ? " stale" : "");
    el.innerHTML = "phase <b>" + (s.phase || "?") + "</b> &middot; step <b>" +
      (s.last_step ?? 0) + "</b> &middot; sim t <b>" + ((s.sim_time ?? 0).toFixed(2)) +
      "s</b> &middot; last update <b>" + age.toFixed(1) + "s</b> ago";
    for (const s2 of SERIES) draw(s2.id, s2);
  } catch (e) {
    const el = document.getElementById("status");
    el.className = "status dead";
    el.textContent = "backend unreachable";
  }
}
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


def _sanitize(value: Any) -> Tuple[Any, bool]:
    """Replace NaN/Inf with None (JSON has no NaN). Returns (clean, had_non_finite)."""
    if isinstance(value, float):
        if math.isfinite(value):
            return value, False
        return None, True
    if isinstance(value, dict):
        bad = False
        out = {}
        for key, item in value.items():
            out[key], flag = _sanitize(item)
            bad = bad or flag
        return out, bad
    if isinstance(value, (list, tuple)):
        bad = False
        out_list = []
        for item in value:
            clean, flag = _sanitize(item)
            out_list.append(clean)
            bad = bad or flag
        return out_list, bad
    return value, False


class MetricsRecorder:
    """Append-only JSONL sink with a bounded in-memory tail.

    One record is one line, flushed immediately, so an external tailer (and a
    human with a text editor) can follow a live 24-hour run.

    Long-run guarantees:

    * ``status.json`` is rewritten at most every ``status_interval`` seconds
      (phase changes, resets and close are written immediately). Rewriting it
      per record was the dominant I/O cost, and on Windows ``os.replace`` fails
      whenever a reader holds the file open.
    * When ``metrics.jsonl`` exceeds ``max_bytes`` it is renamed to
      ``metrics.NNNNNN.jsonl`` and a fresh ``metrics.jsonl`` is opened, so the
      live path is stable for tailers and no record is ever dropped.
    * Write failures never propagate into the simulation loop, but they are
      counted and reported in the status (``io_errors``/``last_io_error``).
    * NaN/Inf values are written as ``null`` and counted
      (``non_finite_records``) so numerical faults are visible, not hidden.
    """

    DEFAULT_MAX_BYTES = 256 * 1024 * 1024

    def __init__(self, run_dir: str, tail_limit: int = 5000,
                 status_interval: float = 1.0,
                 max_bytes: Optional[int] = DEFAULT_MAX_BYTES):
        self.run_dir = str(run_dir)
        self.tail_limit = int(tail_limit)
        self.status_interval = float(status_interval)
        self.max_bytes = int(max_bytes) if max_bytes else None
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.status_path = os.path.join(self.run_dir, "status.json")
        self._lock = threading.Lock()
        self._tail: deque = deque(maxlen=self.tail_limit)
        # Continue numbering after any segments left by a previous (resumed) run.
        self._segment_index = self._existing_segment_count()
        self._fh = open(self.metrics_path, "a", encoding="utf-8", buffering=1)
        self._count = 0
        self._last_step = -1
        self._sim_time = 0.0
        self._phase = "starting"
        self._closed = False
        self._io_errors = 0
        self._last_io_error: Optional[str] = None
        self._status_errors = 0
        self._non_finite = 0
        self._started_wall = time.time()
        self._last_status_write = 0.0
        self._write_status()

    def _existing_segment_count(self) -> int:
        return len(glob.glob(os.path.join(self.run_dir, "metrics.[0-9]*.jsonl")))

    # ── writing ────────────────────────────────────────────────────────────

    def record(self, payload: Dict[str, Any]) -> None:
        """Append one metrics record and refresh status (throttled)."""
        with self._lock:
            if self._closed:
                return
            payload, had_non_finite = _sanitize(dict(payload))
            if had_non_finite:
                self._non_finite += 1
            payload.setdefault("wall_time", time.time())
            try:
                self._fh.write(json.dumps(payload, default=float) + "\n")
                self._fh.flush()
                self._maybe_rotate()
            except (OSError, ValueError) as exc:
                self._io_errors += 1
                self._last_io_error = "%s: %s" % (type(exc).__name__, exc)
            self._tail.append(payload)
            self._count += 1
            if "step" in payload:
                try:
                    self._last_step = int(payload["step"])
                except (TypeError, ValueError):
                    pass
            if "time" in payload:
                try:
                    self._sim_time = float(payload["time"])
                except (TypeError, ValueError):
                    pass
            phase_changed = self._phase != "running"
            self._phase = "running"
            if phase_changed or (time.time() - self._last_status_write) >= self.status_interval:
                self._write_status()

    def _maybe_rotate(self) -> None:
        if self.max_bytes is None or self._fh.tell() < self.max_bytes:
            return
        self._fh.close()
        self._segment_index += 1
        target = os.path.join(self.run_dir, "metrics.%06d.jsonl" % self._segment_index)
        try:
            os.replace(self.metrics_path, target)
        finally:
            # Always reopen, even if the rename failed, so recording continues.
            self._fh = open(self.metrics_path, "a", encoding="utf-8", buffering=1)

    def heartbeat(self) -> None:
        """Refresh ``status.json`` (throttled) when no records are flowing, e.g. paused."""
        with self._lock:
            if self._closed:
                return
            if (time.time() - self._last_status_write) >= self.status_interval:
                self._write_status()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)
            self._write_status()

    def mark_reset(self) -> None:
        """Clear the progress counters after the simulation restarts.

        Without this the status file keeps reporting the step and sim time of the
        run that just ended, so a supervisor would poll a stale step forever.
        """
        with self._lock:
            self._last_step = -1
            self._sim_time = 0.0
            self._write_status()

    def _status_dict(self) -> Dict[str, Any]:
        return {
            "pid": os.getpid(),
            "phase": self._phase,
            "last_step": self._last_step,
            "sim_time": self._sim_time,
            "records": self._count,
            "last_wall_time": time.time(),
            "started_wall_time": self._started_wall,
            "metrics_path": self.metrics_path,
            "segments": self._segment_index + 1,
            "io_errors": self._io_errors,
            "last_io_error": self._last_io_error,
            "status_write_errors": self._status_errors,
            "non_finite_records": self._non_finite,
        }

    def _write_status(self) -> None:
        status = self._status_dict()
        self._last_status_write = time.time()
        tmp = self.status_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(status, fh, indent=2)
            os.replace(tmp, self.status_path)
        except OSError:
            # Typically a reader holding status.json on Windows; the next
            # interval retries. Counted so a permanently stale file is visible.
            self._status_errors += 1

    # ── reading ────────────────────────────────────────────────────────────

    def tail(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._tail)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return self._status_dict()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            # A failed run must stay visibly failed after exit; only a healthy
            # run is closed as "stopped".
            if self._phase != "failed":
                self._phase = "stopped"
            self._write_status()
            self._closed = True
            try:
                self._fh.close()
            except OSError:
                pass


class MetricsHttpServer:
    """Loopback-only dashboard server, so the charts outlive the PULSE GUI.

    It also carries the control channel (``POST /api/control``) a supervising GUI
    uses to pause, resume, reset or stop the worker. Control and dashboards are
    separable: a charts-only or headless run does not need the browser dashboard
    served at all, so it can be turned off with :meth:`disable_dashboard`.
    """

    #: Commands a client may send. An allowlist, because the handler runs inside
    #: the training process and an arbitrary command name must never reach it.
    CONTROL_COMMANDS = ("pause", "resume", "reset", "stop")

    def __init__(
        self,
        recorder: MetricsRecorder,
        host: str = "127.0.0.1",
        port: int = 8765,
        control_handler: Optional[Callable[[str], Any]] = None,
    ):
        self.recorder = recorder
        self.host = host
        self.requested_port = int(port)
        self.port = int(port)
        self.control_handler = control_handler
        self._dashboard_enabled = True
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def disable_dashboard(self) -> None:
        """Stop serving the HTML dashboard and ``/api/metrics``.

        ``/api/status`` and ``/api/control`` stay available: the first is a cheap
        liveness probe, the second is how a supervisor drives the run.
        """
        self._dashboard_enabled = False

    def start(self) -> int:
        recorder = self.recorder

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass

            def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
                body = json.dumps(payload, default=float).encode("utf-8")
                self._send(code, body, "application/json")

            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(length) if length > 0 else b""

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0].rstrip("/") or "/"
                # Read through the owner so disable_dashboard() takes effect on
                # the already-serving socket instead of only on the next start().
                enabled = self.server.owner._dashboard_enabled
                if path == "/":
                    if not enabled:
                        self._send(404, b"not found", "text/plain; charset=utf-8")
                    else:
                        self._send(200, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/metrics":
                    if not enabled:
                        self._send(404, b"not found", "text/plain; charset=utf-8")
                    else:
                        body = json.dumps({"tail": recorder.tail()}, default=float).encode("utf-8")
                        self._send(200, body, "application/json")
                elif path == "/api/status":
                    body = json.dumps(recorder.status(), default=float).encode("utf-8")
                    self._send(200, body, "application/json")
                else:
                    self._send(404, b"not found", "text/plain; charset=utf-8")

            def do_POST(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0].rstrip("/") or "/"
                if path != "/api/control":
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return

                handler = self.server.owner.control_handler
                if handler is None:
                    # Read the body first: leaving it unread desynchronises the
                    # keep-alive connection instead of returning a clean error.
                    self._read_body()
                    self._send_json(503, {"ok": False, "error": "control channel unavailable"})
                    return

                raw = self._read_body()
                try:
                    payload = json.loads(raw.decode("utf-8") or "{}")
                except (UnicodeDecodeError, ValueError):
                    self._send_json(400, {"ok": False, "error": "body must be JSON"})
                    return
                if not isinstance(payload, dict):
                    self._send_json(400, {"ok": False, "error": "body must be a JSON object"})
                    return

                command = payload.get("command")
                if not isinstance(command, str) or command not in self.server.control_commands:
                    self._send_json(
                        400,
                        {
                            "ok": False,
                            "error": "unknown command",
                            "allowed": list(self.server.control_commands),
                        },
                    )
                    return

                try:
                    result = handler(command)
                except Exception as exc:  # a bad command must not kill the worker
                    self._send_json(500, {"ok": False, "error": str(exc)})
                    return
                self._send_json(200, {"ok": True, "result": result})

            def log_message(self, *args) -> None:  # silence per-request stderr spam
                return

        self._httpd = ThreadingHTTPServer((self.host, self.requested_port), Handler)
        self._httpd.daemon_threads = True
        # The handler reaches the live server state through this back-reference
        # rather than copying it, so later mutations are picked up immediately.
        self._httpd.owner = self
        self._httpd.control_commands = self.CONTROL_COMMANDS
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            try:
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
