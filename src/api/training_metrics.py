"""Append-only training metrics sink plus a loopback-only HTTP dashboard.

The recorder is the source of truth for the independent dashboard: it survives
the PULSE GUI being frozen or closed, because it is owned by the training
backend process rather than by the Qt event loop.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

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


class MetricsRecorder:
    """Append-only JSONL sink with a bounded in-memory tail.

    One record is one line, flushed immediately, so an external tailer (and a
    human with a text editor) can follow a live 24-hour run.
    """

    def __init__(self, run_dir: str, tail_limit: int = 5000):
        self.run_dir = str(run_dir)
        self.tail_limit = int(tail_limit)
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.status_path = os.path.join(self.run_dir, "status.json")
        self._lock = threading.Lock()
        self._tail: deque = deque(maxlen=self.tail_limit)
        self._fh = open(self.metrics_path, "a", encoding="utf-8", buffering=1)
        self._count = 0
        self._last_step = -1
        self._sim_time = 0.0
        self._phase = "starting"
        self._closed = False
        self._started_wall = time.time()
        self._write_status()

    # ── writing ────────────────────────────────────────────────────────────

    def record(self, payload: Dict[str, Any]) -> None:
        """Append one metrics record and refresh status."""
        with self._lock:
            if self._closed:
                return
            payload = dict(payload)
            payload.setdefault("wall_time", time.time())
            self._fh.write(json.dumps(payload, default=float) + "\n")
            self._fh.flush()
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
            self._phase = "running"
            self._write_status()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)
            self._write_status()

    def _write_status(self) -> None:
        status = {
            "pid": os.getpid(),
            "phase": self._phase,
            "last_step": self._last_step,
            "sim_time": self._sim_time,
            "records": self._count,
            "last_wall_time": time.time(),
            "started_wall_time": self._started_wall,
            "metrics_path": self.metrics_path,
        }
        tmp = self.status_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(status, fh, indent=2)
            os.replace(tmp, self.status_path)
        except OSError:
            pass

    # ── reading ────────────────────────────────────────────────────────────

    def tail(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._tail)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "pid": os.getpid(),
                "phase": self._phase,
                "last_step": self._last_step,
                "sim_time": self._sim_time,
                "records": self._count,
                "last_wall_time": time.time(),
                "started_wall_time": self._started_wall,
                "metrics_path": self.metrics_path,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._phase = "stopped"
            self._write_status()
            self._closed = True
            try:
                self._fh.close()
            except OSError:
                pass


class MetricsHttpServer:
    """Loopback-only dashboard server, so the charts outlive the PULSE GUI."""

    def __init__(self, recorder: MetricsRecorder, host: str = "127.0.0.1", port: int = 8765):
        self.recorder = recorder
        self.host = host
        self.requested_port = int(port)
        self.port = int(port)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

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

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0].rstrip("/") or "/"
                if path == "/":
                    self._send(200, DASHBOARD_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/metrics":
                    body = json.dumps({"tail": recorder.tail()}, default=float).encode("utf-8")
                    self._send(200, body, "application/json")
                elif path == "/api/status":
                    body = json.dumps(recorder.status(), default=float).encode("utf-8")
                    self._send(200, body, "application/json")
                else:
                    self._send(404, b"not found", "text/plain; charset=utf-8")

            def log_message(self, *args) -> None:  # silence per-request stderr spam
                return

        self._httpd = ThreadingHTTPServer((self.host, self.requested_port), Handler)
        self._httpd.daemon_threads = True
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
