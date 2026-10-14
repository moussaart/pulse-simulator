import json
import tempfile
import time
import urllib.error
import urllib.request

from src.api.ai_training_worker import TrainingWorker


def _worker(tmp_path, **kwargs):
    w = TrainingWorker(run_dir=tmp_path, serve_http=True, http_port=0, **kwargs)
    w.start_http()
    return w


def _post(port, command):
    req = urllib.request.Request(
        "http://127.0.0.1:%d/api/control" % port,
        data=json.dumps({"command": command}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _get(port, path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=2.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_a_supervisor_can_pause_and_resume_the_worker_over_http():
    with tempfile.TemporaryDirectory() as tmp:
        w = _worker(tmp)
        port = w.http.port
        try:
            status, body = _post(port, "pause")
            assert status == 200
            assert body["ok"] is True
            assert body["result"]["phase"] == "paused"
            assert w.status["phase"] == "paused"

            status, body = _post(port, "resume")
            assert status == 200
            assert body["result"]["phase"] == "running"
            assert w.status["phase"] == "running"
        finally:
            w.close()


def test_a_supervisor_can_stop_the_worker_over_http():
    with tempfile.TemporaryDirectory() as tmp:
        w = _worker(tmp)
        port = w.http.port
        try:
            status, body = _post(port, "stop")
            assert status == 200
            assert body["result"]["phase"] == "stopped"
            assert w.stopped is True
        finally:
            w.close()


def test_reset_over_http_restarts_the_simulation_and_the_step_counter():
    with tempfile.TemporaryDirectory() as tmp:
        w = _worker(tmp)
        port = w.http.port
        try:
            for _ in range(3):
                w.tick()
            assert w.status["step"] == 3
            assert w.recorder.status()["last_step"] == 2

            status, body = _post(port, "reset")
            assert status == 200
            assert body["result"]["step"] == 0
            assert w.status["step"] == 0
            assert w.recorder.status()["last_step"] == -1
        finally:
            w.close()


def test_pausing_actually_stops_the_run_in_thread_mode():
    """A paused worker must not advance, not merely report a different phase."""
    with tempfile.TemporaryDirectory() as tmp:
        w = _worker(tmp)
        try:
            w.pause()
            w.run_in_thread()
            time.sleep(0.25)
            assert w.status["step"] == 0
            assert w.recorder.status()["last_step"] == -1
        finally:
            w.close()


def test_resuming_after_a_stop_does_not_revive_the_worker():
    with tempfile.TemporaryDirectory() as tmp:
        w = _worker(tmp)
        try:
            w.stop()
            w.resume()
            assert w.status["phase"] == "stopped"
            assert w.stopped is True
        finally:
            w.close()


def test_the_worker_refuses_unknown_commands_without_leaking_state():
    with tempfile.TemporaryDirectory() as tmp:
        w = _worker(tmp)
        port = w.http.port
        try:
            status, body = _post(port, "shutdown-everything")
            assert status == 400
            assert body["ok"] is False
            assert "allowed" in body
            assert w.status["phase"] == "running"
        finally:
            w.close()


def test_status_over_http_agrees_with_the_worker_state():
    with tempfile.TemporaryDirectory() as tmp:
        w = _worker(tmp)
        port = w.http.port
        try:
            for _ in range(2):
                w.tick()
            remote = _get(port, "/api/status")
            local = w.recorder.status()
            assert remote["last_step"] == local["last_step"]
            assert remote["records"] == local["records"]
            assert remote["phase"] == w.status["phase"]
        finally:
            w.close()
