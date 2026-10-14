import json
import os
import tempfile
import urllib.request

from src.api.training_metrics import MetricsHttpServer, MetricsRecorder


def test_recorder_appends_jsonl_and_keeps_bounded_tail():
    with tempfile.TemporaryDirectory() as tmp:
        rec = MetricsRecorder(run_dir=tmp, tail_limit=5)
        for i in range(20):
            rec.record({"kind": "step", "step": i, "reward": float(i)})
        rec.close()

        path = os.path.join(tmp, "metrics.jsonl")
        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert len(lines) == 20
        assert lines[0]["step"] == 0
        assert lines[-1]["step"] == 19

        tail = rec.tail()
        assert len(tail) == 5
        assert tail[-1]["step"] == 19


def test_recorder_writes_status_file():
    with tempfile.TemporaryDirectory() as tmp:
        rec = MetricsRecorder(run_dir=tmp, tail_limit=5)
        rec.record({"kind": "step", "step": 7, "reward": 1.0})
        rec.close()

        with open(os.path.join(tmp, "status.json"), "r", encoding="utf-8") as fh:
            status = json.load(fh)
        assert status["last_step"] == 7
        assert status["pid"] == os.getpid()


def test_dashboard_serves_html_and_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        rec = MetricsRecorder(run_dir=tmp, tail_limit=10)
        rec.record({"kind": "step", "step": 1, "reward": 0.5, "error": 0.1, "energy": 2.0})

        server = MetricsHttpServer(rec, host="127.0.0.1", port=0)
        server.start()
        try:
            base = f"http://127.0.0.1:{server.port}"
            with urllib.request.urlopen(base + "/", timeout=5) as resp:
                html = resp.read().decode("utf-8")
            assert "PULSE" in html

            with urllib.request.urlopen(base + "/api/metrics", timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            assert payload["tail"][-1]["step"] == 1

            with urllib.request.urlopen(base + "/api/status", timeout=5) as resp:
                status = json.loads(resp.read().decode("utf-8"))
            assert status["last_step"] == 1
        finally:
            server.stop()
            rec.close()
