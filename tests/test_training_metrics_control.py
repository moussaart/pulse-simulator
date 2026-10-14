import json
import tempfile
import urllib.error
import urllib.request

from src.api.training_metrics import MetricsHttpServer, MetricsRecorder


def _post(port, path, payload):
    """POST and return (status, body) even for error responses.

    urlopen raises HTTPError on 4xx/5xx, which would hide the status code the
    assertions are about.
    """
    req = urllib.request.Request(
        "http://127.0.0.1:%d%s" % (port, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_a_control_command_reaches_the_worker_callback():
    with tempfile.TemporaryDirectory() as tmp:
        seen = []
        rec = MetricsRecorder(run_dir=tmp)
        server = MetricsHttpServer(rec, port=0, control_handler=seen.append)
        port = server.start()
        try:
            status, body = _post(port, "/api/control", {"command": "pause"})
            assert status == 200
            assert body["ok"] is True
            assert seen == ["pause"]
        finally:
            server.stop()
            rec.close()


def test_an_unknown_command_is_rejected_without_calling_the_handler():
    with tempfile.TemporaryDirectory() as tmp:
        seen = []
        rec = MetricsRecorder(run_dir=tmp)
        server = MetricsHttpServer(rec, port=0, control_handler=seen.append)
        port = server.start()
        try:
            status, body = _post(port, "/api/control", {"command": "explode"})
            assert status == 400
            assert body["ok"] is False
            assert seen == []
        finally:
            server.stop()
            rec.close()


def test_a_missing_command_field_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        seen = []
        rec = MetricsRecorder(run_dir=tmp)
        server = MetricsHttpServer(rec, port=0, control_handler=seen.append)
        port = server.start()
        try:
            status, body = _post(port, "/api/control", {})
            assert status == 400
            assert body["ok"] is False
            assert seen == []
        finally:
            server.stop()
            rec.close()


def test_the_control_channel_is_refused_when_no_handler_is_wired():
    with tempfile.TemporaryDirectory() as tmp:
        rec = MetricsRecorder(run_dir=tmp)
        server = MetricsHttpServer(rec, port=0)
        port = server.start()
        try:
            status, body = _post(port, "/api/control", {"command": "pause"})
            assert status == 503
            assert body["ok"] is False
        finally:
            server.stop()
            rec.close()


def test_disabling_the_dashboard_hides_the_browser_only_endpoints():
    with tempfile.TemporaryDirectory() as tmp:
        rec = MetricsRecorder(run_dir=tmp)
        rec.record({"kind": "step", "step": 0})
        server = MetricsHttpServer(rec, port=0, control_handler=lambda c: None)
        server.start()
        port = server.port
        try:
            server.disable_dashboard()
            # The HTML page and the metrics tail are only useful to a browser.
            for path in ("/", "/api/metrics"):
                try:
                    urllib.request.urlopen(
                        "http://127.0.0.1:%d%s" % (port, path), timeout=2.0
                    )
                except urllib.error.HTTPError as exc:
                    assert exc.code == 404, path
                else:
                    raise AssertionError("%s should be hidden" % path)
            # /api/status stays: supervisors need a cheap liveness probe.
            with urllib.request.urlopen(
                "http://127.0.0.1:%d/api/status" % port, timeout=2.0
            ) as resp:
                assert resp.status == 200
        finally:
            server.stop()
            rec.close()
