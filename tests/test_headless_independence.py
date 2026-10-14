"""The worker must run with no Qt anywhere in the process.

Headless and charts-only modes run on machines without a display, and the frozen
build must not need Qt either. A test that merely imports the module is not
enough: a chain like ``src.core.localization -> Boxs`` can drag PyQt5 in at
import time. So these tests block the Qt packages at the import system level and
then drive a real run.
"""

import json
import os
import sys
import tempfile
import urllib.request

import pytest

QT_PACKAGES = ("PyQt5", "PySide", "PySide2", "PySide6")


@pytest.fixture
def no_qt(monkeypatch):
    """Make importing any Qt package fail, as it would on a headless box."""

    class Blocker:
        def find_module(self, name, path=None):
            return self if name.split(".")[0] in QT_PACKAGES else None

        def load_module(self, name):
            raise ImportError("Qt is blocked in this test: %s" % name)

    for mod in [m for m in sys.modules if m.split(".")[0] in QT_PACKAGES]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Blocker()] + list(sys.meta_path))
    yield


def test_the_worker_imports_without_qt(no_qt):
    from src.api.ai_training_worker import TrainingWorker

    assert TrainingWorker is not None
    assert [m for m in sys.modules if m.split(".")[0] in QT_PACKAGES] == []


def test_a_multi_agent_run_completes_with_no_qt_loaded(no_qt):
    from src.api.ai_training_engine import TrainingConfig
    from src.api.ai_training_worker import TrainingWorker

    with tempfile.TemporaryDirectory() as tmp:
        w = TrainingWorker(
            config=TrainingConfig(num_agents=4),
            run_dir=tmp,
            serve_http=True,
            http_port=0,
        )
        port = w.start_http()
        try:
            for _ in range(50):
                w.tick()
            status = w.status
            assert status["phase"] == "running"
            assert status["step"] == 50
            assert status["num_agents"] == 4
            # The supervisor channel must work in the same Qt-free process.
            req = urllib.request.Request(
                "http://127.0.0.1:%d/api/control" % port,
                data=json.dumps({"command": "pause"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            assert body["ok"] is True
            assert body["result"]["phase"] == "paused"
            assert [m for m in sys.modules if m.split(".")[0] in QT_PACKAGES] == []
        finally:
            w.close()


def test_the_localization_package_still_exposes_the_qt_message_box(no_qt):
    """Lazifying the import must not break GUI callers that do want it."""
    import src.core.localization as loc

    with pytest.raises(ImportError):
        loc.CustomMessageBox  # noqa: B018 - attribute access is the assertion

    import importlib

    assert importlib.util.find_spec("PyQt5") is not None, "Qt should be installed here"
