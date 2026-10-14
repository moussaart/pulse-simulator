"""Packaged builds must be able to run as the headless worker.

Two things have to hold:

* ``PULSE.exe --ai-training-worker`` starts the worker and never imports the Qt
  GUI, otherwise "Headless" mode silently still needs a display;
* ``multiprocessing.freeze_support()`` runs before anything that could spawn, or
  a spawned child re-executes the frozen GUI instead of a worker.
"""

import ast
import multiprocessing
import runpy
from pathlib import Path

import pytest

from src.api.ai_training_worker import (
    WORKER_FLAG,
    is_worker_invocation,
    strip_worker_flag,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"
HOOK_PY = PROJECT_ROOT / "src" / "api" / "pulse_freeze_hook.py"


def test_the_worker_flag_is_recognised():
    assert is_worker_invocation(["--ai-training-worker"])
    assert is_worker_invocation(["--ai-training-worker", "--max-steps", "10"])
    assert is_worker_invocation(["--http", "--ai-training-worker"])


def test_ordinary_invocations_are_left_alone():
    assert not is_worker_invocation([])
    assert not is_worker_invocation(["main.py"])
    assert not is_worker_invocation(["--ai-training", "--headless"])
    # A near-miss must not hijack the GUI.
    assert not is_worker_invocation(["--ai-training-worker=1"])


def test_the_worker_flag_is_removed_before_parsing():
    """argparse would reject it, so it must be stripped from the child argv."""
    assert strip_worker_flag(["--ai-training-worker", "--max-steps", "3"]) == ["--max-steps", "3"]
    assert strip_worker_flag(["--http", "--ai-training-worker"]) == ["--http"]
    assert strip_worker_flag([]) == []


def test_the_freeze_hook_installs_multiprocessing_support():
    assert HOOK_PY.exists(), "runtime hook referenced by main.spec must exist"
    source = HOOK_PY.read_text(encoding="utf-8")
    assert "freeze_support" in source


def test_the_freeze_hook_is_safe_to_run_unfrozen(monkeypatch):
    """The hook also runs in development, where freeze_support is a no-op."""
    calls = []
    monkeypatch.setattr(multiprocessing, "freeze_support", lambda: calls.append(1))
    monkeypatch.delattr("sys.frozen", raising=False)
    runpy.run_path(str(HOOK_PY), run_name="__hook__")
    assert calls == [], "freeze_support must only run in a frozen build"


def test_main_py_dispatches_the_worker_before_importing_qt():
    """Import order is the whole mechanism; assert it structurally."""
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))

    def line_of(pattern):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(pattern):
                    return node.lineno
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(pattern):
                        return node.lineno
        return None

    gui_line = line_of("src.app.localization_main")
    worker_line = line_of("src.api.ai_training_worker")
    assert gui_line is not None, "main.py must still launch the GUI by default"
    assert worker_line is not None, "main.py must expose worker mode"
    assert worker_line < gui_line, (
        "the worker must be dispatched before the Qt GUI is imported, "
        "or headless mode still needs a display"
    )


def test_main_py_calls_freeze_support_before_launching_anything():
    tree = ast.parse(MAIN_PY.read_text(encoding="utf-8"))
    freeze_lines, launch_lines = [], []
    for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "freeze_support":
                    freeze_lines.append(node.lineno)
            elif isinstance(node.func, ast.Name) and node.func.id in ("main", "worker_main"):
                launch_lines.append(node.lineno)
    assert freeze_lines, "main.py must call multiprocessing.freeze_support()"
    assert launch_lines, "main.py must still launch something"
    assert min(freeze_lines) < min(launch_lines), \
        "freeze_support must precede any launch so child processes re-exec safely"


def test_the_spec_registers_the_runtime_hook():
    spec = (PROJECT_ROOT / "main.spec").read_text(encoding="utf-8")
    assert "pulse_freeze_hook.py" in spec, (
        "main.spec must pass the hook via runtime_hooks; without it the frozen "
        "build cannot spawn the worker"
    )
