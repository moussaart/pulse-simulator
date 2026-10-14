"""End-to-end source-mode process supervision for the training GUI."""

import time

import pytest

from src.api.ai_training_engine import TrainingConfig
from src.api.worker_supervisor import WorkerStartupError, WorkerSupervisor


def test_supervisor_launches_worker_streams_frames_and_controls_lifecycle(tmp_path):
    config = TrainingConfig(
        algorithm="Trilateration",
        movement_pattern="Random Walk",
        measurement_source="uwb",
        num_agents=2,
    )
    supervisor = WorkerSupervisor(startup_timeout=20.0)
    try:
        endpoints = supervisor.start(
            config=config, run_dir=str(tmp_path / "run"), gym_port=0
        )
        assert endpoints["http_port"] > 0
        assert endpoints["frames_port"] > 0
        assert endpoints["gym_port"] > 0

        first = supervisor.wait_for_frame(timeout=15.0)
        assert first["type"] == "state"
        assert first["protocol_version"] == 3
        assert len(first["agents"]) == 2

        paused = supervisor.control("pause")
        assert paused["phase"] == "paused"
        paused_step = paused["step"]
        time.sleep(0.2)
        still_paused = supervisor.status()
        assert still_paused["phase"] == "paused"
        assert still_paused["last_step"] <= paused_step

        resumed = supervisor.control("resume")
        assert resumed["phase"] == "running"
        next_frame = supervisor.wait_for_frame(timeout=5.0)
        assert next_frame["step_id"] > first["step_id"]
    finally:
        supervisor.stop()
    assert supervisor.process is None or supervisor.process.poll() is not None


def test_supervisor_reports_worker_startup_failure_instead_of_timing_out(tmp_path):
    config = TrainingConfig(
        algorithm="not a registered algorithm",
        movement_pattern="Random Walk",
    )
    supervisor = WorkerSupervisor(startup_timeout=10.0)
    try:
        with pytest.raises(WorkerStartupError, match="algorithm|registered"):
            supervisor.start(config=config, run_dir=str(tmp_path / "bad-run"))
    finally:
        supervisor.stop()
