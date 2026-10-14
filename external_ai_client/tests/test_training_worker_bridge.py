"""Exercise the shipped Gym client against the new worker-owned server."""

import threading
import time

import numpy as np

from pulse_ai_client import PulseRLEnv
from src.api.ai_training_engine import TrainingConfig
from src.api.ai_training_worker import TrainingWorker


def test_pulse_rl_env_round_trips_through_worker_and_resets(tmp_path):
    worker = TrainingWorker(
        config=TrainingConfig(
            algorithm="Duty-Cycled UWB-IMU NA-AEKF",
            movement_pattern="Random Walk",
            measurement_source="both",
            num_agents=1,
        ),
        run_dir=str(tmp_path),
        serve_http=True,
        http_port=0,
        serve_gym=True,
        gym_port=0,
    )
    endpoints = worker.start_services()
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    env = PulseRLEnv(
        host="127.0.0.1", port=endpoints["gym_port"],
        num_anchors=4, num_agents=1,
    )
    try:
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert info["states"][0].protocol_version == 3

        obs, reward, terminated, truncated, info = env.step(np.array([1, 1, 1, 0]))
        assert obs.shape == env.observation_space.shape
        assert not terminated and not truncated
        assert info["states"][0].step_id >= 1

        reset_obs, reset_info = env.reset(seed=17)
        assert reset_obs.shape == env.observation_space.shape
        assert reset_info["states"][0].step_id == 0
    finally:
        env.close()
        worker.stop()
        worker.close()
        thread.join(timeout=3.0)
    assert not thread.is_alive()
