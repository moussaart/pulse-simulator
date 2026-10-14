import numpy as np
import pytest

from src.core.uwb.channel_model import ChannelConditions
from src.core.uwb.uwb_devices import Anchor, Position, Tag
from src.api.ai_training_engine import (
    AITrainingEngine,
    TrainingConfig,
    UnknownAlgorithmError,
)


def _anchors():
    return [
        Anchor(Position(0.0, 0.0, 0.0)),
        Anchor(Position(5.0, 0.0, 0.0)),
        Anchor(Position(0.0, 5.0, 0.0)),
        Anchor(Position(5.0, 5.0, 0.0)),
    ]


def _config(**overrides):
    base = dict(
        algorithm="Duty-Cycled UWB-IMU NA-AEKF",
        dt=0.005,
        movement_pattern="Random Walk",
        movement_speed=1.0,
        measurement_source="both",
        num_agents=1,
        allow_gpu=False,
        # Off the anchors: starting at (0,0) coincides with an anchor, giving a
        # true distance of 0 that noise can legitimately push negative.
        start_position=(1.0, 1.0),
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _engine(**overrides):
    return AITrainingEngine(_config(**overrides), anchors=_anchors())


def _uwb_action(n_anchors=4):
    return {"filter": "Trilateration", "anchors": list(range(n_anchors)),
            "measurement_source": "uwb"}


def test_unknown_algorithm_raises_instead_of_silently_falling_back():
    engine = _engine()
    with pytest.raises(UnknownAlgorithmError) as exc:
        engine.resolve_algorithm("Duty-Cycled IMU-UWB Adaptive EKF")
    assert "Duty-Cycled IMU-UWB Adaptive EKF" in str(exc.value)


def test_the_name_the_shipped_wrapper_requests_is_not_registered():
    """env_wrapper.py asks for a name that does not exist; it must fail loudly."""
    engine = _engine()
    assert "Duty-Cycled IMU-UWB Adaptive EKF" not in engine.registered_algorithms()
    assert "Duty-Cycled UWB-IMU NA-AEKF" in engine.registered_algorithms()


def test_every_registered_algorithm_resolves():
    engine = _engine()
    names = engine.registered_algorithms()
    assert names, "algorithm registry is empty"
    for name in names:
        assert engine.resolve_algorithm(name) == name


def test_resolution_is_whitespace_and_case_insensitive_only():
    engine = _engine()
    assert engine.resolve_algorithm("  duty-cycled uwb-imu na-aekf ") == "Duty-Cycled UWB-IMU NA-AEKF"


def test_sim_time_advances_before_motion_like_the_main_manager():
    engine = _engine()
    engine.step([_uwb_action()])
    # main: simulation_time += dt, then update_tag_position(t=simulation_time)
    assert engine.agents[0].sim_time == pytest.approx(engine.dt)
    assert engine.agents[0].sim_time == pytest.approx(0.005)


def test_exact_mode_is_forwarded_to_motion():
    engine = _engine(exact_trajectory_mode=True)
    assert engine.agents[0].exact_mode is True
    engine.step([_uwb_action()])


def test_measurements_prepare_channel_state_per_anchor():
    engine = _engine()
    step = engine.step([_uwb_action()])
    records = step["agents"][0]["measurements"]
    assert len(records) == 4
    for record in records:
        assert record["distance"] > 0
        assert np.isfinite(record["true_distance"])
        assert isinstance(record["is_los"], bool)
    # Sequential path leaves the channel parked on the last anchor, exactly as
    # SimulationManager._measure_single_anchor does.
    assert engine.agents[0].channel._current_anchor_pos is engine.anchors[-1].position


def test_agents_do_not_share_a_channel_model():
    engine = _engine(num_agents=3)
    channels = [a.channel for a in engine.agents]
    assert len({id(c) for c in channels}) == 3


def test_invalid_measurements_skip_estimation_like_the_main_loop():
    """An unusable frame must not produce a half-formed observation."""
    engine = _engine()
    good = engine.step([_uwb_action()])["agents"][0]
    assert good["valid"] is True
    assert good["observation"] is not None

    # Force every ranging call to fail.
    agent = engine.agents[0]
    agent.tag.measure_distance_with_logs = lambda *a, **k: (-1.0, [], None)
    step_id = engine.step_index
    bad = engine.step([_uwb_action()])["agents"][0]
    assert bad["valid"] is False
    assert bad["observation"] is None
    assert bad["metrics"] is None
    assert bad["energy"] is None
    assert bad["consecutive_failures"] == 1
    # The clock still advanced, as it does in the main loop.
    assert agent.sim_time == pytest.approx((step_id + 1) * engine.dt)


def test_filter_state_is_adopted_from_algorithm_output_not_guessed():
    """Shapes must come from the algorithm, and must not drift between steps.

    The duty-cycled filter keeps an 8-element packed state (EKF + yaw + gyro
    bias) alongside a 6x6 EKF covariance. The old window guessed 8 and built an
    8x8 covariance, which silently disagreed with the filter it was driving.
    """
    engine = _engine(algorithm="Duty-Cycled UWB-IMU NA-AEKF")
    action = {"filter": "Duty-Cycled UWB-IMU NA-AEKF", "anchors": [0, 1, 2, 3],
              "measurement_source": "both"}

    shapes = []
    for _ in range(6):
        engine.step([action])
        agent = engine.agents[0]
        shapes.append((np.asarray(agent.state).size, np.asarray(agent.covariance).shape))

    assert engine.agents[0].initialized is True
    # Stable: the engine never resizes mid-run.
    assert len(set(shapes)) == 1, f"state/covariance shapes drifted: {shapes}"

    n_full = engine.algorithm_methods["Duty-Cycled UWB-IMU NA-AEKF"].n_full
    n_ekf = engine.algorithm_methods["Duty-Cycled UWB-IMU NA-AEKF"].n_ekf
    state_size, cov_shape = shapes[0]
    assert state_size == n_full
    assert cov_shape == (n_ekf, n_ekf)


def test_no_state_guessing_happens_on_the_first_step():
    """The very first update passes None, letting the algorithm self-initialise."""
    engine = _engine(algorithm="Duty-Cycled UWB-IMU NA-AEKF")
    agent = engine.agents[0]
    assert agent.state is None and agent.covariance is None and agent.initialized is False
    engine.step([{"filter": "Duty-Cycled UWB-IMU NA-AEKF", "anchors": [0, 1, 2, 3],
                  "measurement_source": "both"}])
    assert agent.state is not None, "algorithm must have self-initialised"
    assert agent.initialized is True


def test_observation_and_energy_share_one_step_id():
    engine = _engine()
    step = engine.step([_uwb_action()])
    per_agent = step["agents"][0]
    assert per_agent["step_id"] == 0
    assert per_agent["observation"]["step_id"] == 0
    anchor_id = per_agent["measurements"][0]["anchor_id"]
    assert per_agent["observation"]["measurements"][anchor_id] == pytest.approx(
        per_agent["measurements"][0]["distance"]
    )
    # Energy is recorded once and reused, never recomputed downstream.
    assert per_agent["energy"]["step_energy_uJ"] == pytest.approx(per_agent["metrics"]["step_energy_uJ"])
    assert per_agent["energy"]["source_step_id"] == 0


def test_observation_uses_same_step_estimate_and_error():
    engine = _engine()
    per_agent = engine.step([_uwb_action()])["agents"][0]
    obs = per_agent["observation"]
    assert obs["step_id"] == 0
    assert obs["tag_position_estimated"] == pytest.approx(list(per_agent["estimate"]))
    assert obs["precision"]["localization_error"] == pytest.approx(per_agent["error"])


def test_imu_gating_follows_algorithm_declaration():
    engine = _engine(algorithm="Trilateration")
    per_agent = engine.step([_uwb_action(4)])["agents"][0]
    # Trilateration declares no IMU dependency, so none may be supplied.
    assert per_agent["algorithm_input_summary"]["provide_imu"] is False

    engine2 = _engine(algorithm="Duty-Cycled UWB-IMU NA-AEKF")
    per_agent2 = engine2.step([{
        "filter": "Duty-Cycled UWB-IMU NA-AEKF", "anchors": [0, 1, 2, 3],
        "measurement_source": "both",
    }])["agents"][0]
    assert per_agent2["algorithm_input_summary"]["provide_imu"] is True


def test_reset_clears_filter_and_energy_state():
    engine = _engine()
    for _ in range(4):
        engine.step([_uwb_action()])
    assert engine.agents[0].sim_time > 0
    assert engine.agents[0].energy.cumulative_energy_uJ > 0

    engine.reset()
    agent = engine.agents[0]
    assert agent.sim_time == 0.0
    assert agent.initialized is False
    assert agent.state is None
    assert agent.covariance is None
    assert agent.Q is None
    assert agent.R is None
    assert agent.energy.cumulative_energy_uJ == 0.0
    assert agent.curr_error == 0.0
    assert engine.step_index == 0


def test_duty_cycle_action_is_applied_and_validated():
    engine = _engine(algorithm="Duty-Cycled UWB-IMU NA-AEKF")
    per_agent = engine.step([{
        "filter": "Duty-Cycled UWB-IMU NA-AEKF", "anchors": [0, 1, 2, 3],
        "measurement_source": "both", "t_imu": 0.09, "t_uwb": 0.01,
    }])["agents"][0]
    assert per_agent["duty_cycle"] == {
        "t_imu": 0.09, "t_uwb": 0.01, "cycle_length": pytest.approx(0.1),
        "active_window": pytest.approx(0.01),
    }

    with pytest.raises(ValueError):
        engine.step([{
            "filter": "Duty-Cycled UWB-IMU NA-AEKF", "anchors": [0, 1, 2, 3],
            "measurement_source": "both", "t_imu": -1.0, "t_uwb": 0.01,
        }])


def test_metrics_record_is_bounded_and_flat():
    engine = _engine()
    for _ in range(50):
        engine.step([_uwb_action()])
    assert len(engine.recent_metrics) <= engine.config.metrics_window
    last = engine.recent_metrics[-1]
    for key in ("step", "time", "error", "step_energy_uJ", "cumulative_energy_uJ"):
        assert key in last
        assert not isinstance(last[key], (list, dict, np.ndarray))


def test_multiple_agents_advance_independently():
    engine = _engine(num_agents=3)
    engine.step([_uwb_action()] * 3)
    times = [a.sim_time for a in engine.agents]
    assert times == pytest.approx([engine.dt] * 3)
    assert engine.agents[0].tag is not engine.agents[1].tag
    assert engine.agents[1].tag is not engine.agents[2].tag


def test_agents_use_their_own_anchor_records():
    engine = _engine(num_agents=2)
    report = engine.step([_uwb_action()] * 2)
    a0, a1 = report["agents"]
    assert a0["measurements"] is not a1["measurements"]
    assert a0["energy"] is not a1["energy"]


def test_no_anchors_is_rejected():
    with pytest.raises(ValueError):
        AITrainingEngine(_config(), anchors=[])
