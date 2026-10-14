"""The training engine must reproduce the main simulation's motion, IMU, LOS and
ranging sequence. `src/app/localization_main.py` and `src/app/Localization_app.py`
are read-only inputs here; only `SimulationManager` is driven.

The stub below satisfies the GUI attributes `SimulationManager` touches, and
`update_visualizations` is replaced with a no-op so the test exercises simulation
behaviour rather than Qt painting.
"""

import numpy as np
import pytest

from src.api.ai_training_engine import (
    AITrainingEngine,
    TrainingConfig,
    UnknownMovementPatternError,
    validate_movement_pattern,
)
from src.core.uwb.channel_model import ChannelConditions
from src.core.uwb.uwb_devices import Anchor, Position, Tag
from src.core.parallel.gpu_backend import get_array_module

DT = 0.005
START = (1.0, 1.0)
STEPS = 10
PATTERNS = ["Circular", "Figure 8", "Square", "Random Walk", "Foot Mounted"]


def _anchors():
    return [
        Anchor(Position(0.0, 0.0, 0.0)),
        Anchor(Position(5.0, 0.0, 0.0)),
        Anchor(Position(0.0, 5.0, 0.0)),
        Anchor(Position(5.0, 5.0, 0.0)),
    ]


class _NoOp:
    def __getattr__(self, _name):
        return lambda *a, **k: None


class _StubParent:
    """Exposes the attributes SimulationManager reads from the main window."""

    def __init__(self, anchors, pattern="Random Walk"):
        self.dt = DT
        self.movement_pattern = pattern
        self.movement_speed = 1.0
        self.point = (0.0, 0.0)
        self.exact_trajectory_mode = False
        self.enable_motion_class = False
        self.add_imperfections = False
        self.los_aware_probabilite_erreur = 0.0
        self.los_aware_alpha = 0.3
        self.algorithm = "Trilateration"
        self.algorithm_imu_overrides = {}
        self.is_paused = False
        self.anchors = anchors
        self.channel_conditions = ChannelConditions()
        self.tag = Tag(Position(*START))
        self.tag.imu_data_on = True
        # GUI surfaces: present but inert.
        self.plot_manager = _NoOp()
        self.nlos_manager = _NoOp()
        self.position_plot = None
        self.distance_plots_window = None
        self.cir_window = None
        self.elapsed_time_display = None
        self.timeline_widget = None
        self.status_display = None
        self.gpu_status_lbl = None
        self.kf_state = None
        self.kf_P = None
        self.kf_initialized = False
        self.previous_kf_state = None
        self.previous_kf_P = None
        self.aekf_Q = None
        self.aekf_R = None
        # energy_calculator / training_api are deliberately absent so the manager
        # skips those optional paths, exactly as it does for a bare window.


def _build_pair(seed=1234, pattern="Random Walk"):
    """Main manager and training engine over identical geometry.

    Two details make the comparison meaningful:

    * Both sides must take the *same* ranging branch. ``SimulationManager`` picks
      the GPU batch path whenever ``gpu_manager.available``, so the engine is
      built with ``allow_gpu=True`` to match.
    * Construction-time randomness is pinned. ``UWBDevice.__init__`` draws a
      ``clock_drift`` per device and ``IMUSimulator`` draws ``acc_bias``/
      ``gyro_bias``, all from the global stream. The two sides are built at
      different points in that stream, so without pinning these the ranges and IMU
      samples could never line up no matter what the engine does.
    """
    from src.gui.managers.simulation_manager import SimulationManager

    parent = _StubParent(_anchors(), pattern=pattern)
    manager = SimulationManager(parent)
    manager.is_paused = False
    manager.simulation_ended = False
    manager.max_duration = None

    # update_simulation() runs the WHOLE frame: clock, motion, LOS, ranging,
    # estimation, energy, snapshot. Tapping update_visualizations reads the frame's
    # own results; calling get_measurements() separately would range a second time
    # and desynchronise the RNG stream.
    manager.frame = None

    def _capture(estimated_pos, error, measurements, frame_los_results=None):
        manager.frame = {
            "estimated_pos": estimated_pos,
            "error": error,
            "measurements": list(measurements),
            "los": [bool(is_los) for _anchor, is_los in (frame_los_results or [])],
        }

    manager.update_visualizations = _capture

    engine = AITrainingEngine(
        TrainingConfig(
            algorithm="Trilateration", dt=DT, movement_pattern=pattern,
            movement_speed=1.0, measurement_source="uwb", num_agents=1,
            start_position=START, allow_gpu=True,
        ),
        anchors=_anchors(),
    )

    main_devices = list(parent.anchors) + [parent.tag]
    train_devices = list(engine.anchors) + [engine.agents[0].tag]
    for i, (a, b) in enumerate(zip(main_devices, train_devices)):
        drift = 1.5 + i
        a.clock_drift = drift
        b.clock_drift = drift

    for tag in (parent.tag, engine.agents[0].tag):
        sim = tag.imu_simulator
        sim.acc_bias = np.array([0.011, -0.022, 0.033])
        sim.gyro_bias = np.array([0.001, -0.002, 0.003])
        sim.prev_position = None
        sim.prev_velocity = None
        sim.prev_orientation = None

    return manager, parent, engine, seed


def _train_action():
    return [{"filter": "Trilateration", "anchors": [0, 1, 2, 3], "measurement_source": "uwb"}]


def _seed(seed):
    """Seed every RNG the simulation touches.

    ``cuda_kernels`` draws ranging noise through ``get_array_module().random``,
    which is CuPy on this machine and has a stream independent of NumPy's. Seeding
    only NumPy leaves the second caller consuming noise the first already drew.
    """
    np.random.seed(seed)
    xp = get_array_module()
    if xp is np:
        return
    try:
        xp.random.seed(seed)
    except Exception:
        pass


def _pairs(steps=STEPS, exact_mode=False, pattern="Random Walk"):
    """Yield one matched step from both sides, with both RNG streams aligned."""
    manager, parent, engine, seed = _build_pair(pattern=pattern)
    if exact_mode:
        parent.exact_trajectory_mode = True
        engine.agents[0].exact_mode = True
    for step in range(steps):
        _seed(seed + step)
        manager.frame = None
        manager.update_simulation()
        main_frame = manager.frame  # None means the frame bailed out early

        _seed(seed + step)
        report = engine.step(_train_action())
        yield {
            "step": step,
            "pattern": pattern,
            "parent": parent,
            "manager": manager,
            "agent": engine.agents[0],
            "main_frame": main_frame,
            "valid": main_frame is not None,
            "per_agent": report["agents"][0],
        }


def test_motion_matches_the_main_manager():
    for c in _pairs():
        tag = c["agent"].tag
        assert (tag.position.x, tag.position.y) == pytest.approx(
            (c["parent"].tag.position.x, c["parent"].tag.position.y), abs=1e-12
        ), f"motion diverged at step {c['step']}"
        assert c["agent"].sim_time == pytest.approx(c["manager"].simulation_time)


@pytest.mark.parametrize("pattern", PATTERNS)
def test_every_movement_pattern_matches_the_main_manager(pattern):
    """Covers each branch of MotionController.update_tag_position."""
    moved = False
    for c in _pairs(pattern=pattern):
        tag = c["agent"].tag
        assert (tag.position.x, tag.position.y) == pytest.approx(
            (c["parent"].tag.position.x, c["parent"].tag.position.y), abs=1e-12
        ), f"{pattern} diverged at step {c['step']}"
        moved = moved or (abs(tag.position.x - START[0]) > 1e-9
                          or abs(tag.position.y - START[1]) > 1e-9)
    assert moved, f"{pattern} never moved; the comparison would be vacuous"


def test_fixed_point_pattern_honours_the_configured_point():
    """Fixed Point parks the tag on `point`, so the plumbing itself is the test."""
    manager, parent, engine, _seed_value = _build_pair(pattern="Fixed Point")
    assert (parent.point, engine.config.start_position) != (None, None)
    for step in range(STEPS):
        _seed(_seed_value + step)
        manager.update_simulation()
        _seed(_seed_value + step)
        engine.step(_train_action())
        assert (engine.agents[0].tag.position.x, engine.agents[0].tag.position.y) == pytest.approx(
            (parent.tag.position.x, parent.tag.position.y), abs=1e-12
        ), f"Fixed Point diverged at step {step}"


def test_imu_samples_match_the_main_manager():
    for c in _pairs():
        main_imu = c["parent"].tag.imu_data
        train_imu = c["agent"].tag.imu_data
        assert len(main_imu) == len(train_imu)
        for field in ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"):
            assert getattr(train_imu, field)[-1] == pytest.approx(
                getattr(main_imu, field)[-1], abs=1e-12
            ), f"IMU {field} diverged at step {c['step']}"


def test_los_and_ranging_match_the_main_manager():
    for c in _pairs():
        per_agent = c["per_agent"]
        assert per_agent["valid"] is c["valid"], f"validity diverged at step {c['step']}"
        if not c["valid"]:
            continue
        frame = c["main_frame"]
        assert len(frame["measurements"]) == len(per_agent["measurements"])
        for i, record in enumerate(per_agent["measurements"]):
            assert record["distance"] == pytest.approx(frame["measurements"][i], abs=1e-9), \
                f"ranging diverged at step {c['step']}, anchor {i}"
            assert record["is_los"] is frame["los"][i], f"LOS diverged at step {c['step']}, anchor {i}"


def test_estimates_and_error_match_the_main_manager_for_trilateration():
    for c in _pairs():
        if not c["valid"]:
            continue
        frame = c["main_frame"]
        estimate = c["per_agent"]["estimate"]
        assert estimate == pytest.approx(tuple(frame["estimated_pos"]), abs=1e-9), \
            f"estimate diverged at step {c['step']}"
        assert c["per_agent"]["error"] == pytest.approx(frame["error"], abs=1e-9), \
            f"error diverged at step {c['step']}"


def test_exact_trajectory_mode_is_honoured_on_both_sides():
    for c in _pairs(exact_mode=True):
        tag = c["agent"].tag
        assert (tag.position.x, tag.position.y) == pytest.approx(
            (c["parent"].tag.position.x, c["parent"].tag.position.y), abs=1e-12
        ), f"exact mode diverged at step {c['step']}"


def test_an_unknown_movement_pattern_is_rejected():
    """MotionController has no fallback branch, so a typo would freeze the tag."""
    with pytest.raises(UnknownMovementPatternError):
        validate_movement_pattern("random")
    with pytest.raises(UnknownMovementPatternError):
        validate_movement_pattern("")
    with pytest.raises(UnknownMovementPatternError):
        TrainingConfig(movement_pattern="Random walk")
    # Real names, and the Custom: escape hatch, stay accepted.
    for name in PATTERNS:
        assert validate_movement_pattern(name) == name
    assert validate_movement_pattern("Custom:walk_01") == "Custom:walk_01"
