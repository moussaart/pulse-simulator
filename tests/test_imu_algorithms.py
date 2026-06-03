import numpy as np
import pytest
from src.core.localization.base_algorithm import AlgorithmInput
from src.user_algorithms.imuspeeddeadreckoningalgorithm import ImuspeeddeadreckoningalgorithmAlgorithm


def test_imu_speed_dr_stationary_bias_convergence():
    """Verify that under stationary conditions, ZUPT is triggered and gyro bias converges."""
    algo = ImuspeeddeadreckoningalgorithmAlgorithm()
    algo.initialize()

    # Initial checks
    assert algo._gyro_bias == 0.0
    assert algo._yaw == 0.0

    state = np.zeros(4)
    covariance = np.eye(4)
    Q = np.eye(4)
    R = np.eye(2)
    initialized = False

    # Run 100 stationary steps with a constant gyroscope offset gz = 0.02
    # Raw accel is constant, so variance of norm will be 0.0 (below threshold)
    # Gyro norm is 0.02 (below default threshold of 0.05)
    for _ in range(100):
        inp = AlgorithmInput(
            measurements=[],
            anchors=[],
            tag=None,
            dt=0.02,
            state=state,
            covariance=covariance,
            Q=Q,
            R=R,
            initialized=initialized,
            imu_data_on=True,
            accel=np.array([0.1, -0.2, 9.81]),
            gyro=np.array([0.0, 0.0, 0.02]),
            params={"movement_speed": 1.5}
        )
        out = algo.update(inp)
        state = out.state
        covariance = out.covariance
        initialized = out.initialized

    # Verify that ZUPT triggered
    assert out.extra_data["zupt_triggered"] is True

    # Verify gyro bias converged close to the raw stationary input (0.02)
    assert np.isclose(algo._gyro_bias, 0.02, atol=1e-2)
    assert np.isclose(out.extra_data["gyro_bias"], 0.02, atol=1e-2)

    # Stationary state should keep velocity strictly at zero
    assert state[2] == 0.0
    assert state[3] == 0.0
    assert out.position == (0.0, 0.0)


def test_imu_speed_dr_moving_speed_scaling():
    """Verify that when moving, velocity scales precisely with movement_speed slider value."""
    algo = ImuspeeddeadreckoningalgorithmAlgorithm()
    algo.initialize()

    state = np.zeros(4)
    state[0], state[1] = 10.0, 20.0  # Seed initial position
    covariance = np.eye(4)
    initialized = True

    # Moving: set gyro norm to 0.1 (above threshold 0.05) so ZUPT is NOT triggered
    # No gyro rotation (gz = 0), so it should move in straight line along positive X axis (yaw = 0)
    # Pass a custom movement speed from the panel = 2.5 m/s
    dt = 0.1
    inp = AlgorithmInput(
        measurements=[],
        anchors=[],
        tag=None,
        dt=dt,
        state=state,
        covariance=covariance,
        initialized=initialized,
        imu_data_on=True,
        accel=np.array([0.0, 0.0, 9.81]),
        gyro=np.array([0.1, 0.0, 0.0]),  # X-rotation triggers moving mode (gyro_norm = 0.1 > 0.05)
        params={"movement_speed": 2.5}
    )
    out = algo.update(inp)

    # ZUPT should NOT be triggered
    assert out.extra_data["zupt_triggered"] is False

    # Velocity should be 2.5 m/s in X, 0.0 in Y
    assert np.isclose(out.state[2], 2.5)
    assert np.isclose(out.state[3], 0.0)

    # Position should be propagated by speed * dt
    expected_x = 10.0 + 2.5 * dt
    expected_y = 20.0
    assert np.isclose(out.position[0], expected_x)
    assert np.isclose(out.position[1], expected_y)


def test_imu_speed_dr_yaw_tracking():
    """Verify that yaw heading integrates correctly and rotates the velocity vector."""
    algo = ImuspeeddeadreckoningalgorithmAlgorithm()
    algo.initialize()

    state = np.zeros(4)
    covariance = np.eye(4)
    initialized = True

    # High gyro norm to prevent stationary mode trigger
    # Constant yaw rate gz = 0.5 rad/s
    dt = 0.1
    movement_speed = 2.0

    for i in range(5):
        inp = AlgorithmInput(
            measurements=[],
            anchors=[],
            tag=None,
            dt=dt,
            state=state,
            covariance=covariance,
            initialized=initialized,
            imu_data_on=True,
            accel=np.array([0.0, 0.0, 9.81]),
            gyro=np.array([0.0, 0.0, 0.5]),
            params={"movement_speed": movement_speed}
        )
        out = algo.update(inp)
        state = out.state

    # Expected integrated yaw = 5 steps * 0.5 rad/s * 0.1s = 0.25 rad
    assert np.isclose(algo._yaw, 0.25)
    assert np.isclose(out.extra_data["yaw"], 0.25)

    # Check velocity is correctly rotated: vx = V * cos(yaw), vy = V * sin(yaw)
    expected_vx = movement_speed * np.cos(0.25)
    expected_vy = movement_speed * np.sin(0.25)
    assert np.isclose(state[2], expected_vx)
    assert np.isclose(state[3], expected_vy)


def test_imu_speed_dr_graceful_guard():
    """Verify algorithm remains robust when IMU is off or missing."""
    algo = ImuspeeddeadreckoningalgorithmAlgorithm()
    algo.initialize()

    state = np.array([5.0, -5.0, 1.0, -1.0])
    covariance = np.eye(4)
    initialized = True

    # Run with imu_data_on = False
    inp = AlgorithmInput(
        measurements=[],
        anchors=[],
        tag=None,
        dt=0.02,
        state=state.copy(),
        covariance=covariance,
        initialized=initialized,
        imu_data_on=False,
        accel=None,
        gyro=None,
    )
    out = algo.update(inp)

    # Position should not change, velocity should be zeroed out
    assert out.position == (5.0, -5.0)
    assert out.state[2] == 0.0
    assert out.state[3] == 0.0


def test_imu_speed_dr_mid_simulation_init():
    """Verify that mid-simulation filter switches force state initialization and seed yaw correctly."""
    algo = ImuspeeddeadreckoningalgorithmAlgorithm()
    algo.initialize()

    # Create dummy tag with specific position and orientation
    class DummyTag:
        class DummyPosition:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        def __init__(self, x, y, yaw):
            self.position = self.DummyPosition(x, y)
            self.orientation = yaw

    tag = DummyTag(-4.8, -1.5, 1.95)

    # Simulate parent GUI environment passing initialized=True but local algorithm is not initialized yet
    state = np.array([10.0, 10.0, 0.0, 0.0]) # Old state from previous algorithm
    covariance = np.eye(4)
    initialized = True # Reset not called in parent GUI, so it comes in as True

    inp = AlgorithmInput(
        measurements=[],
        anchors=[],
        tag=tag,
        dt=0.05,
        state=state.copy(),
        covariance=covariance,
        initialized=initialized,
        imu_data_on=True,
        accel=np.array([0.0, 0.0, 9.81]),
        gyro=np.array([0.0, 0.0, 0.0]),
        params={"movement_speed": 1.0}
    )

    out = algo.update(inp)

    # Confirm that local initialization was forced
    assert algo._initialized is True
    assert out.initialized is True

    # Confirm state took the tag's position and not the old EKF/filter state
    assert np.isclose(out.state[0], -4.8)
    assert np.isclose(out.state[1], -1.5)

    # Confirm that _yaw was correctly seeded from the tag's true orientation
    assert np.isclose(algo._yaw, 1.95)
    assert np.isclose(out.extra_data["yaw"], 1.95)

