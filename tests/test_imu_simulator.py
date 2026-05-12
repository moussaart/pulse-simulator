import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.uwb.imu import IMUSimulator
from src.core.uwb.uwb_devices import Position, Tag


G = 9.81


def make_deterministic_imu(sample_rate=50):
    imu = IMUSimulator(sample_rate=sample_rate)
    imu.acc_noise_std = 0.0
    imu.gyro_noise_std = 0.0
    imu.acc_bias = np.zeros(3)
    imu.gyro_bias = np.zeros(3)
    imu.bias_instability = 0.0
    return imu


def assert_finite_vector(vector):
    assert np.all(np.isfinite(vector)), f"non-finite vector: {vector}"


def test_tag_imu_update_has_no_startup_spike_after_t_zero_sample():
    tag = Tag(Position(0.0, 0.0, 0.0))
    tag.imu_simulator = make_deterministic_imu(sample_rate=50)
    tag.imu_data.clear()

    dt = 0.02
    velocity = np.array([5.0, 0.0, 0.0])

    for t in [0.0, dt, 2.0 * dt, 3.0 * dt]:
        tag.position = Position(*(velocity * t))
        tag.velocity = Position(*velocity)
        tag.acceleration = Position(0.0, 0.0, 0.0)
        tag.orientation = 0.0
        tag.angular_velocity = 0.0
        tag.update_imu(t)

    acc_xy = np.column_stack((tag.imu_data.acc_x, tag.imu_data.acc_y))
    max_horizontal_acc = float(np.max(np.linalg.norm(acc_xy, axis=1)))
    assert max_horizontal_acc < 1e-9
    assert np.allclose(tag.imu_data.acc_z, G, atol=1e-12)


def test_stationary_measurement_is_gravity_only_with_zero_gyro():
    imu = make_deterministic_imu()

    for _ in range(5):
        acc, gyro = imu.generate_imu_data(Position(1.0, -2.0, 0.5), 0.0, 0.02)
        assert_finite_vector(acc)
        assert_finite_vector(gyro)
        assert np.allclose(acc, [0.0, 0.0, G], atol=1e-12)
        assert np.allclose(gyro, [0.0, 0.0, 0.0], atol=1e-12)


def test_constant_velocity_has_no_artificial_acceleration_spike():
    imu = make_deterministic_imu(sample_rate=50)
    dt = 0.02
    velocity = np.array([3.0, -4.0, 0.0])

    measurements = []
    for i in range(6):
        position = velocity * (i * dt)
        acc, _ = imu.generate_imu_data(Position(*position), 0.0, dt)
        measurements.append(acc)

    measurements = np.asarray(measurements)
    horizontal_norm = np.linalg.norm(measurements[:, :2], axis=1)
    assert np.max(horizontal_norm) < 1e-9
    assert np.allclose(measurements[:, 2], G, atol=1e-12)


def test_constant_acceleration_matches_specific_force_in_body_frame():
    imu = make_deterministic_imu(sample_rate=100)
    dt = 0.01
    linear_acc_world = np.array([2.0, -1.0, 0.0])

    for i in range(4):
        t = i * dt
        position = 0.5 * linear_acc_world * t * t
        velocity = linear_acc_world * t
        acc, gyro = imu.generate_imu_data(
            Position(*position),
            0.0,
            dt,
            velocity=velocity,
            acceleration=linear_acc_world,
            angular_velocity=np.zeros(3),
        )
        assert np.allclose(acc, [2.0, -1.0, G], atol=1e-12)
        assert np.allclose(gyro, [0.0, 0.0, 0.0], atol=1e-12)


def test_circular_motion_matches_centripetal_acceleration_and_yaw_rate():
    imu = make_deterministic_imu(sample_rate=100)
    dt = 0.01
    radius = 2.0
    omega = 1.5
    expected_centripetal = omega * omega * radius

    for i in range(1, 6):
        t = i * dt
        theta = omega * t
        position = np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        velocity = np.array([-radius * omega * np.sin(theta), radius * omega * np.cos(theta), 0.0])
        acceleration = np.array([
            -radius * omega * omega * np.cos(theta),
            -radius * omega * omega * np.sin(theta),
            0.0,
        ])
        yaw = theta + np.pi / 2.0
        acc, gyro = imu.generate_imu_data(
            Position(*position),
            yaw,
            dt,
            velocity=velocity,
            acceleration=acceleration,
            angular_velocity=np.array([0.0, 0.0, omega]),
        )

        assert np.allclose(acc, [0.0, expected_centripetal, G], atol=1e-12)
        assert np.allclose(gyro, [0.0, 0.0, omega], atol=1e-12)


def test_vertical_acceleration_is_not_double_counted_or_removed():
    imu = make_deterministic_imu(sample_rate=100)
    linear_acc_world = np.array([0.0, 0.0, 1.25])

    acc, gyro = imu.generate_imu_data(
        Position(0.0, 0.0, 0.0),
        0.0,
        0.01,
        velocity=np.zeros(3),
        acceleration=linear_acc_world,
        angular_velocity=np.zeros(3),
    )

    assert np.allclose(acc, [0.0, 0.0, G + 1.25], atol=1e-12)
    assert np.allclose(gyro, [0.0, 0.0, 0.0], atol=1e-12)


def test_existing_project_trajectory_has_no_initial_acceleration_spike():
    imu = make_deterministic_imu(sample_rate=50)
    trajectory_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "trajectories",
        "trajectory.csv",
    )

    samples = []
    with open(trajectory_path, newline="") as handle:
        for row in csv.DictReader(handle):
            samples.append((float(row["timestamp"]), Position(float(row["x"]), float(row["y"]), float(row["z"]))))

    accelerations = []
    previous_time = samples[0][0]
    for timestamp, position in samples[:8]:
        dt = max(timestamp - previous_time, 0.02)
        acc, _ = imu.generate_imu_data(position, -np.pi / 2.0, dt)
        accelerations.append(acc)
        previous_time = timestamp

    horizontal_norm = np.linalg.norm(np.asarray(accelerations)[:, :2], axis=1)
    assert np.max(horizontal_norm) < 1e-9
