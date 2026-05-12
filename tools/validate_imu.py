"""Generate deterministic IMU validation logs and plots.

Run from the repository root:
    python tools/validate_imu.py

Outputs CSV files and PNG plots under validation/imu by default. The IMU is
configured with noise, bias, scale-factor error, and bias random walk disabled
so measurements can be compared directly with analytical or numerical ground
truth.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.uwb.imu import IMUSimulator
from src.core.uwb.uwb_devices import Position


G = 9.81


@dataclass
class State:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float
    angular_velocity: np.ndarray


def deterministic_imu(sample_rate: float) -> IMUSimulator:
    return IMUSimulator(
        sample_rate=sample_rate,
        acc_noise_std=0.0,
        gyro_noise_std=0.0,
        acc_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        bias_instability=0.0,
        acc_scale_factor=np.zeros(3),
        gyro_scale_factor=np.zeros(3),
    )


def expected_measurement(state: State) -> tuple[np.ndarray, np.ndarray]:
    rotation = IMUSimulator._world_to_body_matrix(state.yaw)
    gravity_world = np.array([0.0, 0.0, -G])
    acc_expected = rotation @ (state.acceleration - gravity_world)
    gyro_expected = rotation @ state.angular_velocity
    return acc_expected, gyro_expected


def simulate_case(name: str, states: list[State], dt: float, output_dir: str, make_plots: bool) -> dict:
    imu = deterministic_imu(sample_rate=1.0 / dt)
    rows = []
    max_acc_error = 0.0
    max_gyro_error = 0.0
    max_acc_norm = 0.0

    previous_t = 0.0
    for index, state in enumerate(states):
        t = index * dt
        sample_dt = dt if index == 0 else t - previous_t
        previous_t = t

        acc_meas, gyro_meas = imu.generate_imu_data(
            Position(*state.position),
            state.yaw,
            sample_dt,
            velocity=state.velocity,
            acceleration=state.acceleration,
            angular_velocity=state.angular_velocity,
        )
        acc_expected, gyro_expected = expected_measurement(state)
        acc_error = acc_meas - acc_expected
        gyro_error = gyro_meas - gyro_expected

        if not np.all(np.isfinite(acc_meas)) or not np.all(np.isfinite(gyro_meas)):
            raise AssertionError(f"{name}: non-finite IMU measurement at t={t:.3f}s")

        max_acc_error = max(max_acc_error, float(np.linalg.norm(acc_error)))
        max_gyro_error = max(max_gyro_error, float(np.linalg.norm(gyro_error)))
        max_acc_norm = max(max_acc_norm, float(np.linalg.norm(acc_meas)))

        rows.append([
            t,
            *state.position,
            *state.velocity,
            *state.acceleration,
            state.yaw,
            *state.angular_velocity,
            *acc_expected,
            *acc_meas,
            *acc_error,
            *gyro_expected,
            *gyro_meas,
            *gyro_error,
        ])

    if max_acc_norm > 200.0:
        raise AssertionError(f"{name}: unrealistic acceleration norm {max_acc_norm:.3f} m/s^2")
    if max_acc_error > 1e-10 or max_gyro_error > 1e-10:
        raise AssertionError(
            f"{name}: validation error too high, acc={max_acc_error:.3e}, gyro={max_gyro_error:.3e}"
        )

    write_csv(name, rows, output_dir)
    if make_plots:
        write_plot(name, rows, output_dir)

    return {
        "case": name,
        "samples": len(rows),
        "max_acc_error": max_acc_error,
        "max_gyro_error": max_gyro_error,
        "max_acc_norm": max_acc_norm,
    }


def write_csv(name: str, rows: list[list[float]], output_dir: str) -> None:
    headers = [
        "t",
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "acc_world_x", "acc_world_y", "acc_world_z",
        "yaw",
        "gyro_world_x", "gyro_world_y", "gyro_world_z",
        "acc_expected_x", "acc_expected_y", "acc_expected_z",
        "acc_measured_x", "acc_measured_y", "acc_measured_z",
        "acc_error_x", "acc_error_y", "acc_error_z",
        "gyro_expected_x", "gyro_expected_y", "gyro_expected_z",
        "gyro_measured_x", "gyro_measured_y", "gyro_measured_z",
        "gyro_error_x", "gyro_error_y", "gyro_error_z",
    ]
    path = os.path.join(output_dir, f"{name}.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_plot(name: str, rows: list[list[float]], output_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.asarray(rows, dtype=float)
    t = data[:, 0]
    first_seconds = t <= min(2.0, t[-1])
    t_first = t[first_seconds]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    axes[0].plot(t, data[:, 1], label="x")
    axes[0].plot(t, data[:, 2], label="y")
    axes[0].plot(t, data[:, 3], label="z")
    axes[0].set_title(f"{name}: ground-truth position")
    axes[0].set_ylabel("m")
    axes[0].legend(loc="best")
    axes[0].grid(True)

    for axis, label in enumerate(["x", "y", "z"]):
        axes[1].plot(t_first, data[first_seconds, 17 + axis], label=f"acc {label}")
    axes[1].set_title("First seconds: measured accelerometer")
    axes[1].set_ylabel("m/s^2")
    axes[1].legend(loc="best")
    axes[1].grid(True)

    acc_error_norm = np.linalg.norm(data[:, 20:23], axis=1)
    gyro_error_norm = np.linalg.norm(data[:, 29:32], axis=1)
    axes[2].plot(t, acc_error_norm, label="accelerometer error norm")
    axes[2].plot(t, gyro_error_norm, label="gyroscope error norm")
    axes[2].set_title("Measurement error over time")
    axes[2].set_xlabel("s")
    axes[2].legend(loc="best")
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=140)
    plt.close(fig)


def build_states(duration: float, dt: float, fn: Callable[[float], State]) -> list[State]:
    return [fn(t) for t in np.arange(0.0, duration + 0.5 * dt, dt)]


def stationary(t: float) -> State:
    return State(np.array([1.0, 2.0, 0.0]), np.zeros(3), np.zeros(3), 0.0, np.zeros(3))


def straight_constant_velocity(t: float) -> State:
    velocity = np.array([1.5, -0.5, 0.0])
    return State(velocity * t, velocity, np.zeros(3), np.arctan2(velocity[1], velocity[0]), np.zeros(3))


def constant_acceleration(t: float) -> State:
    acceleration = np.array([0.8, -0.3, 0.0])
    velocity = acceleration * t
    yaw = np.arctan2(velocity[1], velocity[0]) if np.linalg.norm(velocity[:2]) > 1e-12 else 0.0
    return State(0.5 * acceleration * t * t, velocity, acceleration, yaw, np.zeros(3))


def circular(t: float) -> State:
    radius = 2.0
    omega = 1.25
    theta = omega * t
    position = np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0])
    velocity = np.array([-radius * omega * np.sin(theta), radius * omega * np.cos(theta), 0.0])
    acceleration = np.array([-radius * omega**2 * np.cos(theta), -radius * omega**2 * np.sin(theta), 0.0])
    return State(position, velocity, acceleration, theta + np.pi / 2.0, np.array([0.0, 0.0, omega]))


def vertical(t: float) -> State:
    omega = 2.0
    amplitude = 0.6
    position = np.array([0.0, 0.0, amplitude * np.sin(omega * t)])
    velocity = np.array([0.0, 0.0, amplitude * omega * np.cos(omega * t)])
    acceleration = np.array([0.0, 0.0, -amplitude * omega**2 * np.sin(omega * t)])
    return State(position, velocity, acceleration, 0.0, np.zeros(3))


def project_trajectory_states(path: str) -> tuple[list[State], float]:
    times = []
    positions = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            times.append(float(row["timestamp"]))
            positions.append([float(row["x"]), float(row["y"]), float(row["z"])])

    times_array = np.asarray(times, dtype=float)
    positions_array = np.asarray(positions, dtype=float)
    dt = float(np.median(np.diff(times_array)))
    velocity = np.gradient(positions_array, times_array, axis=0)
    acceleration = np.gradient(velocity, times_array, axis=0)
    yaw = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    yaw[~np.isfinite(yaw)] = 0.0
    yaw_rate = np.gradient(yaw, times_array)

    states = [
        State(positions_array[i], velocity[i], acceleration[i], float(yaw[i]), np.array([0.0, 0.0, yaw_rate[i]]))
        for i in range(len(times_array))
    ]
    return states, dt


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate deterministic IMU measurements against ground truth.")
    parser.add_argument("--output", default=os.path.join(ROOT, "validation", "imu"), help="output directory")
    parser.add_argument("--no-plots", action="store_true", help="write CSV logs only")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    make_plots = not args.no_plots

    dt = 0.01
    summaries = [
        simulate_case("stationary", build_states(3.0, dt, stationary), dt, args.output, make_plots),
        simulate_case("constant_velocity", build_states(3.0, dt, straight_constant_velocity), dt, args.output, make_plots),
        simulate_case("constant_acceleration", build_states(3.0, dt, constant_acceleration), dt, args.output, make_plots),
        simulate_case("circular", build_states(3.0, dt, circular), dt, args.output, make_plots),
        simulate_case("vertical", build_states(3.0, dt, vertical), dt, args.output, make_plots),
    ]

    trajectory_path = os.path.join(ROOT, "data", "trajectories", "trajectory.csv")
    trajectory_states, trajectory_dt = project_trajectory_states(trajectory_path)
    summaries.append(simulate_case("project_trajectory", trajectory_states, trajectory_dt, args.output, make_plots))

    print("IMU validation complete")
    for summary in summaries:
        print(
            f"{summary['case']}: samples={summary['samples']}, "
            f"max_acc_error={summary['max_acc_error']:.3e}, "
            f"max_gyro_error={summary['max_gyro_error']:.3e}, "
            f"max_acc_norm={summary['max_acc_norm']:.3f}"
        )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
