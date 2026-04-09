import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class DoubleintegrationzuptalgorithmAlgorithm(BaseLocalizationAlgorithm):
    """
    Indoor localization using IMU double integration with Zero Velocity Update (ZUPT).

    State vector: [x, y, vx, vy]  (position + velocity in 2D)

    Pipeline per step:
      1. Subtract gravity from accelerometer Z, rotate XY accel into world frame
         using a simple yaw integration from gyro Z.
      2. Integrate acceleration → velocity, integrate velocity → position.
      3. Detect zero-velocity intervals (stance phase) via the accel norm variance
         detector; when triggered, correct velocity to zero via a pseudo-measurement
         Kalman update (ZUPT).
      4. Propagate error-state covariance with process noise Q; apply ZUPT
         measurement update with noise R when stance is detected.

    Only IMU data is used – UWB measurements and anchors are intentionally ignored.
    """

    # ------------------------------------------------------------------ #
    #  ZUPT detector hyper-parameters (tunable via input_data.params)     #
    # ------------------------------------------------------------------ #
    DEFAULT_ZUPT_WINDOW      = 5        # samples in the sliding window
    DEFAULT_ZUPT_THRESHOLD   = 0.08     # m²/s⁴ – accel-norm variance gate
    DEFAULT_GRAVITY          = 9.81     # m/s²

    @property
    def name(self) -> str:
        return "Double Integration + ZUPT (IMU only)"

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Reset all internal buffers."""
        self._accel_norm_buffer: list[float] = []
        self._yaw: float = 0.0          # integrated heading (rad)

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:

        # ── 0. Resolve tunable parameters ──────────────────────────────
        params          = input_data.params or {}
        zupt_window     = int(params.get("zupt_window",     self.DEFAULT_ZUPT_WINDOW))
        zupt_threshold  = float(params.get("zupt_threshold", self.DEFAULT_ZUPT_THRESHOLD))
        gravity         = float(params.get("gravity",        self.DEFAULT_GRAVITY))

        dt          = input_data.dt
        imu_on      = input_data.imu_data_on
        accel_raw   = input_data.accel          # [ax, ay, az]  m/s²
        gyro_raw    = input_data.gyro           # [gx, gy, gz]  rad/s

        # ── 1. Initialisation on first call ────────────────────────────
        state      = input_data.state
        covariance = input_data.covariance
        Q          = input_data.Q
        R          = input_data.R
        initialized = input_data.initialized

        if not initialized:
            # state = [x, y, vx, vy]
            state      = np.zeros(4)
            covariance = np.eye(4) * 1.0

            # Process noise: higher for velocity states (integration drift)
            Q = np.diag([1e-4, 1e-4,   # position noise
                         1e-2, 1e-2])  # velocity noise

            # Measurement noise for ZUPT (we "measure" vx=0, vy=0)
            R = np.eye(2) * 1e-3

            # seed position from tag if available
            if input_data.tag is not None and input_data.tag.position is not None:
                state[0] = input_data.tag.position.x
                state[1] = input_data.tag.position.y

            self._accel_norm_buffer = []
            self._yaw = 0.0
            initialized = True

        # ── 2. Guard: need IMU data ─────────────────────────────────────
        if not imu_on or accel_raw is None or gyro_raw is None:
            x, y = float(state[0]), float(state[1])
            return self._make_output(x, y, state, covariance, Q, R, initialized,
                                     input_data, zupt_triggered=False)

        accel = np.asarray(accel_raw, dtype=float)   # shape (3,)
        gyro  = np.asarray(gyro_raw,  dtype=float)   # shape (3,)

        # ── 3. Integrate gyro Z → yaw ───────────────────────────────────
        #  (Simple Euler; good enough for pedestrian rates < 90 °/s)
        yaw_rate = gyro[2]                           # rad/s  around vertical axis
        self._yaw += yaw_rate * dt
        # keep in (-π, π)
        self._yaw = (self._yaw + np.pi) % (2 * np.pi) - np.pi

        # ── 4. Remove gravity, project to world horizontal plane ────────
        #  Gravity is assumed to be fully on the Z axis of the sensor
        #  (i.e. sensor is roughly horizontal — typical for wrist/waist mount).
        accel_horizontal = accel[:2].copy()          # [ax_sensor, ay_sensor]

        # Rotate sensor frame → world frame by yaw
        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        R_yaw = np.array([[cy, -sy],
                          [sy,  cy]])
        accel_world = R_yaw @ accel_horizontal       # [ax_world, ay_world]

        # ── 5. ZUPT detector ────────────────────────────────────────────
        accel_norm = float(np.linalg.norm(accel))    # full 3-axis norm
        self._accel_norm_buffer.append(accel_norm)
        if len(self._accel_norm_buffer) > zupt_window:
            self._accel_norm_buffer.pop(0)

        # Variance of the norm window as the "stillness" criterion
        norm_variance = float(np.var(self._accel_norm_buffer))
        zupt_triggered = (len(self._accel_norm_buffer) == zupt_window
                          and norm_variance < zupt_threshold)

        # ── 6. State propagation (constant-acceleration kinematic model) ─
        #
        #  x(k+1)  = x(k)  + vx(k)*dt + 0.5*ax*dt²
        #  y(k+1)  = y(k)  + vy(k)*dt + 0.5*ay*dt²
        #  vx(k+1) = vx(k) + ax*dt
        #  vy(k+1) = vy(k) + ay*dt
        #
        F = np.array([[1, 0, dt,  0],
                      [0, 1,  0, dt],
                      [0, 0,  1,  0],
                      [0, 0,  0,  1]])

        B = np.array([[0.5 * dt**2, 0          ],
                      [0,           0.5 * dt**2],
                      [dt,          0          ],
                      [0,           dt         ]])

        state      = F @ state + B @ accel_world
        covariance = F @ covariance @ F.T + Q

        # ── 7. ZUPT measurement update ──────────────────────────────────
        #  Pseudo-measurement: z = [0, 0]  (velocity = zero during stance)
        #  Observation matrix picks out vx, vy from state
        if zupt_triggered:
            H = np.array([[0, 0, 1, 0],
                          [0, 0, 0, 1]])          # (2×4)

            S = H @ covariance @ H.T + R           # innovation covariance (2×2)
            K = covariance @ H.T @ np.linalg.inv(S)  # Kalman gain (4×2)

            z          = np.zeros(2)               # measured velocity = 0
            innovation = z - H @ state             # should be ≈ -[vx, vy]

            state      = state + K @ innovation
            covariance = (np.eye(4) - K @ H) @ covariance

        # ── 8. Pack output ──────────────────────────────────────────────
        x, y = float(state[0]), float(state[1])
        return self._make_output(x, y, state, covariance, Q, R, initialized,
                                 input_data, zupt_triggered,
                                 extra={"yaw_rad":        self._yaw,
                                        "accel_world":    accel_world.tolist(),
                                        "norm_variance":  norm_variance,
                                        "zupt_threshold": zupt_threshold})

    # ------------------------------------------------------------------ #
    #  Helper                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_output(x, y, state, covariance, Q, R, initialized,
                     input_data, zupt_triggered, extra=None) -> AlgorithmOutput:
        extra_data = {"zupt_triggered": zupt_triggered}
        if extra:
            extra_data.update(extra)
        return AlgorithmOutput(
            position          = (x, y),
            state             = state,
            covariance        = covariance,
            initialized       = initialized,
            previous_state    = input_data.state,
            previous_covariance = input_data.covariance,
            Q                 = Q,
            R                 = R,
            extra_data        = extra_data,
        )