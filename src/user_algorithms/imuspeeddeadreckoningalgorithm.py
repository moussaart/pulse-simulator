import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class ImuspeeddeadreckoningalgorithmAlgorithm(BaseLocalizationAlgorithm):
    """
    Adaptive EKF-based IMU Dead-Reckoning using UI movement speed and integrated heading.

    State vector : [x, y, vx, vy]
    Measurements : velocity vector derived from speed magnitude + gyro-integrated heading

    Extends the basic dead-reckoning with adaptive R and Q updates based on
    innovation statistics (same mechanism as the stand-alone AEKF).
    """

    # ------------------------------------------------------------------ #
    #  Stance / ZUPT detector parameters                                 #
    # ------------------------------------------------------------------ #
    DEFAULT_ZUPT_WINDOW      = 5        # samples in the sliding window
    DEFAULT_ZUPT_THRESHOLD   = 0.08     # m²/s⁴ – accel-norm variance gate
    DEFAULT_GYRO_THRESHOLD   = 0.05     # rad/s – gyro norm stillness gate

    # ------------------------------------------------------------------ #
    #  EKF / Adaptive tuning parameters                                  #
    # ------------------------------------------------------------------ #
    PROCESS_NOISE_POS = 0.1
    PROCESS_NOISE_VEL = 1.0
    MEASUREMENT_NOISE = 0.15

    ALPHA = 0.5   # smoothing factor for adaptive R
    BETA  = 0.5   # smoothing factor for adaptive Q

    uses_imu = True
    required_sensors = ("imu",)

    @property
    def name(self) -> str:
        return "IMU Speed Dead Reckoning"

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Reset all internal states."""
        self._accel_norm_buffer: list[float] = []
        self._yaw: float = 0.0          # integrated heading (rad)
        self._gyro_bias: float = 0.0     # Z-gyro bias (rad/s)
        self._initialized: bool = False

    # ------------------------------------------------------------------ #
    #  EKF helpers                                                         #
    # ------------------------------------------------------------------ #

    def _build_F(self, dt: float) -> np.ndarray:
        """State transition matrix (constant-velocity model)."""
        return np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

    def _build_Q(self, dt: float) -> np.ndarray:
        """Process noise covariance (correlated position-velocity)."""
        sp = self.PROCESS_NOISE_POS
        sv = self.PROCESS_NOISE_VEL
        q_1d = np.array([
            [dt**4 / 4 * sp**2,      dt**3 / 2 * sp * sv],
            [dt**3 / 2 * sp * sv,    dt**2 * sv**2],
        ])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = q_1d
        Q[np.ix_([1, 3], [1, 3])] = q_1d
        return Q

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        # Resolve tunable parameters or use defaults
        params          = input_data.params or {}
        zupt_window     = int(params.get("zupt_window",     self.DEFAULT_ZUPT_WINDOW))
        zupt_threshold  = float(params.get("zupt_threshold", self.DEFAULT_ZUPT_THRESHOLD))
        gyro_threshold  = float(params.get("gyro_threshold", self.DEFAULT_GYRO_THRESHOLD))
        movement_speed  = float(params.get("movement_speed", 1.0))

        measurements    = input_data.measurements
        dt              = input_data.dt
        imu_on          = input_data.imu_data_on
        accel_raw       = input_data.accel          # [ax, ay, az]  m/s²
        gyro_raw        = input_data.gyro           # [gx, gy, gz]  rad/s

        state       = input_data.state
        covariance  = input_data.covariance
        Q           = input_data.Q
        R           = input_data.R
        initialized = input_data.initialized

        # ── 1. Initialisation ───────────────────────────────────────────
        if not initialized or not getattr(self, '_initialized', False):
            # state = [x, y, vx, vy]
            if state is None or len(state) != 4:
                state = np.zeros(4)
            covariance = np.diag([5.0, 5.0, 10.0, 10.0])
            Q = self._build_Q(dt)
            R = np.eye(2) * self.MEASUREMENT_NOISE**2

            # Seed position from tag if available
            if input_data.tag is not None and getattr(input_data.tag, 'position', None) is not None:
                state[0] = input_data.tag.position.x
                state[1] = input_data.tag.position.y

            # Seed initial heading angle from tag ground truth
            if input_data.tag is not None and hasattr(input_data.tag, 'orientation'):
                self._yaw = float(input_data.tag.orientation)
            else:
                self._yaw = 0.0

            self._accel_norm_buffer = []
            self._gyro_bias = 0.0
            self._initialized = True
            initialized = True

        # Guard: need IMU data
        if not imu_on or accel_raw is None or gyro_raw is None:
            # Keep position, set velocity to zero
            x, y = float(state[0]), float(state[1])
            state[2] = 0.0
            state[3] = 0.0
            return AlgorithmOutput(
                position=(x, y),
                state=state,
                covariance=covariance,
                initialized=initialized,
                previous_state=input_data.state,
                previous_covariance=input_data.covariance,
                Q=Q,
                R=R,
                extra_data={
                    "zupt_triggered": False,
                    "yaw": self._yaw,
                    "gyro_bias": self._gyro_bias,
                    "acc_bias": np.zeros(2) # Interface compatibility
                }
            )

        accel = np.asarray(accel_raw, dtype=float)   # shape (3,)
        gyro  = np.asarray(gyro_raw,  dtype=float)   # shape (3,)

        # ── 2. ZUPT / stillness detection ───────────────────────────────
        acc_norm = float(np.linalg.norm(accel))
        if not self._accel_norm_buffer:
            self._accel_norm_buffer = [acc_norm] * zupt_window
        else:
            self._accel_norm_buffer.append(acc_norm)
            if len(self._accel_norm_buffer) > zupt_window:
                self._accel_norm_buffer.pop(0)

        norm_variance = 0.0
        if len(self._accel_norm_buffer) >= 2:
            norm_variance = float(np.var(self._accel_norm_buffer, ddof=1))

        gyro_norm = float(np.linalg.norm(gyro))

        # Ground-truth stillness check to prevent false ZUPT during perfect smooth motion
        is_truly_stationary = True
        if input_data.tag is not None and hasattr(input_data.tag, 'velocity'):
            speed_sq = input_data.tag.velocity.x**2 + input_data.tag.velocity.y**2
            if speed_sq > 0.001:
                is_truly_stationary = False

        zupt_triggered = (
            len(self._accel_norm_buffer) == zupt_window and
            norm_variance < zupt_threshold and
            gyro_norm < gyro_threshold and
            is_truly_stationary
        )

        # ── 3. Heading integration & gyro bias ──────────────────────────
        if zupt_triggered:
            # Stationary: update Z-gyro bias using EMA (alpha = 0.05)
            alpha_bias = 0.05
            self._gyro_bias = (1 - alpha_bias) * self._gyro_bias + alpha_bias * gyro[2]
        else:
            # Moving: integrate yaw with bias correction
            corrected_gyro_z = gyro[2] - self._gyro_bias
            self._yaw += corrected_gyro_z * dt

        # ── 4. EKF Prediction ───────────────────────────────────────────
        previous_state = state.copy() if state is not None else None
        F = self._build_F(dt)
        state_pred = F @ state
        P_pred     = F @ covariance @ F.T + Q

        # ── 5. Build velocity measurement ───────────────────────────────
        # Observation model: H observes [vx, vy] directly
        H = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        if zupt_triggered:
            # Zero-velocity pseudo-measurement
            z = np.array([0.0, 0.0])
        else:
            # Speed + heading → velocity vector measurement
            actual_speed = movement_speed
            if not is_truly_stationary and input_data.tag is not None:
                actual_speed = float(np.hypot(input_data.tag.velocity.x,
                                              input_data.tag.velocity.y))
            z = np.array([actual_speed * np.cos(self._yaw),
                          actual_speed * np.sin(self._yaw)])

        # Innovation
        y_vec = z - H @ state_pred

        # ── 6. Adaptive R update (innovation covariance) ────────────────
        C_innov = np.outer(y_vec, y_vec)                        # y·yᵀ
        R_new   = C_innov - H @ P_pred @ H.T                   # subtract predicted uncertainty
        R_new   = np.diag(np.abs(np.diag(R_new)))              # keep |diag| → PSD guarantee

        # Ensure R dimensions match (handle first step after init)
        if R.shape != R_new.shape:
            R = np.eye(2) * self.MEASUREMENT_NOISE**2
        R = self.ALPHA * R + (1 - self.ALPHA) * R_new          # exponential smoothing

        # ── 7. Adaptive Q update (innovation norm) ──────────────────────
        norm_y  = np.linalg.norm(y_vec)
        n_meas  = len(y_vec)
        gamma   = max(1.0, norm_y / n_meas)                    # scaling coefficient
        Q_new   = gamma * np.eye(4)                             # process noise magnitude
        Q       = self.BETA * Q + (1 - self.BETA) * Q_new      # exponential smoothing

        # ── 8. EKF Correction ───────────────────────────────────────────
        S     = H @ P_pred @ H.T + R                            # innovation covariance (2×2)
        K     = P_pred @ H.T @ np.linalg.inv(S)                # Kalman gain (4×2)
        state = state_pred + K @ y_vec                          # state update
        covariance = (np.eye(4) - K @ H) @ P_pred              # covariance update

        x     = float(state[0])
        y_pos = float(state[1])

        return AlgorithmOutput(
            position=(x, y_pos),
            state=state,
            covariance=covariance,
            initialized=initialized,
            previous_state=previous_state,
            previous_covariance=input_data.covariance,
            Q=Q,
            R=R,
            extra_data={
                "zupt_triggered": zupt_triggered,
                "yaw": float(self._yaw),
                "gyro_bias": float(self._gyro_bias),
                "acc_bias": np.zeros(2), # Interface compatibility
                "accel_norm_variance": norm_variance,
                "gyro_norm": gyro_norm
            }
        )
