import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class DutyCycledImuUwbAdaptiveEkfAlgorithm(BaseLocalizationAlgorithm):
    """
    Duty-Cycled Fused IMU-UWB Adaptive EKF for 2D tag localization.

    State layout (packed into a single 7-vector, mirroring duty_cycled_imu_uwb_aekf):
        EKF state  = [x, y, vx, vy]                    (indices 0-3, used by Kalman math)
        Aux state  = [yaw, gyro_bias, cycle_time]       (indices 4-6, IMU + duty-cycle book-keeping)

    Combines:
      - IMU Speed Dead Reckoning: heading integration, speed propagation, ZUPT
      - Adaptive EKF: UWB distance updates with adaptive R and Q
      - Duty cycling: UWB measurements are only consumed during a configurable
        active window within each cycle, IMU prediction runs continuously

    Within each cycle (default 4.0s total), UWB is only fused during the
    final active_window seconds (default 1.0s) -- e.g. with the defaults,
    seconds [0, 3) run IMU-only dead-reckoning, seconds [3, 4) fuse UWB.
    Both the cycle length and the active window are configurable via the
    constructor so they can be tuned without subclassing.

    Works in three prediction modes depending on what IMU data is available
    this tick, independent of the duty-cycle gate:
        UWB-only  -> constant-velocity prediction + UWB distance AEKF
        IMU-only  -> heading+speed dead-reckoning + ZUPT
        Hybrid    -> IMU prediction + UWB AEKF correction (only inside the window)

    NOTE: this port assumes IMU data arrives the same way it does in
    ImuspeeddeadreckoningalgorithmAlgorithm -- as input_data.accel /
    input_data.gyro, gated by input_data.imu_data_on -- rather than nested
    under input_data.tag.imu_data as in the original static method. If the
    real AlgorithmInput still nests IMU under the tag, this needs adjusting.
    """

    PROCESS_NOISE_POS = 0.1
    PROCESS_NOISE_VEL = 1.0
    MEASUREMENT_NOISE = 0.15

    ALPHA = 0.5          # smoothing factor for R
    BETA  = 0.5          # smoothing factor for Q
    OMEGA = 0.7           # IMU-velocity blend weight in prediction
    BIAS_ALPHA = 0.05     # gyro bias EMA smoothing factor

    DEFAULT_ZUPT_THRESHOLD = 0.08   # m/s² – accel-norm stillness gate (IMU fallback)
    DEFAULT_GYRO_THRESHOLD = 0.1    # rad/s – gyro-norm stillness gate (IMU fallback)
    GT_STILLNESS_EPS        = 0.001  # m²/s² – ground-truth speed² stillness gate

    DEFAULT_CYCLE_LENGTH  = 3.0   # s – total duty-cycle period
    DEFAULT_ACTIVE_WINDOW = 1.0   # s – portion of the cycle (at the end) where UWB is fused

    n_ekf  = 4   # EKF dimension
    n_full = 7   # EKF + yaw + gyro_bias + cycle_time

    uses_imu = True
    required_sensors = ("imu",)

    def __init__(self, *args, cycle_length: float = None, active_window: float = None, **kwargs):
        """
        cycle_length  : total duty-cycle period in seconds (default 4.0, matches
                        the original duty_cycled_imu_uwb_aekf).
        active_window : seconds at the END of each cycle during which UWB
                        measurements are fused (default 1.0). Must be <= cycle_length.

        Both default to None here rather than the class constants directly so
        that *args/**kwargs from a no-arg factory/registry instantiation
        (e.g. AlgorithmClass()) still works unchanged -- this constructor is
        purely additive on top of whatever BaseLocalizationAlgorithm expects.
        """
        super().__init__(*args, **kwargs)
        self.cycle_length  = float(cycle_length) if cycle_length is not None else self.DEFAULT_CYCLE_LENGTH
        self.active_window = float(active_window) if active_window is not None else self.DEFAULT_ACTIVE_WINDOW
        if self.active_window > self.cycle_length:
            raise ValueError(
                f"active_window ({self.active_window}) cannot exceed cycle_length ({self.cycle_length})"
            )

    # ------------------------------------------------------------------ #
    #  BaseLocalizationAlgorithm interface                                #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "Duty-Cycled IMU-UWB Adaptive EKF"

    def initialize(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        measurements = input_data.measurements
        anchors      = input_data.anchors
        dt           = input_data.dt
        imu_on       = input_data.imu_data_on
        accel_raw    = input_data.accel
        gyro_raw     = input_data.gyro
        tag          = input_data.tag

        state       = input_data.state
        covariance  = input_data.covariance
        Q           = input_data.Q
        R           = input_data.R
        initialized = input_data.initialized

        # ── 1. Initialisation ───────────────────────────────────────────
        if not initialized or state is None or covariance is None:
            state, covariance, Q, R = self._initialise(tag)
            initialized = True

        state, yaw, gyro_bias, cycle_time = self._unpack_state(state)

        # ── 2. Advance duty-cycle timer, decide if UWB is gated this tick ─
        cycle_time = self._advance_cycle(cycle_time, dt)
        uwb_window_open = self._in_active_window(cycle_time)
        effective_measurements = measurements if uwb_window_open else None

        # ── 3. Stationarity check (ground truth first, IMU fallback) ────
        is_stationary = self._check_stationary(tag, imu_on, accel_raw, gyro_raw)

        # ── 4. Prediction (UWB-only / IMU-only / ZUPT path) ──────────────
        Q = self._build_Q(Q, dt)
        state_pred, yaw, gyro_bias = self._predict(
            state, dt, tag, imu_on, gyro_raw, yaw, gyro_bias, is_stationary
        )
        P_pred = self._predict_covariance(covariance, dt, Q)

        # ── 5. UWB adaptive measurement update (only inside the duty-cycle window) ─
        state, P, Q, R = self._update(state_pred, P_pred, effective_measurements, anchors, Q, R)

        # ── 6. ZUPT measurement update (velocity -> 0) ───────────────────
        if is_stationary:
            state, P = self._apply_zupt(state, P)

        # ── 7. Covariance repair (symmetry + PSD) ────────────────────────
        P = self._repair_covariance(P)

        full_state = self._pack_state(state, yaw, gyro_bias, cycle_time)

        return AlgorithmOutput(
            position=(float(state[0]), float(state[1])),
            state=full_state,
            covariance=P,
            initialized=initialized,
            Q=Q,
            R=R,
            extra_data={
                "zupt_triggered": is_stationary,
                "yaw": float(yaw),
                "gyro_bias": float(gyro_bias),
                "cycle_time": float(cycle_time),
                "uwb_window_open": uwb_window_open,
            },
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _unpack_state(self, state) -> tuple:
        state = np.asarray(state, dtype=float).ravel()
        if len(state) < self.n_full:
            pad = np.zeros(self.n_full - len(state))
            if len(state) == self.n_ekf and hasattr(self, "_init_yaw"):
                pad[0] = self._init_yaw
            state = np.concatenate([state, pad])
        ekf_state  = state[: self.n_ekf].copy()
        yaw        = float(state[4])
        gyro_bias  = float(state[5])
        cycle_time = float(state[6])
        return ekf_state, yaw, gyro_bias, cycle_time

    def _pack_state(self, ekf_state, yaw, gyro_bias, cycle_time) -> np.ndarray:
        return np.array(
            [ekf_state[0], ekf_state[1], ekf_state[2], ekf_state[3], yaw, gyro_bias, cycle_time],
            dtype=float,
        )

    # ── Duty cycle ─────────────────────────────────────────────────────

    def _advance_cycle(self, cycle_time: float, dt: float) -> float:
        """
        Advance the duty-cycle timer by dt, wrapping at self.cycle_length.

        Mirrors doc 5's section 0 timer logic exactly, but against the
        configurable self.cycle_length instead of a hardcoded 4.0.
        """
        cycle_time += dt
        if cycle_time >= self.cycle_length:
            cycle_time = 0.0
        return cycle_time

    def _in_active_window(self, cycle_time: float) -> bool:
        """
        True when this tick falls inside the UWB-active portion of the
        cycle. The active window is the final self.active_window seconds
        of each self.cycle_length-second cycle, matching doc 5's
        `3.0 <= cycle_time < 4.0` gate generalised to configurable bounds.
        """
        window_start = self.cycle_length - self.active_window
        return window_start <= cycle_time < self.cycle_length

    def _initialise(self, tag):
        x0   = float(getattr(getattr(tag, "position", None), "x", 0.0))
        y0   = float(getattr(getattr(tag, "position", None), "y", 0.0))
        yaw0 = float(getattr(tag, "orientation", 0.0)) if hasattr(tag, "orientation") else 0.0

        state      = np.array([x0, y0, 0.0, 0.0], dtype=float)
        covariance = np.diag([5.0, 5.0, 10.0, 10.0]).astype(float)
        Q          = self._build_Q(None, dt=0.05)
        R          = None  # sized lazily once we know how many anchors we have

        # yaw/gyro_bias/cycle_time are stashed via the packed state, not
        # separate fields, so initial yaw needs to ride along on the first
        # _pack_state call; cycle_time starts at 0.0 (handled by the zero-pad
        # in _unpack_state, no extra bookkeeping needed here)
        self._init_yaw = yaw0
        return state, covariance, Q, R

    # ── Stationarity ───────────────────────────────────────────────────

    def _check_stationary(self, tag, imu_on, accel_raw, gyro_raw) -> bool:
        # Primary: ground-truth velocity (available in simulation)
        if tag is not None and hasattr(tag, "velocity"):
            speed_sq = tag.velocity.x ** 2 + tag.velocity.y ** 2
            return speed_sq < self.GT_STILLNESS_EPS

        # Fallback: raw IMU thresholds when ground truth isn't available
        if imu_on and accel_raw is not None and gyro_raw is not None:
            accel = np.asarray(accel_raw, dtype=float)
            gyro  = np.asarray(gyro_raw, dtype=float)
            acc_norm  = float(np.linalg.norm(accel))
            gyro_norm = float(np.linalg.norm(gyro))
            return acc_norm < self.DEFAULT_ZUPT_THRESHOLD and gyro_norm < self.DEFAULT_GYRO_THRESHOLD

        return False

    # ── Prediction ──────────────────────────────────────────────────────

    def _build_F(self, dt: float) -> np.ndarray:
        return np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

    def _build_Q(self, Q, dt: float) -> np.ndarray:
        if Q is not None and Q.shape == (self.n_ekf, self.n_ekf):
            return Q
        sp = self.PROCESS_NOISE_POS
        sv = self.PROCESS_NOISE_VEL
        q_1d = np.array([
            [dt**4 / 4 * sp**2,    dt**3 / 2 * sp * sv],
            [dt**3 / 2 * sp * sv,  dt**2 * sv**2],
        ])
        Q = np.zeros((self.n_ekf, self.n_ekf))
        Q[np.ix_([0, 2], [0, 2])] = q_1d
        Q[np.ix_([1, 3], [1, 3])] = q_1d
        return Q

    def _predict(self, ekf_state, dt, tag, imu_on, gyro_raw, yaw, gyro_bias, is_stationary):
        F = self._build_F(dt)

        gz = float(np.asarray(gyro_raw, dtype=float)[2]) if (imu_on and gyro_raw is not None) else 0.0
        imu_active = imu_on and gyro_raw is not None

        if is_stationary:
            # ── ZUPT path: gyro bias update + zero velocity ──
            gyro_bias = (1 - self.BIAS_ALPHA) * gyro_bias + self.BIAS_ALPHA * gz
            state_pred = F @ ekf_state
            state_pred[2] = 0.0
            state_pred[3] = 0.0

        elif imu_active:
            # ── IMU dead-reckoning path ──
            corrected_gz = gz - gyro_bias
            yaw += corrected_gz * dt

            actual_speed = 1.0
            if tag is not None and hasattr(tag, "velocity"):
                actual_speed = float(np.hypot(tag.velocity.x, tag.velocity.y))

            vx_imu = actual_speed * np.cos(yaw)
            vy_imu = actual_speed * np.sin(yaw)

            state_pred = F @ ekf_state
            # Blend: trust IMU velocity heavily
            state_pred[2] = (1 - self.OMEGA) * state_pred[2] + self.OMEGA * vx_imu
            state_pred[3] = (1 - self.OMEGA) * state_pred[3] + self.OMEGA * vy_imu

        else:
            # ── UWB-only path: constant velocity ──
            state_pred = F @ ekf_state

        return state_pred, yaw, gyro_bias

    def _predict_covariance(self, P, dt, Q) -> np.ndarray:
        F = self._build_F(dt)
        return F @ P @ F.T + Q

    # ── Measurement helpers ──────────────────────────────────────────────

    def _predicted_distance(self, state, anchor) -> float:
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        return float(np.sqrt(dx**2 + dy**2))

    def _distance_jacobian_row(self, state, anchor) -> np.ndarray:
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        d  = max(np.sqrt(dx**2 + dy**2), 1e-6)
        return np.array([dx / d, dy / d, 0.0, 0.0])

    # ── Adaptive UWB update ────────────────────────────────────────────────

    def _update(self, state_pred, P_pred, measurements, anchors, Q, R):
        """
        Joint adaptive UWB update:
          - Builds full H and innovation vector y
          - Adapts R  (AEKF §5.1)
          - Adapts Q  (AEKF §5.2)
          - Applies standard EKF correction

        Falls back to pure prediction (no correction) when no UWB
        measurements are available this tick.
        """
        has_uwb = measurements is not None and anchors is not None and len(measurements) > 0
        if not has_uwb:
            return state_pred, P_pred, Q, R

        n = min(len(measurements), len(anchors))

        if R is None or R.shape[0] != n:
            R = np.eye(n, dtype=float) * self.MEASUREMENT_NOISE**2

        H     = np.zeros((n, self.n_ekf))
        y_vec = np.zeros(n)

        for i in range(n):
            z = float(measurements[i])
            if np.isnan(z) or z <= 0:
                continue
            H[i]     = self._distance_jacobian_row(state_pred, anchors[i])
            y_vec[i] = z - self._predicted_distance(state_pred, anchors[i])

        # ── Adaptive R (AEKF §5.1) ────────────────────────────────────────
        C_innov = np.outer(y_vec, y_vec)
        R_new   = C_innov - H @ P_pred @ H.T
        R_new   = np.diag(np.abs(np.diag(R_new)))
        R       = self.ALPHA * R + (1 - self.ALPHA) * R_new

        # ── Adaptive Q (AEKF §5.2) ────────────────────────────────────────
        norm_y = np.linalg.norm(y_vec)
        gamma  = max(1.0, norm_y / max(n, 1))
        Q_new  = gamma * np.eye(self.n_ekf)
        Q      = self.BETA * Q + (1 - self.BETA) * Q_new

        # ── EKF correction (with jitter fallback if S is singular) ───────
        S = H @ P_pred @ H.T + R
        S = (S + S.T) / 2.0
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            jitter = max(1e-6, 1e-3 * np.trace(S) / max(S.shape[0], 1))
            S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * jitter)

        K     = P_pred @ H.T @ S_inv
        state = state_pred + K @ y_vec
        P     = (np.eye(self.n_ekf) - K @ H) @ P_pred

        return state, P, Q, R

    # ── ZUPT measurement update ────────────────────────────────────────────

    def _apply_zupt(self, state, P):
        H_z = np.array([[0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)
        y_z = np.array([0.0 - state[2], 0.0 - state[3]])
        R_z = np.diag([1e-4, 1e-4])
        S_z = H_z @ P @ H_z.T + R_z
        K_z = P @ H_z.T @ np.linalg.inv(S_z)
        state = state + K_z @ y_z
        P     = (np.eye(self.n_ekf) - K_z @ H_z) @ P
        return state, P

    # ── Covariance repair ──────────────────────────────────────────────────

    def _repair_covariance(self, P) -> np.ndarray:
        P = (P + P.T) / 2.0
        try:
            min_eig = float(np.min(np.real(np.linalg.eigvals(P))))
        except np.linalg.LinAlgError:
            min_eig = 0.0
        if min_eig < 1e-9:
            P = P + np.eye(self.n_ekf) * (1e-9 - min_eig)
        return P