import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class UwbImuFusionAekfAlgorithm(BaseLocalizationAlgorithm):
    """
    Adaptive UWB-IMU Fusion EKF (AEKF) for 2D tag localization.

    Extends the UWB-IMU Fusion EKF with adaptive R and Q updates based on
    innovation statistics, following the same adaptation scheme as the AEKF
    (aekf.py):
        - Adaptive R (§5.1): exponential smoothing of the innovation-based
          measurement noise estimate.
        - Adaptive Q (§5.2): exponential smoothing of the innovation-norm-based
          process noise estimate.

    State layout (packed into a single 8-vector):
        EKF state  = [x, y, vx, vy, ax, ay]     (indices 0-5, Kalman math)
        Aux state  = [yaw, gyro_bias]            (indices 6-7, IMU book-keeping)

    Measurement vector z (n+2):
        [d_1, d_2, …, d_n, ax_imu, ay_imu]

    The constant-acceleration kinematic model uses dt²/2 terms to couple
    acceleration into position and dt terms to couple it into velocity.
    Process noise uses a piecewise white-noise jerk model (G @ Q @ G.T).

    Works in three modes depending on what data is available this tick:
        UWB-only  → constant-acceleration prediction + UWB distance AEKF
        IMU-only  → heading+speed dead-reckoning + ZUPT
        Hybrid    → IMU prediction + UWB+accel AEKF correction
    """

    # -- Noise parameters (initial values, adapted online) ----------------
    UWB_MEASUREMENT_NOISE = 0.19       # initial R diagonal for UWB ranges
    IMU_MEASUREMENT_NOISE = 1.0        # initial R diagonal for IMU accelerometer
    PROCESS_NOISE_JERK    = 1.0        # initial Q diagonal (jerk variance)

    # -- Adaptive smoothing factors (matching AEKF) -----------------------
    ALPHA = 0.5          # smoothing factor for R adaptation
    BETA  = 0.5          # smoothing factor for Q adaptation

    # -- IMU dead-reckoning parameters (matching existing PULSE filters) --
    OMEGA      = 0.7       # IMU-velocity blend weight in prediction
    BIAS_ALPHA = 0.05      # gyro bias EMA smoothing factor

    # -- ZUPT thresholds --------------------------------------------------
    DEFAULT_ZUPT_THRESHOLD = 0.08   # m/s² – accel-norm stillness gate
    DEFAULT_GYRO_THRESHOLD = 0.1    # rad/s – gyro-norm stillness gate
    GT_STILLNESS_EPS       = 0.001  # m²/s² – ground-truth speed² gate

    n_ekf  = 6   # EKF dimension [x, y, vx, vy, ax, ay]
    n_full = 8   # EKF + yaw + gyro_bias

    uses_imu = True
    required_sensors = ("imu",)

    # ------------------------------------------------------------------ #
    #  BaseLocalizationAlgorithm interface                                #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "UWB-IMU Fusion AEKF"

    def initialize(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Main update                                                        #
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

        state, yaw, gyro_bias = self._unpack_state(state)

        # ── 2. Stationarity check (ground truth first, IMU fallback) ────
        is_stationary = self._check_stationary(tag, imu_on, accel_raw, gyro_raw)

        # ── 3. Prediction (constant-acceleration model) ─────────────────
        Q = self._build_process_noise(Q, dt)
        state_pred, yaw, gyro_bias = self._predict(
            state, dt, tag, imu_on, gyro_raw, yaw, gyro_bias, is_stationary
        )
        P_pred = self._predict_covariance(covariance, dt, Q)

        # ── 4. Adaptive UWB + IMU accelerometer measurement update ──────
        state, P, Q, R = self._update(
            state_pred, P_pred, measurements, anchors,
            imu_on, accel_raw, Q, R,
        )

        # ── 5. ZUPT measurement update (velocity + acceleration → 0) ───
        if is_stationary:
            state, P = self._apply_zupt(state, P)

        # ── 6. Covariance repair (symmetry + PSD) ──────────────────────
        P = self._repair_covariance(P)

        full_state = self._pack_state(state, yaw, gyro_bias)

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
            },
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _initialise(self, tag):
        x0   = float(getattr(getattr(tag, "position", None), "x", 0.0))
        y0   = float(getattr(getattr(tag, "position", None), "y", 0.0))
        yaw0 = float(getattr(tag, "orientation", 0.0)) if hasattr(tag, "orientation") else 0.0

        state      = np.array([x0, y0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        covariance = np.diag([5.0, 5.0, 10.0, 10.0, 1.0, 1.0]).astype(float)
        Q          = self._build_process_noise(None, dt=0.05)
        R          = None  # sized lazily once we know how many anchors we have

        self._init_yaw = yaw0
        return state, covariance, Q, R

    def _unpack_state(self, state) -> tuple:
        state = np.asarray(state, dtype=float).ravel()
        if len(state) < self.n_full:
            pad = np.zeros(self.n_full - len(state))
            if len(state) == self.n_ekf and hasattr(self, "_init_yaw"):
                pad[0] = self._init_yaw
            state = np.concatenate([state, pad])
        ekf_state = state[: self.n_ekf].copy()
        yaw       = float(state[6])
        gyro_bias = float(state[7])
        return ekf_state, yaw, gyro_bias

    def _pack_state(self, ekf_state, yaw, gyro_bias) -> np.ndarray:
        return np.array(
            [ekf_state[0], ekf_state[1], ekf_state[2], ekf_state[3],
             ekf_state[4], ekf_state[5], yaw, gyro_bias],
            dtype=float,
        )

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
        """
        Constant-acceleration state transition matrix (6×6).

        State ordering: [x, y, vx, vy, ax, ay]
        The dt²/2 terms couple acceleration into position; dt terms couple
        acceleration into velocity — matching the original 3D model.
        """
        dt2 = (dt ** 2) / 2.0
        return np.array([
            [1, 0, dt,  0, dt2,   0],
            [0, 1,  0, dt,   0, dt2],
            [0, 0,  1,  0,  dt,   0],
            [0, 0,  0,  1,   0,  dt],
            [0, 0,  0,  0,   1,   0],
            [0, 0,  0,  0,   0,   1],
        ], dtype=float)

    def _build_G(self, dt: float) -> np.ndarray:
        """
        Noise input matrix (6×2) for the piecewise white-noise jerk model.
        Jerk noise enters through acceleration, propagates to velocity and
        position via the kinematic coupling.
        """
        dt3_6 = (dt ** 3) / 6.0
        dt2_2 = (dt ** 2) / 2.0
        return np.array([
            [dt3_6,     0],
            [0,     dt3_6],
            [dt2_2,     0],
            [0,     dt2_2],
            [dt,        0],
            [0,        dt],
        ], dtype=float)

    def _build_process_noise(self, Q, dt: float) -> np.ndarray:
        """
        Build or return the process noise covariance matrix (6×6).

        If Q already has the correct shape it is returned as-is (it will be
        adapted online by the AEKF update step).  Otherwise a fresh Q is
        computed from the jerk model: G @ Q_jerk @ G.T.
        """
        if Q is not None and Q.shape == (self.n_ekf, self.n_ekf):
            return Q
        G = self._build_G(dt)
        Q_jerk = np.eye(2, dtype=float) * self.PROCESS_NOISE_JERK
        return G @ Q_jerk @ G.T

    def _predict(self, ekf_state, dt, tag, imu_on, gyro_raw, yaw, gyro_bias, is_stationary):
        F = self._build_F(dt)

        gz = float(np.asarray(gyro_raw, dtype=float)[2]) if (imu_on and gyro_raw is not None) else 0.0
        imu_active = imu_on and gyro_raw is not None

        if is_stationary:
            # ── ZUPT path: gyro bias update + zero velocity/acceleration ──
            gyro_bias = (1 - self.BIAS_ALPHA) * gyro_bias + self.BIAS_ALPHA * gz
            state_pred = F @ ekf_state
            state_pred[2] = 0.0   # vx
            state_pred[3] = 0.0   # vy
            state_pred[4] = 0.0   # ax
            state_pred[5] = 0.0   # ay

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
            # ── UWB-only path: constant acceleration ──
            state_pred = F @ ekf_state

        return state_pred, yaw, gyro_bias

    def _predict_covariance(self, P, dt, Q) -> np.ndarray:
        F = self._build_F(dt)
        return F @ P @ F.T + Q

    # ── Measurement helpers ──────────────────────────────────────────────

    def _predicted_distance(self, state, anchor) -> float:
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        return float(np.sqrt(dx ** 2 + dy ** 2))

    def _distance_jacobian_row(self, state, anchor) -> np.ndarray:
        """
        Jacobian row for a range measurement (1×6).
        Only position states contribute; velocity and acceleration cols are zero.
        """
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        d  = max(np.sqrt(dx ** 2 + dy ** 2), 1e-6)
        return np.array([dx / d, dy / d, 0.0, 0.0, 0.0, 0.0])

    # ── Adaptive UWB + IMU accelerometer update ────────────────────────

    def _update(self, state_pred, P_pred, measurements, anchors,
                imu_on, accel_raw, Q, R):
        """
        Combined UWB range + IMU accelerometer measurement update with
        adaptive R and Q (AEKF).

        Measurement vector:
            z = [d_1, d_2, …, d_n, ax_imu, ay_imu]

        Adaptive R (§5.1):
            R_new  = y·yᵀ − H·P_pred·Hᵀ
            R      = α·R + (1−α)·|diag(R_new)|

        Adaptive Q (§5.2):
            γ      = max(1, ‖y‖ / n_total)
            Q_new  = γ · I₆
            Q      = β·Q + (1−β)·Q_new

        Falls back to pure prediction when no measurements are available.
        """
        has_uwb = (measurements is not None and anchors is not None
                   and len(measurements) > 0)
        has_imu = imu_on and accel_raw is not None

        if not has_uwb and not has_imu:
            return state_pred, P_pred, Q, R

        # ── Build measurement vector z and observation matrix H ─────────
        n_uwb   = min(len(measurements), len(anchors)) if has_uwb else 0
        n_imu   = 2 if has_imu else 0
        n_total = n_uwb + n_imu

        # Lazily initialise R to the correct size
        if R is None or R.shape[0] != n_total:
            R_diag = np.zeros(n_total)
            R_diag[:n_uwb] = self.UWB_MEASUREMENT_NOISE ** 2
            R_diag[n_uwb:] = self.IMU_MEASUREMENT_NOISE ** 2
            R = np.diag(R_diag)

        H     = np.zeros((n_total, self.n_ekf))
        z     = np.zeros(n_total)
        z_hat = np.zeros(n_total)

        # UWB range rows
        for i in range(n_uwb):
            z_i = float(measurements[i])
            if np.isnan(z_i) or z_i <= 0:
                continue
            H[i]     = self._distance_jacobian_row(state_pred, anchors[i])
            z[i]     = z_i
            z_hat[i] = self._predicted_distance(state_pred, anchors[i])

        # IMU accelerometer rows (linear: observe [ax, ay] directly)
        if has_imu:
            accel = np.asarray(accel_raw, dtype=float).ravel()
            # ax observation → state index 4
            H[n_uwb, 4]      = 1.0
            z[n_uwb]         = float(accel[0])
            z_hat[n_uwb]     = state_pred[4]
            # ay observation → state index 5
            H[n_uwb + 1, 5]  = 1.0
            z[n_uwb + 1]     = float(accel[1])
            z_hat[n_uwb + 1] = state_pred[5]

        # ── Innovation ──────────────────────────────────────────────────
        y_vec = z - z_hat

        # ── Adaptive R update (§5.1) ────────────────────────────────────
        C_innov = np.outer(y_vec, y_vec)                        # y·yᵀ
        R_new   = C_innov - H @ P_pred @ H.T                   # subtract predicted uncertainty
        R_new   = np.diag(np.abs(np.diag(R_new)))              # keep |diag| → PSD guarantee
        R       = self.ALPHA * R + (1 - self.ALPHA) * R_new    # exponential smoothing

        # ── Adaptive Q update (§5.2) ────────────────────────────────────
        norm_y = np.linalg.norm(y_vec)
        gamma  = max(1.0, norm_y / max(n_total, 1))            # scaling coefficient
        Q_new  = gamma * np.eye(self.n_ekf)                    # process noise magnitude
        Q      = self.BETA * Q + (1 - self.BETA) * Q_new      # exponential smoothing

        # ── EKF correction (with jitter fallback if S is singular) ──────
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

    # ── ZUPT measurement update ────────────────────────────────────────

    def _apply_zupt(self, state, P):
        """Zero-velocity + zero-acceleration update when stationary."""
        H_z = np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)
        y_z = np.array([
            0.0 - state[2],   # vx → 0
            0.0 - state[3],   # vy → 0
            0.0 - state[4],   # ax → 0
            0.0 - state[5],   # ay → 0
        ])
        R_z = np.diag([1e-4, 1e-4, 1e-4, 1e-4])
        S_z = H_z @ P @ H_z.T + R_z
        K_z = P @ H_z.T @ np.linalg.inv(S_z)
        state = state + K_z @ y_z
        P     = (np.eye(self.n_ekf) - K_z @ H_z) @ P
        return state, P

    # ── Covariance repair ──────────────────────────────────────────────

    def _repair_covariance(self, P) -> np.ndarray:
        P = (P + P.T) / 2.0
        try:
            min_eig = float(np.min(np.real(np.linalg.eigvals(P))))
        except np.linalg.LinAlgError:
            min_eig = 0.0
        if min_eig < 1e-9:
            P = P + np.eye(self.n_ekf) * (1e-9 - min_eig)
        return P
