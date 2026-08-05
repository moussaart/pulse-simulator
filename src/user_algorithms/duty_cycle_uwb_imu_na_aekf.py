import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class DutyCycleUwbImuFusionNaAekfAlgorithm(BaseLocalizationAlgorithm):
    """
    NLOS-Aware Adaptive UWB-IMU Fusion EKF (NA-AEKF) for 2D tag localization.

    Extends the UWB-IMU Fusion AEKF with NLOS-gated inflation of the adaptive
    R term, following the same pattern as naaekf.py:
        - For any anchor flagged as NLOS, the freshly-estimated measurement-noise
          variance r_i is scaled up by LAMBDA_NLOS before exponential smoothing.
        - This down-weights NLOS measurements in the Kalman gain without
          discarding them outright.
        - IMU accelerometer rows are never inflated (IMU is always "LOS").

    State layout (packed into a single 8-vector):
        EKF state  = [x, y, vx, vy, ax, ay]     (indices 0-5, Kalman math)
        Aux state  = [yaw, gyro_bias]            (indices 6-7, IMU book-keeping)

    Measurement vector z (n+2):
        [d_1, d_2, …, d_n, ax_imu, ay_imu]

    Works in three modes depending on what data is available this tick:
        UWB-only  → constant-acceleration prediction + UWB distance NA-AEKF
        IMU-only  → heading+speed dead-reckoning + ZUPT
        Hybrid    → IMU prediction + UWB+accel NA-AEKF correction
    """

    # -- Noise parameters (initial values, adapted online) ----------------
    UWB_MEASUREMENT_NOISE = 0.19       # initial R diagonal for UWB ranges
    IMU_MEASUREMENT_NOISE = 1.0        # initial R diagonal for IMU accelerometer
    PROCESS_NOISE_JERK    = 1.0        # initial Q diagonal (jerk variance)

    # -- Adaptive smoothing factors ---------------------------------------
    ALPHA = 0.3          # smoothing factor for R adaptation (matching naaekf)
    BETA  = 0.5          # smoothing factor for Q adaptation

    # -- NLOS gating ------------------------------------------------------
    LAMBDA_NLOS = 10.0    # inflation factor for r_i when anchor i is NLOS

    # -- IMU dead-reckoning parameters ------------------------------------
    OMEGA      = 0.7       # IMU-velocity blend weight in prediction
    BIAS_ALPHA = 0.05      # gyro bias EMA smoothing factor

    # -- ZUPT thresholds --------------------------------------------------
    DEFAULT_ZUPT_THRESHOLD = 0.08   # m/s² – accel-norm stillness gate
    DEFAULT_GYRO_THRESHOLD = 0.1    # rad/s – gyro-norm stillness gate
    GT_STILLNESS_EPS       = 0.001  # m²/s² – ground-truth speed² gate

    # -- Duty-cycled UWB scheduling (Defaults, can be overridden dynamically) --
    IMU_ONLY_DURATION = 3   # seconds
    HYBRID_DURATION    = 1   # seconds
    DUTY_CYCLE_PERIOD  = IMU_ONLY_DURATION + HYBRID_DURATION

    n_ekf  = 6   # EKF dimension [x, y, vx, vy, ax, ay]
    n_full = 8   # EKF + yaw + gyro_bias

    uses_imu = True
    required_sensors = ("imu",)

    # ------------------------------------------------------------------ #
    #  BaseLocalizationAlgorithm interface                                #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "Duty-Cycled UWB-IMU NA-AEKF"

    def initialize(self) -> None:
        # Persistent duty-cycle clock — survives across every update() call.
        # Only reset here (filter (re)initialisation), never inside update().
        self._elapsed_time = 0.0

        # Persistent, independently-smoothed adaptive R blocks. Keeping these
        # separate (instead of one combined R sized to n_total) means the
        # UWB noise estimate isn't wiped out every time UWB gets paused
        # during an IMU-only window — it just stops updating and resumes
        # from where it left off.
        self._R_uwb_diag = None   # shape (n_anchors_last_seen,)
        self._R_imu_diag = None   # shape (2,)

        # Dynamic cycle length
        self._cycle_length = self.DUTY_CYCLE_PERIOD
        self._active_window = self.HYBRID_DURATION

    def set_duty_cycle(self, cycle_length: float, active_window: float) -> None:
        """
        Dynamically override the duty cycle parameters from an RL agent.
        """
        self._cycle_length = float(cycle_length)
        self._active_window = float(active_window)

    def _uwb_active_this_tick(self, dt: float) -> bool:
        """
        Advance the duty-cycle clock and decide whether UWB is enabled
        for the current tick. First IMU_ONLY_DURATION seconds of each
        DUTY_CYCLE_PERIOD-second cycle -> IMU-only; remainder -> hybrid.
        """
        self._elapsed_time += float(dt)
        phase = self._elapsed_time % max(self._cycle_length, 1e-6)
        imu_duration = max(0.0, self._cycle_length - self._active_window)
        return phase >= imu_duration

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
        # NLOS status (0=LOS, 1=NLOS)
        is_los       = input_data.is_los

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

        # ── 4. NLOS-aware adaptive UWB + IMU measurement update ─────────
        # Duty-cycle gate: decides whether UWB is allowed into the
        # correction step this tick. State/covariance are untouched by
        # this decision — only which measurement blocks are assembled.
        uwb_enabled = self._uwb_active_this_tick(dt)
        state, P, Q, R = self._update(
            state_pred, P_pred, measurements, anchors,
            imu_on, accel_raw, Q, R, is_los, uwb_enabled,
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
                "uwb_window_open": uwb_enabled,
                "t_imu": float(self._cycle_length),
                "t_uwb": float(self._active_window),
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
        if tag is not None and hasattr(tag, "velocity"):
            speed_sq = tag.velocity.x ** 2 + tag.velocity.y ** 2
            return speed_sq < self.GT_STILLNESS_EPS

        if imu_on and accel_raw is not None and gyro_raw is not None:
            accel = np.asarray(accel_raw, dtype=float)
            gyro  = np.asarray(gyro_raw, dtype=float)
            acc_norm  = float(np.linalg.norm(accel))
            gyro_norm = float(np.linalg.norm(gyro))
            return acc_norm < self.DEFAULT_ZUPT_THRESHOLD and gyro_norm < self.DEFAULT_GYRO_THRESHOLD

        return False

    # ── Prediction ──────────────────────────────────────────────────────

    def _build_F(self, dt: float) -> np.ndarray:
        """Constant-acceleration state transition matrix (6×6)."""
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
        """Noise input matrix (6×2) for the piecewise white-noise jerk model."""
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
        """Build or return the process noise covariance matrix (6×6)."""
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
            gyro_bias = (1 - self.BIAS_ALPHA) * gyro_bias + self.BIAS_ALPHA * gz
            state_pred = F @ ekf_state
            state_pred[2] = 0.0
            state_pred[3] = 0.0
            state_pred[4] = 0.0
            state_pred[5] = 0.0

        elif imu_active:
            corrected_gz = gz - gyro_bias
            yaw += corrected_gz * dt

            actual_speed = 1.0
            if tag is not None and hasattr(tag, "velocity"):
                actual_speed = float(np.hypot(tag.velocity.x, tag.velocity.y))

            vx_imu = actual_speed * np.cos(yaw)
            vy_imu = actual_speed * np.sin(yaw)

            state_pred = F @ ekf_state
            state_pred[2] = (1 - self.OMEGA) * state_pred[2] + self.OMEGA * vx_imu
            state_pred[3] = (1 - self.OMEGA) * state_pred[3] + self.OMEGA * vy_imu

        else:
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
        """Jacobian row for a range measurement (1×6)."""
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        d  = max(np.sqrt(dx ** 2 + dy ** 2), 1e-6)
        return np.array([dx / d, dy / d, 0.0, 0.0, 0.0, 0.0])

    # ── NLOS gating helper ───────────────────────────────────────────────

    def _nlos_mask(self, is_los, n) -> np.ndarray:
        """
        Returns a boolean array of length n, True where the measurement is NLOS.

        `is_los` follows the convention: 0 = LOS, 1 = NLOS, despite the name.
        If missing or malformed, every measurement is treated as LOS so the
        filter degrades gracefully to the Fusion AEKF behaviour.
        """
        if is_los is None:
            return np.zeros(n, dtype=bool)
        flags = np.asarray(is_los).reshape(-1)
        if flags.shape[0] != n:
            return np.zeros(n, dtype=bool)
        return flags.astype(bool)

    # ── NLOS-aware adaptive UWB + IMU accelerometer update ─────────────

    def _update(self, state_pred, P_pred, measurements, anchors,
                imu_on, accel_raw, Q, R, is_los, uwb_enabled=True):
        """
        Combined UWB range + IMU accelerometer measurement update with
        NLOS-aware adaptive R and adaptive Q.

        Extends the Fusion AEKF _update with per-anchor NLOS gating:
          - Computes r_i,new per measurement from innovation statistics
          - Inflates r_i,new by LAMBDA_NLOS for NLOS-flagged UWB anchors
          - IMU accelerometer rows are never inflated (always trusted)
          - Smooths R via exponential averaging
          - Adapts Q from innovation norm (unchanged from AEKF)

        `uwb_enabled` implements the IMU/UWB duty cycle: when False, the
        UWB block is dropped from H/z/R entirely for this tick, even if
        `measurements`/`anchors` are non-empty — the correction step then
        uses IMU-only rows. State and covariance are otherwise handled
        identically in both modes; nothing is reinitialised on a mode flip.

        Falls back to pure prediction when no measurements are available.
        """
        has_uwb = (uwb_enabled and measurements is not None
                   and anchors is not None and len(measurements) > 0)
        has_imu = imu_on and accel_raw is not None

        if not has_uwb and not has_imu:
            return state_pred, P_pred, Q, R

        # ── Build measurement vector z and observation matrix H ─────────
        n_uwb   = min(len(measurements), len(anchors)) if has_uwb else 0
        n_imu   = 2 if has_imu else 0
        n_total = n_uwb + n_imu

        # NLOS mask for UWB anchors only
        nlos_mask_uwb = self._nlos_mask(is_los, n_uwb)

        # ── Assemble R from persistent, independently-smoothed blocks ───
        # (see initialize(): self._R_uwb_diag / self._R_imu_diag). This
        # avoids wiping out the learned UWB noise estimate every time the
        # duty cycle pauses/resumes UWB — only the active block(s) are
        # touched below; the inactive block is carried over untouched.
        if self._R_uwb_diag is None or self._R_uwb_diag.shape[0] != n_uwb:
            self._R_uwb_diag = np.full(n_uwb, self.UWB_MEASUREMENT_NOISE ** 2)
        if self._R_imu_diag is None:
            self._R_imu_diag = np.full(2, self.IMU_MEASUREMENT_NOISE ** 2)

        R_diag = np.concatenate([
            self._R_uwb_diag if n_uwb > 0 else np.zeros(0),
            self._R_imu_diag if n_imu > 0 else np.zeros(0),
        ])
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
            H[n_uwb, 4]      = 1.0
            z[n_uwb]         = float(accel[0])
            z_hat[n_uwb]     = state_pred[4]
            H[n_uwb + 1, 5]  = 1.0
            z[n_uwb + 1]     = float(accel[1])
            z_hat[n_uwb + 1] = state_pred[5]

        # ── Innovation ──────────────────────────────────────────────────
        y_vec = z - z_hat

        # ── Adaptive R update with NLOS gating ──────────────────────────
        C_innov  = np.outer(y_vec, y_vec)                                   # y·yᵀ
        diag_new = np.abs(np.diag(C_innov) - np.diag(H @ P_pred @ H.T))   # r_i,new per measurement

        # Per-anchor NLOS inflation: r_i,new *= LAMBDA_NLOS for NLOS UWB anchors
        # IMU rows (indices n_uwb:) are never inflated
        diag_new[:n_uwb] = np.where(
            nlos_mask_uwb, self.LAMBDA_NLOS * diag_new[:n_uwb], diag_new[:n_uwb]
        )

        R_new = np.diag(diag_new)                                          # PSD by construction
        R     = self.ALPHA * R + (1 - self.ALPHA) * R_new                  # exponential smoothing

        # Persist the smoothed values back into the per-block store so the
        # next tick (possibly after a duty-cycle mode flip) picks up from
        # here instead of an out-of-date or reset value.
        new_diag = np.diag(R)
        if n_uwb > 0:
            self._R_uwb_diag = new_diag[:n_uwb].copy()
        if n_imu > 0:
            self._R_imu_diag = new_diag[n_uwb:].copy()

        # ── Adaptive Q update (§5.2) ────────────────────────────────────
        norm_y = np.linalg.norm(y_vec)
        gamma  = max(1.0, norm_y / max(n_total, 1))
        Q_new  = gamma * np.eye(self.n_ekf)
        Q      = self.BETA * Q + (1 - self.BETA) * Q_new

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
            0.0 - state[2],
            0.0 - state[3],
            0.0 - state[4],
            0.0 - state[5],
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
