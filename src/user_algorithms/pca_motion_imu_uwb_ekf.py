import numpy as np
from collections import deque
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput

class PcaMotionImuUwbEkfAlgorithm(BaseLocalizationAlgorithm):
    """
    Fused IMU-UWB Adaptive EKF for 2D tag localization.
    
    This filter uses the exact same prediction, shadow filter, and update logic 
    as DutyCycledImuUwbAdaptiveEkfAlgorithm, but it replaces the time-based 
    duty cycle with a PCA motion detection algorithm. UWB is only fused when 
    the PCA algorithm confidently detects 'Linear Motion'.
    """

    PROCESS_NOISE_POS = 0.1
    PROCESS_NOISE_VEL = 1.0
    MEASUREMENT_NOISE = 0.15

    ALPHA = 0.5          
    BETA  = 0.5          
    OMEGA = 0.7           
    BIAS_ALPHA = 0.05     

    DEFAULT_ZUPT_THRESHOLD = 0.08   
    DEFAULT_GYRO_THRESHOLD = 0.1    
    GT_STILLNESS_EPS        = 0.001  

    n_ekf  = 4   
    n_full = 7   # EKF + yaw + gyro_bias + cycle_time (kept for compatibility)

    uses_imu = True
    required_sensors = ("imu",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Shadow IMU-only filter state (from duty-cycled logic)
        self._shadow_active = False
        self._shadow_state = None       
        self._shadow_P = None           
        self._shadow_yaw = 0.0
        self._shadow_gyro_bias = 0.0
        self._shadow_position = None    
        self._shadow_error = 0.0        

        # PCA detection state
        self.window_size = 50
        self.accel_window = deque(maxlen=self.window_size)
        self.smoothing_window_size = 20
        self.detection_history = deque(maxlen=self.smoothing_window_size)
        self.smoothed_state = "Unknown"
        self._prev_uwb_open = False

    @property
    def name(self) -> str:
        return "PCA Motion IMU-UWB EKF"

    def initialize(self) -> None:
        self.accel_window.clear()
        self.detection_history.clear()
        self.smoothed_state = "Unknown"
        self._prev_uwb_open = False

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

        # ── 2. PCA Motion Detection & Smoothing ──────────────────────────
        if imu_on and accel_raw is not None:
            self.accel_window.append(np.array(accel_raw, dtype=float))
            
        raw_detection = self._detect_motion_pca()
        
        if raw_detection != "Unknown":
            self.detection_history.append(raw_detection)
            
        if len(self.detection_history) > 0:
            counts = {}
            for d in self.detection_history:
                counts[d] = counts.get(d, 0) + 1
            self.smoothed_state = max(counts, key=counts.get)
        else:
            self.smoothed_state = "Unknown"

        # ── 3. Decide if UWB is gated this tick (PCA replaces duty cycle) ─
        prev_uwb_open = self._prev_uwb_open
        uwb_window_open = (self.smoothed_state == "Linear Motion")
        self._prev_uwb_open = uwb_window_open
        
        effective_measurements = measurements if uwb_window_open else None
        cycle_time += dt # Keep cycle time updated for state vector compatibility

        # ── 4. Stationarity check (ground truth first, IMU fallback) ────
        is_stationary = self._check_stationary(tag, imu_on, accel_raw, gyro_raw)

        # ── 5. Shadow IMU-only filter management ─────────────────────────
        if uwb_window_open and not prev_uwb_open:
            self._shadow_active = True
            self._shadow_state = state.copy()
            self._shadow_P = covariance.copy()
            self._shadow_yaw = yaw
            self._shadow_gyro_bias = gyro_bias
        elif not uwb_window_open and prev_uwb_open:
            self._shadow_active = False
            self._shadow_state = None
            self._shadow_P = None
            self._shadow_position = None
            self._shadow_error = 0.0

        # ── 6. Prediction (UWB-only / IMU-only / ZUPT path) ──────────────
        Q = self._build_Q(Q, dt)
        state_pred, yaw, gyro_bias = self._predict(
            state, dt, tag, imu_on, gyro_raw, yaw, gyro_bias, is_stationary
        )
        P_pred = self._predict_covariance(covariance, dt, Q)

        # ── 6b. Shadow filter: IMU-only prediction (no UWB corrections) ──
        if self._shadow_active and self._shadow_state is not None:
            shadow_Q = self._build_Q(None, dt)
            shadow_pred, self._shadow_yaw, self._shadow_gyro_bias = self._predict(
                self._shadow_state, dt, tag, imu_on, gyro_raw,
                self._shadow_yaw, self._shadow_gyro_bias, is_stationary
            )
            shadow_P_pred = self._predict_covariance(self._shadow_P, dt, shadow_Q)
            if is_stationary:
                shadow_pred, shadow_P_pred = self._apply_zupt(shadow_pred, shadow_P_pred)
            self._shadow_state = shadow_pred
            self._shadow_P = self._repair_covariance(shadow_P_pred)
            self._shadow_position = (float(shadow_pred[0]), float(shadow_pred[1]))
            if tag is not None and hasattr(tag, 'position'):
                gt_x = float(getattr(tag.position, 'x', 0.0))
                gt_y = float(getattr(tag.position, 'y', 0.0))
                self._shadow_error = float(np.sqrt(
                    (shadow_pred[0] - gt_x)**2 + (shadow_pred[1] - gt_y)**2
                ))

        # ── 7. UWB adaptive measurement update (gated by PCA window) ───────
        state, P, Q, R = self._update(state_pred, P_pred, effective_measurements, anchors, Q, R)

        # ── 8. ZUPT measurement update (velocity -> 0) ───────────────────
        if is_stationary:
            state, P = self._apply_zupt(state, P)

        # ── 9. Covariance repair (symmetry + PSD) ────────────────────────
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
                "shadow_imu_position": list(self._shadow_position) if self._shadow_position else None,
                "shadow_imu_error": float(self._shadow_error) if self._shadow_active else None,
                "pca_motion_state": self.smoothed_state
            },
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #
    
    def _detect_motion_pca(self) -> str:
        if len(self.accel_window) < self.window_size:
            return "Unknown"
            
        data = np.array(self.accel_window)
        data_centered = data - np.mean(data, axis=0)
        cov = np.cov(data_centered, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-9
        
        evals, _ = np.linalg.eigh(cov)
        evals = np.sort(evals)[::-1]
        total_var = np.sum(evals)
        
        if total_var < 1e-6:
            return "Linear Motion"
            
        ratios = evals / total_var
        if ratios[0] > 0.85:
            return "Linear Motion"
        elif (ratios[0] + ratios[1]) > 0.90:
            return "Circular/Curvilinear Motion"
        else:
            return "Random Walk"

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

    def _initialise(self, tag):
        x0   = float(getattr(getattr(tag, "position", None), "x", 0.0))
        y0   = float(getattr(getattr(tag, "position", None), "y", 0.0))
        yaw0 = float(getattr(tag, "orientation", 0.0)) if hasattr(tag, "orientation") else 0.0

        state      = np.array([x0, y0, 0.0, 0.0], dtype=float)
        covariance = np.diag([5.0, 5.0, 10.0, 10.0]).astype(float)
        Q          = self._build_Q(None, dt=0.05)
        R          = None  

        self._init_yaw = yaw0
        return state, covariance, Q, R

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
            gyro_bias = (1 - self.BIAS_ALPHA) * gyro_bias + self.BIAS_ALPHA * gz
            state_pred = F @ ekf_state
            state_pred[2] = 0.0
            state_pred[3] = 0.0

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

    def _predicted_distance(self, state, anchor) -> float:
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        return float(np.sqrt(dx**2 + dy**2))

    def _distance_jacobian_row(self, state, anchor) -> np.ndarray:
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        d  = max(np.sqrt(dx**2 + dy**2), 1e-6)
        return np.array([dx / d, dy / d, 0.0, 0.0])

    def _update(self, state_pred, P_pred, measurements, anchors, Q, R):
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

        C_innov = np.outer(y_vec, y_vec)
        R_new   = C_innov - H @ P_pred @ H.T
        R_new   = np.diag(np.abs(np.diag(R_new)))
        R       = self.ALPHA * R + (1 - self.ALPHA) * R_new

        norm_y = np.linalg.norm(y_vec)
        gamma  = max(1.0, norm_y / max(n, 1))
        Q_new  = gamma * np.eye(self.n_ekf)
        Q      = self.BETA * Q + (1 - self.BETA) * Q_new

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

    def _repair_covariance(self, P) -> np.ndarray:
        P = (P + P.T) / 2.0
        try:
            min_eig = float(np.min(np.real(np.linalg.eigvals(P))))
        except np.linalg.LinAlgError:
            min_eig = 0.0
        if min_eig < 1e-9:
            P = P + np.eye(self.n_ekf) * (1e-9 - min_eig)
        return P
