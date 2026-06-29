import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.core.uwb.uwb_devices import Tag, Position
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from src.core.parallel.gpu_backend import get_array_module, to_cpu
from src.core.parallel.parallel_utils import vectorized_jacobian



class LocalizationAlgorthimes():
    
    
    Localization_algorthimes = ["Trilateration", "NLOS-Aware AEKF", "Improved Adaptive EKF", 
                                "IMU Only", "IMU-UWB AEKF"]
    
    
    @staticmethod
    def _cv_process_noise(dt: float, accel_variance: float = 0.1) -> np.ndarray:
        """
        Constant-Velocity (CV) continuous white-acceleration model process noise.

        Q = sigma_a^2 * [[dt^4/4,        0, dt^3/2,        0],
                         [       0, dt^4/4,        0, dt^3/2],
                         [dt^3/2,        0,    dt^2,        0],
                         [       0, dt^3/2,        0,    dt^2]]

        accel_variance is sigma_a^2.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = accel_variance
        Q = np.array([
            [dt4/4.0,      0.0, dt3/2.0,      0.0],
            [     0.0, dt4/4.0,      0.0, dt3/2.0],
            [dt3/2.0,      0.0,     dt2,      0.0],
            [     0.0, dt3/2.0,      0.0,     dt2]
        ], dtype=float) * q
        return Q

    @staticmethod
    def trilateration(measurements, anchors):
        # Simple trilateration implementation
        # This is a basic implementation and could be improved
        if len(measurements) < 3:
            return (0, 0)
            
        # Use first three anchors for basic trilateration
        p1 = (anchors[0].position.x, anchors[0].position.y)
        p2 = (anchors[1].position.x, anchors[1].position.y)
        p3 = (anchors[2].position.x, anchors[2].position.y)
        r1, r2, r3 = measurements[0], measurements[1], measurements[2]
        
        # Basic trilateration calculation
        A = 2 * np.array([
            [p2[0] - p1[0], p2[1] - p1[1]],
            [p3[0] - p1[0], p3[1] - p1[1]]
        ])
        
        b = np.array([
            [r1**2 - r2**2 - p1[0]**2 + p2[0]**2 - p1[1]**2 + p2[1]**2],
            [r1**2 - r3**2 - p1[0]**2 + p3[0]**2 - p1[1]**2 + p3[1]**2]
        ])
        
        try:
            x = np.linalg.solve(A, b)
            x_flat = x.flatten()
            return (float(x_flat[0]), float(x_flat[1]))
        except np.linalg.LinAlgError:
            return (0, 0)

    @staticmethod
    def Nlos_aware_aekf(measurements, tag, anchors, aekf_state, aekf_P, aekf_initialized, 
                        is_los, alpha = 0.3, beta = 2.0, nlos_factor = 10.0, dt=0.05, 
                        imu_data_on=False, u=None, Q=None, R=None):
        """
        LOS-Aware Adaptive Extended Kalman Filter implementation
        Args:
            measurements: Distance measurements
            tag: Tag object
            anchors: List of anchor objects
            aekf_state: Current state estimate
            aekf_P: Current covariance matrix
            aekf_initialized: Boolean indicating if filter is initialized
            is_los: List of booleans (0 for LOS, 1 for NLOS)
            alpha: Smoothing factor for R adaptation (default 0.3)
            beta: Smoothing factor for Q adaptation (default 2.0)
            nlos_factor: Scaling factor for NLOS measurements (default 10.0)
            dt: Time step
            imu_data_on: If True, includes acceleration control input u(t)
            u: Control input vector [ax, ay] representing acceleration in x and y directions
            Q: Initial process noise covariance matrix (optional)
            R: Initial measurement noise covariance matrix (optional)
            i: Iteration counter
        """
        if not aekf_initialized:
            # Start from origin by request
            aekf_state = np.array([0.0, 0.0, 0.0, 0.0])
            aekf_P = np.eye(4) * 1.0
            aekf_initialized = True
        
        # Prediction step
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        
        # Control input matrix B and acceleration u(t)
        if imu_data_on and u is not None:
            B = np.array([[0.5*dt**2, 0],
                         [0, 0.5*dt**2],
                         [dt, 0],
                         [0, dt]])
            aekf_state = F @ aekf_state + B @ u
        else:
            aekf_state = F @ aekf_state
            
        # Use provided Q or default CV model
        if Q is None:
            Q = LocalizationAlgorthimes._cv_process_noise(dt, accel_variance=0.1)
        aekf_P = F @ aekf_P @ F.T + Q
        
        # Update step
        if len(measurements) > 0:
            z = np.array(measurements)
            
            # Vectorized Jacobian (GPU-accelerated when CuPy is available)
            anchor_positions = np.array([[a.position.x, a.position.y]
                                         for a in anchors[:len(measurements)]])
            H, h = vectorized_jacobian(aekf_state, anchor_positions)
            H = to_cpu(H)
            h = to_cpu(h)
            
            # Use provided R or default; reset if size changed (anchor count changed)
            if R is None or R.shape[0] != len(measurements):
                r_scale = float(np.mean(np.diag(R))) if R is not None and R.size > 0 else 0.1
                R = np.eye(len(measurements), dtype=float) * r_scale
            
            # Innovation sequence
            innovation = z - h
            
            # Adapt R and Q immediately but with smoothing and bounds
            innovation_cov = np.outer(innovation, innovation)
            R_new = innovation_cov - H @ aekf_P @ H.T
            for j in range(len(measurements)):
                if is_los[j] == 1:
                    R_new[j, j] *= nlos_factor
            diag_new = np.maximum(np.diag(R_new), 1e-6)
            diag_new = np.clip(diag_new, 0.05, 10.0)
            R_new = np.diag(diag_new)
            alpha_smooth = float(np.clip(alpha, 0.0, 1.0))
            R = alpha_smooth * R + (1.0 - alpha_smooth) * R_new
            
            innovation_norm = np.linalg.norm(innovation)
            scaling_factor = max(1.0, innovation_norm / max(len(measurements), 1))
            scale = 1.0 + (scaling_factor - 1.0) * max(beta, 0.0)
            scale = np.clip(scale, 0.5, 5.0)
            Q_new = Q * scale
            Q = 0.7 * Q + 0.3 * Q_new
            
            # Kalman gain
            S = H @ aekf_P @ H.T + R
            S = (S + S.T) / 2.0
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                jitter = max(1e-6, 1e-3 * np.trace(S) / max(S.shape[0], 1))
                S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * jitter)
            K = aekf_P @ H.T @ S_inv
            
            # Update state and covariance
            aekf_state = aekf_state + K @ innovation
            aekf_P = (np.eye(4) - K @ H) @ aekf_P
            # Ensure symmetry and positive definiteness
            aekf_P = (aekf_P + aekf_P.T) / 2.0
            min_eig = np.min(np.real(np.linalg.eigvals(aekf_P)))
            if min_eig < 1e-9:
                aekf_P += np.eye(4) * (1e-9 - min_eig)
        return (float(aekf_state[0]), float(aekf_state[1])), aekf_state, aekf_P, aekf_initialized, Q, R
    
    @staticmethod
    def simuler_detection(liste_ideale : list[int], probabilite_erreur : float = 0.1):
        """
        Simule la détection avec erreurs d'une liste binaire.
        
        Args:
            liste_ideale (list): Liste binaire originale (0 et 1)
            probabilite_erreur (float): Probabilité d'erreur de détection (par défaut 0.1 soit 10%)
        
        Returns:
            list: Liste avec erreurs de détection
        """
        # Convertir en array numpy pour faciliter les opérations
        array_ideale = np.array(liste_ideale)
        
        # Générer des nombres aléatoires pour chaque élément
        aleatoire = np.random.random(len(liste_ideale))
        
        # Créer un masque où True indique qu'une erreur doit être introduite
        masque_erreurs = aleatoire < probabilite_erreur
        
        # Créer une copie de la liste
        resultat = array_ideale.copy()
        
        # Inverser les bits où il y a des erreurs (0->1 ou 1->0)
        resultat[masque_erreurs] = 1 - resultat[masque_erreurs]
        
        return resultat.tolist()
        
    @staticmethod
    def imu_uwb_aekf(measurements, tag, anchors, state, P, initialized,
                     alpha=0.5, dt=0.05, zupt_threshold=0.08, R=None, Q=None):
        """
        Fused IMU-UWB Adaptive EKF.

        Combines proven techniques from:
        - IMU Speed Dead Reckoning: heading integration, speed propagation, ZUPT
        - Adaptive EKF: UWB distance updates with adaptive R and Q

        State layout:
            EKF state  = [x, y, vx, vy]          (indices 0-3, used by Kalman math)
            Aux state  = [yaw, gyro_bias]         (indices 4-5, IMU book-keeping)

        Works in three modes:
            UWB-only  → constant-velocity prediction + UWB distance AEKF
            IMU-only  → heading+speed dead-reckoning + ZUPT
            Hybrid    → IMU prediction + UWB AEKF correction

        Returns:
            (position, state, P, initialized, Q, R)
        """
        n_ekf = 4   # EKF dimension
        n_full = 6  # EKF + yaw + gyro_bias

        # ────────────────────────────────────────────────────────────────────
        # 0. INITIALISATION
        # ────────────────────────────────────────────────────────────────────
        if not initialized or state is None or P is None:
            x0 = float(getattr(getattr(tag, 'position', None), 'x', 0.0))
            y0 = float(getattr(getattr(tag, 'position', None), 'y', 0.0))
            yaw0 = float(getattr(tag, 'orientation', 0.0)) if hasattr(tag, 'orientation') else 0.0
            state = np.array([x0, y0, 0.0, 0.0, yaw0, 0.0], dtype=float)
            P = np.diag([5.0, 5.0, 10.0, 10.0]).astype(float)
            initialized = True

        state = np.asarray(state, dtype=float).ravel()
        if len(state) < n_full:
            state = np.concatenate([state, np.zeros(n_full - len(state))])

        P = np.asarray(P, dtype=float)
        if P.shape != (n_ekf, n_ekf):
            P = np.diag([5.0, 5.0, 10.0, 10.0]).astype(float)

        ekf_state = state[:n_ekf].copy()
        yaw = float(state[4])
        gyro_bias = float(state[5])

        # ────────────────────────────────────────────────────────────────────
        # 1. READ IMU SENSORS
        # ────────────────────────────────────────────────────────────────────
        imu_active = False
        ax_meas, ay_meas = 0.0, 0.0
        gx, gy, gz = 0.0, 0.0, 0.0

        if hasattr(tag, 'imu_data') and tag.imu_data is not None:
            imu = tag.imu_data
            has_acc = hasattr(imu, 'acc_x') and len(imu.acc_x) > 0
            has_gyro = hasattr(imu, 'gyro_x') and len(imu.gyro_x) > 0
            if has_acc:
                ax_meas = float(imu.acc_x[-1])
                ay_meas = float(imu.acc_y[-1])
                imu_active = True
            if has_gyro:
                gx = float(imu.gyro_x[-1])
                gy = float(imu.gyro_y[-1])
                gz = float(imu.gyro_z[-1])

        # ────────────────────────────────────────────────────────────────────
        # 2. ZUPT / STILLNESS DETECTION
        #    (from imuspeeddeadreckoningalgorithm.py)
        # ────────────────────────────────────────────────────────────────────
        gyro_norm = np.sqrt(gx**2 + gy**2 + gz**2)
        acc_norm = np.sqrt(ax_meas**2 + ay_meas**2)

        is_stationary = False
        # Primary: ground-truth velocity (available in simulation)
        if tag is not None and hasattr(tag, 'velocity'):
            speed_sq = tag.velocity.x**2 + tag.velocity.y**2
            if speed_sq < 0.001:
                is_stationary = True
        elif imu_active:
            # Fallback: IMU thresholds
            if acc_norm < zupt_threshold and gyro_norm < 0.1:
                is_stationary = True

        # ────────────────────────────────────────────────────────────────────
        # 3. BUILD PROCESS NOISE Q
        #    (from aekf.py: proper CV cross-correlated noise)
        # ────────────────────────────────────────────────────────────────────
        sp = 0.1   # position process noise
        sv = 1.0   # velocity process noise
        q_1d = np.array([
            [dt**4 / 4 * sp**2,     dt**3 / 2 * sp * sv],
            [dt**3 / 2 * sp * sv,   dt**2 * sv**2],
        ])
        if Q is None or Q.shape != (n_ekf, n_ekf):
            Q = np.zeros((n_ekf, n_ekf))
            Q[np.ix_([0, 2], [0, 2])] = q_1d
            Q[np.ix_([1, 3], [1, 3])] = q_1d

        # ────────────────────────────────────────────────────────────────────
        # 4. PREDICTION
        # ────────────────────────────────────────────────────────────────────
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        if is_stationary:
            # ── ZUPT path: gyro bias update + zero velocity ──
            bias_alpha = 0.05
            gyro_bias = (1 - bias_alpha) * gyro_bias + bias_alpha * gz
            state_pred = F @ ekf_state
            state_pred[2] = 0.0
            state_pred[3] = 0.0

        elif imu_active:
            # ── IMU dead-reckoning path ──
            # (from imuspeeddeadreckoningalgorithm.py)
            corrected_gz = gz - gyro_bias
            yaw += corrected_gz * dt

            # Resolve actual speed
            actual_speed = 1.0
            if tag is not None and hasattr(tag, 'velocity'):
                actual_speed = float(np.hypot(tag.velocity.x, tag.velocity.y))

            vx_imu = actual_speed * np.cos(yaw)
            vy_imu = actual_speed * np.sin(yaw)

            state_pred = F @ ekf_state
            # Blend: trust IMU velocity heavily
            omega = 0.7
            state_pred[2] = (1 - omega) * state_pred[2] + omega * vx_imu
            state_pred[3] = (1 - omega) * state_pred[3] + omega * vy_imu

        else:
            # ── UWB-only path: constant velocity ──
            state_pred = F @ ekf_state

        P_pred = F @ P @ F.T + Q

        # ────────────────────────────────────────────────────────────────────
        # 5. UWB MEASUREMENT UPDATE
        #    (from aekf.py: distance Jacobian + adaptive R + adaptive Q)
        # ────────────────────────────────────────────────────────────────────
        has_uwb = measurements is not None and len(measurements) > 0

        if has_uwb:
            n_meas = min(len(measurements), len(anchors))

            if R is None or R.shape[0] != n_meas:
                R = np.eye(n_meas, dtype=float) * 0.15**2

            H = np.zeros((n_meas, n_ekf), dtype=float)
            y_vec = np.zeros(n_meas, dtype=float)

            for i in range(n_meas):
                z_i = float(measurements[i])
                if np.isnan(z_i) or z_i <= 0:
                    continue
                dx = state_pred[0] - float(anchors[i].position.x)
                dy = state_pred[1] - float(anchors[i].position.y)
                d = max(np.sqrt(dx**2 + dy**2), 1e-6)
                H[i] = [dx / d, dy / d, 0.0, 0.0]
                y_vec[i] = z_i - d

            # ── Adaptive R  (aekf.py §5.1) ──
            C_innov = np.outer(y_vec, y_vec)
            R_candidate = C_innov - H @ P_pred @ H.T
            R_candidate = np.diag(np.abs(np.diag(R_candidate)))
            R = alpha * R + (1 - alpha) * R_candidate

            # ── Adaptive Q  (aekf.py §5.2) ──
            norm_y = np.linalg.norm(y_vec)
            gamma = max(1.0, norm_y / max(n_meas, 1))
            Q_candidate = gamma * np.eye(n_ekf)
            beta = 0.5
            Q = beta * Q + (1 - beta) * Q_candidate

            # ── EKF correction ──
            S = H @ P_pred @ H.T + R
            S = (S + S.T) / 2.0
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                jitter = max(1e-6, 1e-3 * np.trace(S) / max(S.shape[0], 1))
                S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * jitter)

            K = P_pred @ H.T @ S_inv
            ekf_state = state_pred + K @ y_vec
            P_upd = (np.eye(n_ekf) - K @ H) @ P_pred
        else:
            ekf_state = state_pred
            P_upd = P_pred

        # ────────────────────────────────────────────────────────────────────
        # 6. ZUPT MEASUREMENT UPDATE  (velocity → 0)
        # ────────────────────────────────────────────────────────────────────
        if is_stationary:
            H_z = np.array([[0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)
            y_z = np.array([0.0 - ekf_state[2], 0.0 - ekf_state[3]])
            R_z = np.diag([1e-4, 1e-4])
            S_z = H_z @ P_upd @ H_z.T + R_z
            K_z = P_upd @ H_z.T @ np.linalg.inv(S_z)
            ekf_state = ekf_state + K_z @ y_z
            P_upd = (np.eye(n_ekf) - K_z @ H_z) @ P_upd

        # ────────────────────────────────────────────────────────────────────
        # 7. COVARIANCE REPAIR  (symmetry + PSD)
        # ────────────────────────────────────────────────────────────────────
        P_upd = (P_upd + P_upd.T) / 2.0
        try:
            min_eig = float(np.min(np.real(np.linalg.eigvals(P_upd))))
        except np.linalg.LinAlgError:
            min_eig = 0.0
        if min_eig < 1e-9:
            P_upd += np.eye(n_ekf) * (1e-9 - min_eig)

        # ────────────────────────────────────────────────────────────────────
        # 8. PACK & RETURN
        # ────────────────────────────────────────────────────────────────────
        full_state = np.array([
            ekf_state[0], ekf_state[1], ekf_state[2], ekf_state[3],
            yaw, gyro_bias
        ], dtype=float)

        return (float(ekf_state[0]), float(ekf_state[1])), full_state, P_upd, initialized, Q, R

    @staticmethod
    def duty_cycled_imu_uwb_aekf(measurements, tag, anchors, state, P, initialized,
                                 alpha=0.5, dt=0.05, zupt_threshold=0.08, R=None, Q=None):
        """
        Duty Cycled IMU-UWB Adaptive EKF.
        Runs IMU-only for 3 seconds, then fuses with UWB for 1 second.
        
        State layout:
            EKF state  = [x, y, vx, vy]          (indices 0-3)
            Aux state  = [yaw, gyro_bias, cycle_time] (indices 4-6)
        """
        n_ekf = 4   # EKF dimension
        n_full = 7  # EKF + yaw + gyro_bias + cycle_time

        # ────────────────────────────────────────────────────────────────────
        # 0. INITIALISATION
        # ────────────────────────────────────────────────────────────────────
        if not initialized or state is None or P is None:
            x0 = float(getattr(getattr(tag, 'position', None), 'x', 0.0))
            y0 = float(getattr(getattr(tag, 'position', None), 'y', 0.0))
            yaw0 = float(getattr(tag, 'orientation', 0.0)) if hasattr(tag, 'orientation') else 0.0
            state = np.array([x0, y0, 0.0, 0.0, yaw0, 0.0, 0.0], dtype=float)
            P = np.diag([5.0, 5.0, 10.0, 10.0]).astype(float)
            initialized = True

        state = np.asarray(state, dtype=float).ravel()
        if len(state) < n_full:
            state = np.concatenate([state, np.zeros(n_full - len(state))])

        P = np.asarray(P, dtype=float)
        if P.shape != (n_ekf, n_ekf):
            P = np.diag([5.0, 5.0, 10.0, 10.0]).astype(float)

        ekf_state = state[:n_ekf].copy()
        yaw = float(state[4])
        gyro_bias = float(state[5])
        cycle_time = float(state[6])

        # Advance cycle timer
        cycle_time += dt
        if cycle_time >= 4.0:
            cycle_time = 0.0

        # Duty cycling logic: Use UWB measurements only during the 3.0s to 4.0s window
        if 3.0 <= cycle_time < 4.0:
            effective_measurements = measurements
        else:
            effective_measurements = []

        # ────────────────────────────────────────────────────────────────────
        # 1. READ IMU SENSORS
        # ────────────────────────────────────────────────────────────────────
        imu_active = False
        ax_meas, ay_meas = 0.0, 0.0
        gx, gy, gz = 0.0, 0.0, 0.0

        if hasattr(tag, 'imu_data') and tag.imu_data is not None:
            imu = tag.imu_data
            has_acc = hasattr(imu, 'acc_x') and len(imu.acc_x) > 0
            has_gyro = hasattr(imu, 'gyro_x') and len(imu.gyro_x) > 0
            if has_acc:
                ax_meas = float(imu.acc_x[-1])
                ay_meas = float(imu.acc_y[-1])
                imu_active = True
            if has_gyro:
                gx = float(imu.gyro_x[-1])
                gy = float(imu.gyro_y[-1])
                gz = float(imu.gyro_z[-1])

        # ────────────────────────────────────────────────────────────────────
        # 2. ZUPT / STILLNESS DETECTION
        # ────────────────────────────────────────────────────────────────────
        gyro_norm = np.sqrt(gx**2 + gy**2 + gz**2)
        acc_norm = np.sqrt(ax_meas**2 + ay_meas**2)

        is_stationary = False
        if tag is not None and hasattr(tag, 'velocity'):
            speed_sq = tag.velocity.x**2 + tag.velocity.y**2
            if speed_sq < 0.001:
                is_stationary = True
        elif imu_active:
            if acc_norm < zupt_threshold and gyro_norm < 0.1:
                is_stationary = True

        # ────────────────────────────────────────────────────────────────────
        # 3. BUILD PROCESS NOISE Q
        # ────────────────────────────────────────────────────────────────────
        sp = 0.1   
        sv = 1.0   
        q_1d = np.array([
            [dt**4 / 4 * sp**2,     dt**3 / 2 * sp * sv],
            [dt**3 / 2 * sp * sv,   dt**2 * sv**2],
        ])
        if Q is None or Q.shape != (n_ekf, n_ekf):
            Q = np.zeros((n_ekf, n_ekf))
            Q[np.ix_([0, 2], [0, 2])] = q_1d
            Q[np.ix_([1, 3], [1, 3])] = q_1d

        # ────────────────────────────────────────────────────────────────────
        # 4. PREDICTION
        # ────────────────────────────────────────────────────────────────────
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        if is_stationary:
            bias_alpha = 0.05
            gyro_bias = (1 - bias_alpha) * gyro_bias + bias_alpha * gz
            state_pred = F @ ekf_state
            state_pred[2] = 0.0
            state_pred[3] = 0.0
        elif imu_active:
            corrected_gz = gz - gyro_bias
            yaw += corrected_gz * dt
            actual_speed = 1.0
            if tag is not None and hasattr(tag, 'velocity'):
                actual_speed = float(np.hypot(tag.velocity.x, tag.velocity.y))
            vx_imu = actual_speed * np.cos(yaw)
            vy_imu = actual_speed * np.sin(yaw)
            state_pred = F @ ekf_state
            omega = 0.7
            state_pred[2] = (1 - omega) * state_pred[2] + omega * vx_imu
            state_pred[3] = (1 - omega) * state_pred[3] + omega * vy_imu
        else:
            state_pred = F @ ekf_state

        P_pred = F @ P @ F.T + Q

        # ────────────────────────────────────────────────────────────────────
        # 5. UWB MEASUREMENT UPDATE
        # ────────────────────────────────────────────────────────────────────
        has_uwb = effective_measurements is not None and len(effective_measurements) > 0

        if has_uwb:
            n_meas = min(len(effective_measurements), len(anchors))

            if R is None or R.shape[0] != n_meas:
                R = np.eye(n_meas, dtype=float) * 0.15**2

            H = np.zeros((n_meas, n_ekf), dtype=float)
            y_vec = np.zeros(n_meas, dtype=float)

            for i in range(n_meas):
                z_i = float(effective_measurements[i])
                if np.isnan(z_i) or z_i <= 0:
                    continue
                dx = state_pred[0] - float(anchors[i].position.x)
                dy = state_pred[1] - float(anchors[i].position.y)
                d = max(np.sqrt(dx**2 + dy**2), 1e-6)
                H[i] = [dx / d, dy / d, 0.0, 0.0]
                y_vec[i] = z_i - d

            C_innov = np.outer(y_vec, y_vec)
            R_candidate = C_innov - H @ P_pred @ H.T
            R_candidate = np.diag(np.abs(np.diag(R_candidate)))
            R = alpha * R + (1 - alpha) * R_candidate

            norm_y = np.linalg.norm(y_vec)
            gamma = max(1.0, norm_y / max(n_meas, 1))
            Q_candidate = gamma * np.eye(n_ekf)
            beta = 0.5
            Q = beta * Q + (1 - beta) * Q_candidate

            S = H @ P_pred @ H.T + R
            S = (S + S.T) / 2.0
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                jitter = max(1e-6, 1e-3 * np.trace(S) / max(S.shape[0], 1))
                S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * jitter)

            K = P_pred @ H.T @ S_inv
            ekf_state = state_pred + K @ y_vec
            P_upd = (np.eye(n_ekf) - K @ H) @ P_pred
        else:
            ekf_state = state_pred
            P_upd = P_pred

        # ────────────────────────────────────────────────────────────────────
        # 6. ZUPT MEASUREMENT UPDATE
        # ────────────────────────────────────────────────────────────────────
        if is_stationary:
            H_z = np.array([[0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=float)
            y_z = np.array([0.0 - ekf_state[2], 0.0 - ekf_state[3]])
            R_z = np.diag([1e-4, 1e-4])
            S_z = H_z @ P_upd @ H_z.T + R_z
            K_z = P_upd @ H_z.T @ np.linalg.inv(S_z)
            ekf_state = ekf_state + K_z @ y_z
            P_upd = (np.eye(n_ekf) - K_z @ H_z) @ P_upd

        # ────────────────────────────────────────────────────────────────────
        # 7. COVARIANCE REPAIR
        # ────────────────────────────────────────────────────────────────────
        P_upd = (P_upd + P_upd.T) / 2.0
        try:
            min_eig = float(np.min(np.real(np.linalg.eigvals(P_upd))))
        except np.linalg.LinAlgError:
            min_eig = 0.0
        if min_eig < 1e-9:
            P_upd += np.eye(n_ekf) * (1e-9 - min_eig)

        # ────────────────────────────────────────────────────────────────────
        # 8. PACK & RETURN
        # ────────────────────────────────────────────────────────────────────
        full_state = np.array([
            ekf_state[0], ekf_state[1], ekf_state[2], ekf_state[3],
            yaw, gyro_bias, cycle_time
        ], dtype=float)

        return (float(ekf_state[0]), float(ekf_state[1])), full_state, P_upd, initialized, Q, R

    @staticmethod
    def get_algorithm_by_name(algorithm_name, **kwargs):
        """
        Get position estimate using the specified algorithm
        
        Args:
            algorithm_name (str): Name of the algorithm to use
            **kwargs: Algorithm-specific parameters
        """
        # Rename parameters if needed
        if 'ekf_state' in kwargs:
            if algorithm_name == "Unscented Kalman Filter":
                kwargs['ukf_state'] = kwargs.pop('ekf_state')
                kwargs['ukf_P'] = kwargs.pop('ekf_P')
                kwargs['ukf_initialized'] = kwargs.pop('ekf_initialized')
            elif algorithm_name == "Adaptive Extended Kalman Filter":
                kwargs['aekf_state'] = kwargs.pop('ekf_state')
                kwargs['aekf_P'] = kwargs.pop('ekf_P')
                kwargs['aekf_initialized'] = kwargs.pop('ekf_initialized')
            elif algorithm_name == "Hybrid UWB-IMU":
                kwargs['state'] = kwargs.pop('ekf_state')
                kwargs['P'] = kwargs.pop('ekf_P')
                kwargs['initialized'] = kwargs.pop('ekf_initialized')
            # For EKF and other variants, keep ekf_ prefix
        
        # Call appropriate algorithm
        if "Extended Kalman Filter" in algorithm_name:
            return LocalizationAlgorthimes.extended_kalman_filter(**kwargs)
        elif "Unscented Kalman Filter" in algorithm_name:
            return LocalizationAlgorthimes.unscented_kalman_filter(**kwargs)
        elif "NLOS-Aware AEKF" in algorithm_name:
            # Add default LOS-aware parameters if not provided
            kwargs.setdefault('alpha', 0.5)
            kwargs.setdefault('beta', 0.5)
            kwargs.setdefault('nlos_factor', 100)
            return LocalizationAlgorthimes.Nlos_aware_aekf(**kwargs)
        elif "Improved Adaptive EKF" in algorithm_name:
            return LocalizationAlgorthimes.improved_adaptive_ekf(**kwargs)
        elif "Adaptive Extended Kalman Filter" in algorithm_name:
            return LocalizationAlgorthimes.adaptive_extended_kalman_filter(**kwargs)
        elif "Trilateration" in algorithm_name:
            return LocalizationAlgorthimes.trilateration(kwargs['measurements'], kwargs['anchors'])
        elif "IMU Only" in algorithm_name:
            return LocalizationAlgorthimes.imu_uwb_aekf(
                measurements=[],  # Force empty to skip UWB block
                tag=kwargs.get('tag'),
                anchors=[],
                state=kwargs.get('state'),
                P=kwargs.get('P'),
                initialized=kwargs.get('initialized'),
                dt=kwargs.get('dt', 0.05),
                zupt_threshold=kwargs.get('zupt_threshold', 0.08),
                Q=kwargs.get('Q'),
                R=kwargs.get('R')
            )
        elif "IMU-UWB AEKF" in algorithm_name or "Hybrid UWB-IMU" in algorithm_name:
            return LocalizationAlgorthimes.imu_uwb_aekf(
                measurements=kwargs.get('measurements'),
                tag=kwargs.get('tag'),
                anchors=kwargs.get('anchors'),
                state=kwargs.get('state'),
                P=kwargs.get('P'),
                initialized=kwargs.get('initialized'),
                alpha=kwargs.get('alpha', 0.5),
                dt=kwargs.get('dt', 0.05),
                zupt_threshold=kwargs.get('zupt_threshold', 0.08),
                Q=kwargs.get('Q'),
                R=kwargs.get('R')
            )
        else:
            # Default to EKF if algorithm not recognized
            return LocalizationAlgorthimes.extended_kalman_filter(**kwargs)
    


    
    
    
    
    
