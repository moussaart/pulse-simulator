import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.core.uwb.uwb_devices import Tag, Position
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from src.core.parallel.gpu_backend import get_array_module, to_cpu
from src.core.parallel.parallel_utils import vectorized_jacobian



class LocalizationAlgorthimes():
    
    
    Localization_algorthimes = ["Trilateration", "NLOS-Aware AEKF", "Improved Adaptive EKF", 
                                "IMU Only", "IMU assisted NLOS-Aware AEKF"]
    
    
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
    def imu_only_filter(tag, measurements, state, P, initialized, dt=0.05, zupt_threshold=0.05):
        """
        IMU-only position tracking using a Kalman filter with Zero Velocity Update (ZUPT)
        
        Args:
            tag: Tag object containing IMU data (tag.imu_data of type IMUData)
            state: Current state vector [x, y, vx, vy, ax, ay]
            P: Current error covariance matrix
            initialized: Boolean indicating if filter is initialized
            dt: Time step in seconds
            zupt_threshold: Acceleration threshold for zero velocity detection
            
        Returns:
            tuple: (x, y) position estimate
        """
        n = 6  # State: [x, y, vx, vy, bx, by]  (use last two as accel biases)

        # Basic guards
        if dt is None or not np.isfinite(dt) or dt <= 0:
            dt = 0.05
        dt2 = dt * dt

        # Initialize state if needed
        if not initialized or state is None or P is None:
            state = np.array([
                float(getattr(getattr(tag, 'position', None), 'x', 0.0)),
                float(getattr(getattr(tag, 'position', None), 'y', 0.0)),
                0.0, 0.0,  # velocity
                0.0, 0.0   # accel biases bx, by
            ], dtype=float)
            # Larger uncertainty on biases
            P = np.diag([1.0, 1.0, 0.2, 0.2, 0.5, 0.5]).astype(float)
            initialized = True

        # Ensure shapes and finiteness
        state = np.asarray(state, dtype=float).reshape(n)
        P = np.asarray(P, dtype=float).reshape(n, n)
        P = (P + P.T) / 2.0

        # Latest IMU measurements
        ax_meas, ay_meas = 0.0, 0.0
        if measurements is not None and len(measurements) >= 2:
            ax_meas = float(measurements[0]) if np.isfinite(measurements[0]) else 0.0
            ay_meas = float(measurements[1]) if np.isfinite(measurements[1]) else 0.0

        # Gyro magnitude for ZUPT aid if available
        gyro_mag = None
        if hasattr(tag, 'imu_data') and tag.imu_data is not None:
            try:
                gx = tag.imu_data.gyro_x[-1] if hasattr(tag.imu_data, 'gyro_x') and len(tag.imu_data.gyro_x) > 0 else 0.0
                gy = tag.imu_data.gyro_y[-1] if hasattr(tag.imu_data, 'gyro_y') and len(tag.imu_data.gyro_y) > 0 else 0.0
                gz = tag.imu_data.gyro_z[-1] if hasattr(tag.imu_data, 'gyro_z') and len(tag.imu_data.gyro_z) > 0 else 0.0
                gyro_mag = float(np.sqrt(gx * gx + gy * gy + gz * gz))
            except Exception:
                gyro_mag = None

        # Bias-corrected acceleration for double integration
        bx, by = state[4], state[5]
        ax_corr = ax_meas - bx
        ay_corr = ay_meas - by

        # Double integration (semi-implicit): x_k+1 = x_k + v_k dt + 0.5 a dt^2; v_k+1 = v_k + a dt
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        x = x + vx * dt + 0.5 * ax_corr * dt2
        y = y + vy * dt + 0.5 * ay_corr * dt2
        vx = vx + ax_corr * dt
        vy = vy + ay_corr * dt

        state_pred = np.array([x, y, vx, vy, bx, by], dtype=float)

        # Linearized dynamics wrt state (accelerations are inputs; biases affect pos/vel)
        F = np.array([
            [1, 0, dt, 0, -0.5 * dt2, 0],
            [0, 1, 0, dt, 0, -0.5 * dt2],
            [0, 0, 1, 0, -dt, 0],
            [0, 0, 0, 1, 0, -dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], dtype=float)

        # Process noise: acceleration measurement noise and bias RW
        accel_noise_var = 0.15  # (m/s^2)^2
        bias_rw_var = 1e-4      # bias random walk variance per second

        # Map accel noise into pos/vel through integration
        q11 = 0.25 * dt2 * dt2 * accel_noise_var
        q13 = 0.5 * dt2 * dt * accel_noise_var
        q33 = dt2 * accel_noise_var

        Q = np.zeros((n, n), dtype=float)
        # x axis pos/vel block
        Q[0, 0] = q11
        Q[0, 2] = q13
        Q[2, 0] = q13
        Q[2, 2] = q33
        # y axis pos/vel block
        Q[1, 1] = q11
        Q[1, 3] = q13
        Q[3, 1] = q13
        Q[3, 3] = q33
        # bias RW
        Q[4, 4] = bias_rw_var * dt
        Q[5, 5] = bias_rw_var * dt

        P_pred = F @ P @ F.T + Q
        P_pred = (P_pred + P_pred.T) / 2.0

        # ZUPT detection: low accel and low rotation → assume zero velocity
        acc_magnitude = float(np.sqrt(ax_meas * ax_meas + ay_meas * ay_meas))
        apply_zupt = acc_magnitude < zupt_threshold
        if gyro_mag is not None:
            apply_zupt = apply_zupt and (gyro_mag < 0.1)

        state_upd = state_pred
        P_upd = P_pred

        if apply_zupt:
            # Velocity = 0 measurement
            H_zupt = np.zeros((2, n), dtype=float)
            H_zupt[0, 2] = 1.0
            H_zupt[1, 3] = 1.0
            z_zupt = np.array([0.0, 0.0], dtype=float)
            innov = z_zupt - (H_zupt @ state_pred)
            Rz = np.eye(2, dtype=float) * 1e-3
            S = H_zupt @ P_pred @ H_zupt.T + Rz
            S = (S + S.T) / 2.0
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * 1e-6)
            K = P_pred @ H_zupt.T @ S_inv
            I = np.eye(n)
            state_upd = state_pred + K @ innov
            KH = K @ H_zupt
            P_upd = (I - KH) @ P_pred @ (I - KH).T + K @ Rz @ K.T

            # Bias correction: when stationary, measured accel should be ~0 in x, y → move bias toward measurement
            bias_alpha = 0.05  # adaptation rate during ZUPT
            state_upd[4] = (1.0 - bias_alpha) * state_upd[4] + bias_alpha * ax_meas
            state_upd[5] = (1.0 - bias_alpha) * state_upd[5] + bias_alpha * ay_meas

        # Mild velocity damping to mitigate integration drift between ZUPTs
        vel_damp = 0.01
        state_upd[2] *= (1.0 - vel_damp)
        state_upd[3] *= (1.0 - vel_damp)

        # Final symmetry and PSD fix
        P_upd = (P_upd + P_upd.T) / 2.0
        try:
            min_eig = float(np.min(np.real(np.linalg.eigvals(P_upd))))
        except np.linalg.LinAlgError:
            min_eig = 0.0
        if min_eig < 1e-9:
            P_upd += np.eye(n) * (1e-9 - min_eig)

        return (float(state_upd[0]), float(state_upd[1])), state_upd, P_upd, initialized
       
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
            # Ensure measurements are provided; if not, derive from tag IMU data
            measurements = kwargs.get('measurements')
            if measurements is None and kwargs.get('tag') is not None and hasattr(kwargs['tag'], 'imu_data'):
                ax = kwargs['tag'].imu_data.acc_x[-1] if len(kwargs['tag'].imu_data.acc_x) > 0 else 0.0
                ay = kwargs['tag'].imu_data.acc_y[-1] if len(kwargs['tag'].imu_data.acc_y) > 0 else 0.0
                measurements = [float(ax), float(ay)]
            return LocalizationAlgorthimes.imu_only_filter(
                kwargs.get('tag'),
                measurements,
                kwargs.get('state'),
                kwargs.get('P'),
                kwargs.get('initialized'),
                kwargs.get('dt', 0.05),
                kwargs.get('zupt_threshold', 0.05)
            )
        elif "Hybrid UWB-IMU" in algorithm_name:
            return LocalizationAlgorthimes.IMU_assisted_Nlos_aware_aekf(
                kwargs.get('measurements'),
                kwargs.get('tag'),
                kwargs.get('anchors'),
                kwargs.get('state'),
                kwargs.get('P'),
                kwargs.get('initialized'),
                kwargs.get('is_los'),
                kwargs.get('dt', 0.05),
                kwargs.get('alpha', 0.3),
                kwargs.get('beta', 2.0),
                kwargs.get('nlos_factor', 10.0),
                kwargs.get('zupt_threshold', 0.05)
            )
        else:
            # Default to EKF if algorithm not recognized
            return LocalizationAlgorthimes.extended_kalman_filter(**kwargs)
    


    
    
    
    
    
