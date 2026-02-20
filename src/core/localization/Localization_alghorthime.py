import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.core.uwb.uwb_devices import Tag, Position
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter



class LocalizationAlgorthimes():
    
    
    Localization_algorthimes = ["Trilateration",  "Extended Kalman Filter" , 
                                "Unscented Kalman Filter" ,"Adaptive Extended Kalman Filter", 
                                "NLOS-Aware AEKF", "Improved Adaptive EKF", 
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
    def extended_kalman_filter(measurements, tag, anchors, ekf_state, ekf_P, ekf_initialized, dt=0.05, imu_data_on=False, u=None, Q=None, R=None):
        """
        Implement EKF algorithm with optional control input and noise matrices
        Args:
            measurements: Distance measurements from anchors
            tag: Tag object containing position info
            anchors: List of anchor objects
            ekf_state: Current state estimate [x, y, vx, vy]
            ekf_P: Current error covariance matrix
            ekf_initialized: Boolean indicating if filter is initialized
            dt: Time step
            imu_data_on: If True, includes acceleration control input u(t)
            u: Control input vector [ax, ay] representing acceleration in x and y directions
            Q: Process noise covariance matrix (optional)
            R: Measurement noise covariance matrix (optional)
        """
        
        if not ekf_initialized:
            ekf_state = np.array([0.0, 0.0, 0.0, 0.0])
            ekf_P = np.eye(4) * 1.0
            ekf_initialized = True
        
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
            ekf_state = F @ ekf_state + B @ u
        else:
            ekf_state = F @ ekf_state
            
        # Use provided Q or default CV model process noise
        if Q is None:
            Q = LocalizationAlgorthimes._cv_process_noise(dt, accel_variance=0.1)
        ekf_P = F @ ekf_P @ F.T + Q
        
        # Update step
        if len(measurements) > 0:
            z = np.array(measurements)
            H = np.zeros((len(measurements), 4))
            h = np.zeros(len(measurements))
            
            # Calculate measurement Jacobian and predicted measurements
            for i, anchor in enumerate(anchors[:len(measurements)]):
                dx = ekf_state[0] - anchor.position.x
                dy = ekf_state[1] - anchor.position.y
                d = np.sqrt(dx**2 + dy**2)
                if d < 1e-6:
                    d = 1e-6
                H[i,0] = dx/d
                H[i,1] = dy/d
                h[i] = d
            
            # Use provided R or default
            if R is None:
                R = np.eye(len(measurements), dtype=float) * 0.2  # Slightly higher default noise
            
            # Kalman gain
            S = H @ ekf_P @ H.T + R
            # Regularize S to avoid singularity
            S = (S + S.T) / 2.0
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * 1e-6)
            K = ekf_P @ H.T @ S_inv
            
            # Update state and covariance
            ekf_state = ekf_state + K @ (z - h)
            ekf_P = (np.eye(4) - K @ H) @ ekf_P
        
        return (float(ekf_state[0]), float(ekf_state[1])), ekf_state, ekf_P,ekf_initialized
    
    @staticmethod
    def unscented_kalman_filter(measurements, tag, anchors, ukf_state, ukf_P, ukf_initialized, dt=0.05, imu_data_on=False, u=None, Q=None, R=None, alpha=0.3, beta=2.0, kappa=-1):
        """
        Custom UKF implementation with vectorized sigma point operations for improved performance.
        Args:
            measurements: Distance measurements from anchors
            tag: Tag object containing position info
            anchors: List of anchor objects
            ukf_state: Current state estimate [x, y, vx, vy]
            ukf_P: Current error covariance matrix
            ukf_initialized: Boolean indicating if filter is initialized
            dt: Time step
            imu_data_on: If True, includes acceleration control input u(t)
            u: Control input vector [ax, ay] representing acceleration in x and y directions
            Q: Process noise covariance matrix (optional)
            R: Measurement noise covariance matrix (optional)
            alpha: UKF scaling parameter
            beta: UKF scaling parameter (Gaussian assumption for 2 is optimal)
            kappa: UKF scaling parameter
        """
        n = 4  # State dimension [x, y, vx, vy]
        m = len(measurements) if measurements is not None else 0

        # Pre-compute state transition matrix for vectorized prediction
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Pre-compute control input contribution if enabled
        B_u = None
        if imu_data_on and u is not None:
            B_matrix = np.array([[0.5*dt**2, 0],
                                 [0, 0.5*dt**2],
                                 [dt, 0],
                                 [0, dt]])
            B_u = B_matrix @ u

        # Pre-compute anchor positions as numpy array for vectorized measurement function
        if m > 0:
            anchor_positions = np.array([[a.position.x, a.position.y] for a in anchors[:m]])

        def fx_vectorized(sigmas_batch):
            """Vectorized state transition for all sigma points at once"""
            # sigmas_batch shape: (2n+1, n)
            result = sigmas_batch @ F.T  # Batch matrix multiplication
            if B_u is not None:
                result = result + B_u  # Broadcast addition
            return result

        def hx_vectorized(sigmas_batch):
            """Vectorized measurement function for all sigma points at once"""
            if m == 0:
                return np.zeros((sigmas_batch.shape[0], 0))
            # sigmas_batch shape: (2n+1, n), anchor_positions shape: (m, 2)
            # Compute distances from all sigma points to all anchors
            # dx shape: (2n+1, m), dy shape: (2n+1, m)
            dx = sigmas_batch[:, 0:1] - anchor_positions[:, 0]  # Broadcasting
            dy = sigmas_batch[:, 1:2] - anchor_positions[:, 1]  # Broadcasting
            distances = np.sqrt(dx**2 + dy**2)  # shape: (2n+1, m)
            return distances

        if not ukf_initialized or ukf_state is None:
            ukf_state = np.array([0.0, 0.0, 0.0, 0.0])
            ukf_P = np.eye(n) * 1.0
            ukf_initialized = True

        # Sigma points calculation
        lambda_ = alpha**2 * (n + kappa) - n
        Wm = np.full(2 * n + 1, 1. / (2 * (n + lambda_)))
        Wc = np.full(2 * n + 1, 1. / (2 * (n + lambda_)))
        Wm[0] = lambda_ / (n + lambda_)
        Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

        # Generate sigma points
        sigmas = np.zeros((2 * n + 1, n))
        try:
            U = np.linalg.cholesky((n + lambda_) * ukf_P)
        except np.linalg.LinAlgError:
            jitter = 1e-6 * np.eye(n)
            U = np.linalg.cholesky((n + lambda_) * (ukf_P + jitter))
        sigmas[0] = ukf_state
        for k in range(n):
            sigmas[k + 1] = ukf_state + U[k]
            sigmas[n + k + 1] = ukf_state - U[k]

        # Vectorized Prediction - process all sigma points at once
        sigmas_f = fx_vectorized(sigmas)
        
        # Compute weighted mean
        x_pred = np.dot(Wm, sigmas_f)
        
        # Compute weighted covariance (vectorized)
        diff = sigmas_f - x_pred  # shape: (2n+1, n)
        P_pred = np.einsum('i,ij,ik->jk', Wc, diff, diff)  # Vectorized outer product sum
        
        if Q is None:
            Q_val = LocalizationAlgorthimes._cv_process_noise(dt, accel_variance=0.1)
        else:
            Q_val = Q
        P_pred += Q_val

        # Vectorized Update
        if m > 0 and measurements is not None:
            # Vectorized measurement prediction for all sigma points
            sigmas_h = hx_vectorized(sigmas_f)
            z_pred = np.dot(Wm, sigmas_h)
            
            # Vectorized measurement covariance
            diff_z = sigmas_h - z_pred  # shape: (2n+1, m)
            Pz = np.einsum('i,ij,ik->jk', Wc, diff_z, diff_z)

            if R is None:
                R_val = np.eye(m, dtype=float) * 0.1
            else:
                R_val = R
            Pz += R_val

            # Vectorized cross-covariance
            diff_x = sigmas_f - x_pred  # shape: (2n+1, n)
            Pxz = np.einsum('i,ij,ik->jk', Wc, diff_x, diff_z)
            
            # Regularize Pz to avoid singularity
            Pz = (Pz + Pz.T) / 2.0
            try:
                Pz_inv = np.linalg.inv(Pz)
            except np.linalg.LinAlgError:
                Pz_inv = np.linalg.inv(Pz + np.eye(Pz.shape[0]) * 1e-6)
            K = Pxz @ Pz_inv
            
            # Convert measurements to numpy array for vectorized subtraction
            measurements_arr = np.array(measurements)
            ukf_state = x_pred + K @ (measurements_arr - z_pred)
            ukf_P = P_pred - K @ Pz @ K.T
        else:
            ukf_state = x_pred
            ukf_P = P_pred
            
        # Ensure P is symmetric and positive semi-definite
        ukf_P = (ukf_P + ukf_P.T) / 2.0

        return (float(ukf_state[0]), float(ukf_state[1])), ukf_state, ukf_P, ukf_initialized
    
    @staticmethod
    def adaptive_extended_kalman_filter(measurements, tag, anchors, aekf_state, aekf_P, aekf_initialized, 
                                      dt=0.05, imu_data_on=False, u=None, Q=None, R=None):
        """
        Adaptive Extended Kalman Filter implementation that adapts Q and R matrices based on innovations
        Args:
            measurements: Distance measurements
            tag: Tag object
            anchors: List of anchor objects
            aekf_state: Current state estimate
            aekf_P: Current covariance matrix
            aekf_initialized: Boolean indicating if filter is initialized
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
            H = np.zeros((len(measurements), 4))
            h = np.zeros(len(measurements))
            
            # Calculate measurement Jacobian and predicted measurements
            for j, anchor in enumerate(anchors[:len(measurements)]):
                dx = aekf_state[0] - anchor.position.x
                dy = aekf_state[1] - anchor.position.y
                d = np.sqrt(dx**2 + dy**2)
                if d < 1e-6:
                    d = 1e-6
                H[j,0] = dx/d
                H[j,1] = dy/d
                h[j] = d
            
            # Use provided R or default; reset if size changed (anchor count changed)
            if R is None or R.shape[0] != len(measurements):
                r_scale = float(np.mean(np.diag(R))) if R is not None and R.size > 0 else 0.2
                R = np.eye(len(measurements), dtype=float) * r_scale
            
            # Innovation sequence
            innovation = z - h
            
            # Adapt R and Q immediately but with smoothing and bounds
            innovation_cov = np.outer(innovation, innovation)
            R_new = innovation_cov - H @ aekf_P @ H.T
            diag_new = np.maximum(np.diag(R_new), 1e-6)
            diag_new = np.clip(diag_new, 0.05, 10.0)
            R_new = np.diag(diag_new)
            R = 0.7 * R + 0.3 * R_new
            
            # Adapt Q based on innovation magnitude
            innovation_norm = np.linalg.norm(innovation)
            scaling_factor = max(1.0, innovation_norm / max(len(measurements), 1))
            Q_new = Q * np.clip(scaling_factor, 0.5, 5.0)
            Q = 0.7 * Q + 0.3 * Q_new
            
            # Kalman gain
            S = H @ aekf_P @ H.T + R
            S = (S + S.T) / 2.0
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * 1e-6)
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
            H = np.zeros((len(measurements), 4))
            h = np.zeros(len(measurements))
            
            # Calculate measurement Jacobian and predicted measurements
            for j, anchor in enumerate(anchors[:len(measurements)]):
                dx = aekf_state[0] - anchor.position.x
                dy = aekf_state[1] - anchor.position.y
                d = np.sqrt(dx**2 + dy**2)
                if d < 1e-6:
                    d = 1e-6
                H[j,0] = dx/d
                H[j,1] = dy/d
                h[j] = d
            
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
    def improved_adaptive_ekf(measurements, tag, anchors, aekf_state, aekf_P, aekf_initialized, dt=0.05, 
                              mu=0.95, alpha=0.3, xi=20, lambda_min=0.1, lambda_max=3.0, tau=0.95,
                              iteration_count=0, prev_R=None, innovation_history=None, 
                              imu_data_on=False, u=None, Q=None, R=None):
        """
        Improved Adaptive Extended Kalman Filter (IAEKF) implementation
        
        Args:
            measurements: Distance measurements from anchors
            tag: Tag object containing current position estimate
            anchors: List of anchor objects
            aekf_state: Current state estimate [x, vx, y, vy]
            aekf_P: Current error covariance matrix
            aekf_initialized: Boolean indicating if filter is initialized
            dt: Time step
            window_size: Maximum window size for innovation estimation
            mu, alpha, xi: Adaptive window size parameters
            lambda_min, lambda_max: Innovation bounds for window adaptation
            tau: Forgetting factor for measurement noise adaptation
            iteration_count: Current iteration number
            prev_R: Previous measurement noise covariance matrix
            innovation_history: List to store recent innovations for window calculations
            Q: Initial process noise covariance matrix (optional)
            R: Initial measurement noise covariance matrix (optional)
        """
        n = 4  # State dimension [x, y, vx, vy]
        m = len(measurements) if measurements is not None else 0
        
        # Initialize state if needed
        if not aekf_initialized or aekf_state is None:
            aekf_state = np.array([tag.position.x, tag.position.y, 0.0, 0.0])
            aekf_P = np.diag([1.0, 1.0, 0.1, 0.1])
            aekf_initialized = True
            iteration_count = 0
            tau = 0.95
            
        # Initialize R_prev if needed; reset if size changed (anchor count changed)
        if prev_R is not None and m > 0 and prev_R.shape[0] != m:
            r_scale = float(np.mean(np.diag(prev_R))) if prev_R.size > 0 else 0.1
            prev_R = np.eye(m) * r_scale
        if prev_R is None and R is not None:
            prev_R = R if R.shape[0] == m else np.eye(m) * 0.1
        elif prev_R is None:
            prev_R = np.eye(m) * 0.1 if m > 0 else np.eye(1) * 0.1
            
        # Initialize or update innovation history
        # Clear history if measurement dimension changed (anchor count changed)
        if innovation_history is None:
            innovation_history = []
        elif len(innovation_history) > 0 and m > 0 and len(innovation_history[-1]) != m:
            innovation_history = []
        
        # State transition matrix F
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Prediction step with optional control input
        if imu_data_on and u is not None:
            B = np.array([
                [0.5*dt**2, 0],
                [0, 0.5*dt**2],
                [dt, 0],
                [0, dt]
            ])
            x_pred = F @ aekf_state + B @ u
        else:
            x_pred = F @ aekf_state
        
        # Use provided Q or default
        if Q is None:
            Q = LocalizationAlgorthimes._cv_process_noise(dt, accel_variance=0.1)
        P_pred = F @ aekf_P @ F.T + Q
        
        if len(measurements) > 0:
            # Measurement matrix H (Jacobian)
            H = np.zeros((m, 4))
            h = np.zeros(m)
            
            # Calculate measurement Jacobian and predicted measurements
            for i, anchor in enumerate(anchors[:m]):
                dx = x_pred[0] - anchor.position.x
                dy = x_pred[1] - anchor.position.y
                d = np.sqrt(dx**2 + dy**2)
                if d < 1e-6:
                    d = 1e-6
                H[i,0] = dx/d  # ∂h/∂x
                H[i,1] = dy/d  # ∂h/∂y
                h[i] = d
            
            # Innovation
            z = np.array(measurements)
            innovation = z - h
            
            # Store current innovation in history
            innovation_history.append(innovation)
            if len(innovation_history) > xi:
                innovation_history.pop(0)
            
            # Calculate adaptive window size based on normalized innovation squared
            E_z = H @ P_pred @ H.T + prev_R
            # Regularize E_z to avoid singularity
            E_z = (E_z + E_z.T) / 2.0
            try:
                E_z_inv = np.linalg.inv(E_z)
            except np.linalg.LinAlgError:
                E_z_inv = np.linalg.inv(E_z + np.eye(E_z.shape[0]) * 1e-6)
            e = innovation.T @ E_z_inv @ innovation / len(measurements)
            
            # Determine adaptive window size M
            if e >= lambda_max:
                M = 1
            elif e <= lambda_min:
                M = min(xi, len(innovation_history))
            else:
                M = min(int(xi * (mu ** ((e - lambda_min) / alpha))), len(innovation_history))
            
            # Calculate innovation covariance using window
            innovation_cov = np.zeros_like(prev_R)
            for i in range(-M, 0):
                innovation_cov += np.outer(innovation_history[i], innovation_history[i])
            innovation_cov /= M
            
            # Adaptive measurement noise with improved stability
            tau_k = (1 - tau) / (1 - tau**(iteration_count + 1))
            R = (1 - tau_k) * prev_R + tau_k * (innovation_cov - H @ P_pred @ H.T)
            
            # Ensure R remains positive definite
            R = np.diag(np.maximum(np.diag(R), 1e-6))
            
            # Adaptive process noise Q based on innovation sequence
            innovation_magnitude = np.mean([np.linalg.norm(inn) for inn in innovation_history[-M:]])
            Q_scale = np.clip(innovation_magnitude / len(measurements), 0.1, 2.0)
            Q = LocalizationAlgorthimes._cv_process_noise(dt, accel_variance=0.1 * Q_scale)
            
            # Update prediction covariance with adaptive Q
            P_pred = F @ aekf_P @ F.T + Q
            
            # Kalman gain
            S = H @ P_pred @ H.T + R
            S = (S + S.T) / 2.0
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * 1e-6)
            K = P_pred @ H.T @ S_inv
            
            # Update state and covariance
            aekf_state = x_pred + K @ innovation
            aekf_P = (np.eye(4) - K @ H) @ P_pred
            
            # Ensure symmetry and positive definiteness
            aekf_P = (aekf_P + aekf_P.T) / 2
            min_eig = np.min(np.real(np.linalg.eigvals(aekf_P)))
            if min_eig < 1e-6:
                aekf_P += np.eye(4) * (1e-6 - min_eig)
        else:
            aekf_state = x_pred
            aekf_P = P_pred
            R = prev_R
            
        # Return just the position tuple to match other localization methods (x, y)
        return (float(aekf_state[0]), float(aekf_state[1])), innovation_history, aekf_state, aekf_P, aekf_initialized, Q, R
    
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
    def IMU_assisted_Nlos_aware_aekf(measurements, tag, anchors, state, P, initialized, 
                             is_los=None, dt=0.05, alpha=0.3, beta=2.0, 
                             nlos_factor=10.0, zupt_threshold=0.05, R=None):
        """
        Hybrid UWB-IMU localization filter that combines NLOS-aware AEKF with IMU-based tracking
        
        Args:
            measurements: Distance measurements from UWB
            tag: Tag object containing position and IMU data
            anchors: List of anchor objects
            state: Current state estimate [x, y, vx, vy, ax, ay]
            P: Current covariance matrix
            initialized: Boolean indicating if filter is initialized
            is_los: List of booleans (0 for LOS, 1 for NLOS)
            dt: Time step in seconds
            alpha: Adaptive parameter for R update
            beta: Adaptive parameter for Q update
            nlos_factor: Scaling factor for NLOS measurements
            zupt_threshold: Acceleration threshold for zero velocity detection
            
        Returns:
            tuple: (x, y) position estimate
        """
        n = 6  # State dimension [x, y, vx, vy, ax, ay]
        m = len(measurements) if measurements is not None else 0
        
        # Initialize state if needed
        if not initialized or state is None:
            state = np.array([
                tag.position.x, 
                tag.position.y, 
                0.0,  # vx
                0.0,  # vy
                0.0,  # ax
                0.0   # ay
            ])
            P = np.diag([1.0, 1.0, 0.1, 0.1, 0.05, 0.05])
            initialized = True
        
        # Extract IMU data if available
        if hasattr(tag, 'imu_data') and len(tag.imu_data.timestamps) > 0:
            ax_imu = tag.imu_data.acc_x[-1] if len(tag.imu_data.acc_x) > 0 else 0
            ay_imu = tag.imu_data.acc_y[-1] if len(tag.imu_data.acc_y) > 0 else 0
            
            # Get gyroscope data if available for improved ZUPT
            gx = tag.imu_data.gyro_x[-1] if hasattr(tag.imu_data, 'gyro_x') and len(tag.imu_data.gyro_x) > 0 else 0
            gy = tag.imu_data.gyro_y[-1] if hasattr(tag.imu_data, 'gyro_y') and len(tag.imu_data.gyro_y) > 0 else 0
            gz = tag.imu_data.gyro_z[-1] if hasattr(tag.imu_data, 'gyro_z') and len(tag.imu_data.gyro_z) > 0 else 0
            
            # Calculate acceleration and angular velocity magnitudes
            acc_magnitude = np.sqrt(ax_imu**2 + ay_imu**2)
            gyro_magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        else:
            ax_imu, ay_imu = 0, 0
            gx, gy, gz = 0, 0, 0
            acc_magnitude = 0
            gyro_magnitude = 0
        
        # State transition matrix
        F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Process noise matrix - adaptive based on IMU reliability
        sigma_a = 0.1  # Base acceleration noise
        sigma_j = 0.01  # Base jerk noise
        
        # Increase process noise if IMU data is unreliable (high acceleration or angular velocity)
        imu_reliability = 1.0
        if acc_magnitude > 5.0 or gyro_magnitude > 3.0:  # Thresholds for unreliable IMU data
            imu_reliability = min(1.0, 5.0 / (acc_magnitude + 0.1) + 3.0 / (gyro_magnitude + 0.1))
            sigma_a *= (2.0 - imu_reliability)
            sigma_j *= (2.0 - imu_reliability)
        
        Q = np.zeros((6, 6))
        # Position noise
        Q[0:2, 0:2] = np.eye(2) * (dt**4/4) * sigma_a
        # Velocity noise
        Q[2:4, 2:4] = np.eye(2) * (dt**2) * sigma_a
        # Acceleration noise
        Q[4:6, 4:6] = np.eye(2) * dt * sigma_j
        
        # Prediction step
        state_pred = F @ state
        
        # Incorporate IMU measurements into the prediction
        # Adaptive IMU weight based on reliability
        imu_weight = 0.7 * imu_reliability  # Weight for IMU measurements
        state_pred[4] = (1 - imu_weight) * state_pred[4] + imu_weight * ax_imu
        state_pred[5] = (1 - imu_weight) * state_pred[5] + imu_weight * ay_imu
        
        P_pred = F @ P @ F.T + Q
        
        # Improved ZUPT detection using both acceleration and angular velocity
        # ZUPT is more likely when both acceleration and angular velocity are low
        zupt_acc_condition = acc_magnitude < zupt_threshold
        zupt_gyro_condition = gyro_magnitude < 0.1  # rad/s threshold
        apply_zupt = zupt_acc_condition and zupt_gyro_condition
        
        # If we have velocity history, use it for improved ZUPT
        velocity_magnitude = np.sqrt(state[2]**2 + state[3]**2)
        if velocity_magnitude < 0.05:  # Very low velocity
            apply_zupt = apply_zupt or (acc_magnitude < zupt_threshold * 2)  # More lenient threshold
        
        # Determine which measurements are available and build a unified measurement model
        has_uwb = m > 0
        has_imu = ax_imu != 0 or ay_imu != 0
        
        # Count total measurements
        total_measurements = m + (2 if has_imu else 0) + (2 if apply_zupt else 0)
        
        if total_measurements > 0:
            # Create unified measurement vector, Jacobian, and noise matrix
            z_unified = np.zeros(total_measurements)
            h_unified = np.zeros(total_measurements)
            H_unified = np.zeros((total_measurements, 6))
            R_unified = np.zeros((total_measurements, total_measurements))
            
            # Current index in the unified measurement vector
            idx = 0
            
            # Add UWB measurements if available
            if has_uwb:
                # Default all measurements to LOS if not specified
                if is_los is None:
                    is_los = [0] * m
                
                # Add UWB measurements to unified vector
                z_unified[idx:idx+m] = np.array(measurements)
                
                # Calculate predicted UWB measurements and Jacobian
                for i, anchor in enumerate(anchors[:m]):
                    dx = state_pred[0] - anchor.position.x
                    dy = state_pred[1] - anchor.position.y
                    d = np.sqrt(dx**2 + dy**2)
                    if d < 1e-6:
                        d = 1e-6
                    h_unified[idx+i] = d
                    H_unified[idx+i, 0] = dx/d
                    H_unified[idx+i, 1] = dy/d
                
                # Set UWB measurement noise with NLOS consideration
                R_uwb = np.eye(m) * 0.1
                for i in range(m):
                    if is_los[i] == 1:  # NLOS case
                        R_uwb[i, i] *= nlos_factor
                
                # Adaptive R estimation for UWB
                y_uwb = z_unified[idx:idx+m] - h_unified[idx:idx+m]
                innovation_cov = np.outer(y_uwb, y_uwb)
                H_uwb = H_unified[idx:idx+m, :]
                R_new = innovation_cov - H_uwb @ P_pred @ H_uwb.T
                
                # Apply NLOS scaling to R_new
                for i in range(m):
                    if is_los[i] == 1:  # NLOS case
                        R_new[i, i] = max(R_new[i, i], R_uwb[i, i])
                
                # Ensure positive diagonal and blend with prior/nominal R_uwb
                R_new = np.diag(np.maximum(np.diag(R_new), 1e-6))
                alpha_smooth = float(np.clip(alpha, 0.0, 1.0))
                R_uwb = alpha_smooth * R_uwb + (1.0 - alpha_smooth) * R_new
                
                # Add to unified R matrix
                R_unified[idx:idx+m, idx:idx+m] = R_uwb
                
                # Update index
                idx += m
            
            # Add IMU measurements if available
            if has_imu:
                # Add IMU measurements to unified vector
                z_unified[idx:idx+2] = np.array([ax_imu, ay_imu])
                
                # IMU measures accelerations directly
                h_unified[idx] = state_pred[4]
                h_unified[idx+1] = state_pred[5]
                
                # IMU measurement Jacobian
                H_unified[idx, 4] = 1    # ax measurement
                H_unified[idx+1, 5] = 1  # ay measurement
                
                # Adaptive R for IMU based on acceleration magnitude and reliability
                imu_noise_scale = 0.1 + 0.05 * acc_magnitude + 0.1 * (1.0 - imu_reliability)
                R_imu = np.eye(2) * imu_noise_scale
                R_unified[idx:idx+2, idx:idx+2] = R_imu
                
                # Update index
                idx += 2
            
            # Add ZUPT measurements if applicable
            if apply_zupt:
                # Zero velocity measurements
                z_unified[idx:idx+2] = np.array([0.0, 0.0])
                
                # ZUPT measures velocities
                h_unified[idx] = state_pred[2]
                h_unified[idx+1] = state_pred[3]
                
                # ZUPT measurement Jacobian
                H_unified[idx, 2] = 1    # vx measurement
                H_unified[idx+1, 3] = 1  # vy measurement
                
                # Adaptive ZUPT measurement noise based on confidence
                zupt_confidence = 1.0
                if acc_magnitude > zupt_threshold * 0.5:
                    zupt_confidence *= (zupt_threshold / acc_magnitude)
                if gyro_magnitude > 0.05:
                    zupt_confidence *= (0.1 / gyro_magnitude)
                
                # Higher confidence = lower noise
                R_zupt = np.eye(2) * (0.001 / zupt_confidence)
                R_unified[idx:idx+2, idx:idx+2] = R_zupt
            
            # Unified Kalman update
            if R is not None and np.shape(R) == np.shape(R_unified):
                # Smooth the unified R with previous R to avoid jumps
                alpha_smooth = float(np.clip(alpha, 0.0, 1.0))
                R_unified = alpha_smooth * R + (1 - alpha_smooth) * R_unified
            R = R_unified
            y_unified = z_unified - h_unified
            S_unified = H_unified @ P_pred @ H_unified.T + R_unified
            
            # Ensure S_unified is invertible
            # Ensure symmetry and regularize S_unified
            S_unified = (S_unified + S_unified.T) / 2.0
            try:
                S_unified_inv = np.linalg.inv(S_unified)
            except np.linalg.LinAlgError:
                S_unified_inv = np.linalg.inv(S_unified + np.eye(S_unified.shape[0]) * 1e-6)
            K_unified = P_pred @ H_unified.T @ S_unified_inv
            
            # Update state and covariance
            state = state_pred + K_unified @ y_unified
            P = (np.eye(6) - K_unified @ H_unified) @ P_pred
            
            # Apply velocity constraints if we have high confidence in ZUPT
            if apply_zupt and zupt_confidence > 0.8:
                # Directly dampen velocities
                damping_factor = 0.8
                state[2] *= (1.0 - damping_factor)  # vx
                state[3] *= (1.0 - damping_factor)  # vy
        else:
            # If no measurements, just use prediction
            state = state_pred
            P = P_pred
        
        # Ensure symmetry and positive definiteness
        P = (P + P.T) / 2
        min_eig = np.min(np.real(np.linalg.eigvals(P)))
        if min_eig < 1e-6:
            P += np.eye(6) * (1e-6 - min_eig)
        
        return (float(state[0]), float(state[1])) , state, P, initialized, R
    
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
    


    
    
    
    
    
