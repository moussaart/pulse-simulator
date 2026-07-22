import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.core.uwb.uwb_devices import Tag, Position
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import ExtendedKalmanFilter
from src.core.parallel.gpu_backend import get_array_module, to_cpu
from src.core.parallel.parallel_utils import vectorized_jacobian



class LocalizationAlgorthimes():
    
    
    Localization_algorthimes = ["Trilateration"]
    
    
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
    


    
    
    
    
    
