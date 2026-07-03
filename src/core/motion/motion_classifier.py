import numpy as np
from collections import deque

class IndependentMotionClassifier:
    """
    Real-time motion classifier that hooks into the PULSE simulator's live data stream
    independently of any specific localization algorithm.
    Analyzes IMU data (Accel, Gyro), Velocity, and Ground Truth motion state to classify movement into:
    - Linear Motion
    - Circular/Curvilinear Motion
    - Random Walk
    """
    
    def __init__(self):
        self.window_size = 50  # Size of sliding window (e.g., 0.5s at 100Hz)
        self.accel_window = deque(maxlen=self.window_size)
        self.gyro_window = deque(maxlen=self.window_size)
        
    def initialize(self) -> None:
        self.accel_window.clear()
        self.gyro_window.clear()

    # METHOD 1: Gyroscope Kinematic Thresholding
    def method_1_gyro_kinematic(self, gyro_window) -> str:
        if len(gyro_window) < 10:
            return "Unknown"
        mags = [np.linalg.norm(g) for g in gyro_window]
        mean_mag = np.mean(mags)
        var_mag = np.var(mags)
        threshold = 0.5
        if mean_mag < threshold and var_mag < 0.01:
            return "Linear Motion"
        elif mean_mag >= threshold and var_mag < 0.1:
            return "Circular/Curvilinear Motion"
        else:
            return "Random Walk"

    # METHOD 2: Principal Component Analysis (PCA)
    def method_2_pca(self, accel_window) -> str:
        if len(accel_window) < self.window_size:
            return "Unknown"
        data = np.array(accel_window)
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

    # METHOD 3: Centripetal Acceleration Analysis
    def method_3_centripetal(self, accel: np.ndarray, gyro: np.ndarray, vel: np.ndarray) -> str:
        v_mag = np.linalg.norm(vel)
        if v_mag < 0.05:
            return "Linear Motion" 
        a_theory = np.cross(gyro, vel)
        a_theory_mag = np.linalg.norm(a_theory)
        accel_linear = np.array([accel[0], accel[1], accel[2] - 9.81])
        v_dir = vel / v_mag
        a_long = np.dot(accel_linear, v_dir) * v_dir
        a_lat = accel_linear - a_long
        a_lat_mag = np.linalg.norm(a_lat)
        error = abs(a_theory_mag - a_lat_mag)
        
        if a_theory_mag < 0.05 and error < 0.5:
            return "Linear Motion"
        elif error < 1.0 and a_theory_mag >= 0.05:
            return "Circular/Curvilinear Motion"
        else:
            return "Random Walk"

    # METHOD 4: Variance & Zero-Crossing Rate
    def method_4_variance_zcr(self, accel_window) -> str:
        if len(accel_window) < self.window_size:
            return "Unknown"
        data = np.array(accel_window)
        data_xy = data[:, 0:2]
        data_centered = data_xy - np.mean(data_xy, axis=0)
        variance = np.var(data_centered, axis=0)
        mean_var = np.mean(variance)
        zero_crossings = np.sum(np.diff(np.sign(data_centered), axis=0) != 0, axis=0)
        mean_zcr = np.mean(zero_crossings) / self.window_size
        
        if mean_var < 0.1 and mean_zcr < 0.05:
            return "Linear Motion"
        elif mean_zcr > 0.15:
            return "Random Walk"
        else:
            return "Circular/Curvilinear Motion"

    def get_ground_truth_state(self, tag) -> str:
        v_mag = np.linalg.norm([tag.velocity.x, tag.velocity.y])
        w = getattr(tag, 'angular_velocity', 0.0)
        if v_mag > 0.01 and abs(w) > 0.05:
            return "Circular/Curvilinear Motion"
        elif v_mag > 0.01 and abs(w) <= 0.05:
            return "Linear Motion"
        else:
            return "Random Walk"

    def update(self, tag, selected_method: str = "Method 1: Gyro Kinematic") -> tuple[str, str] | None:
        if not hasattr(tag, 'imu_data') or tag.imu_data is None or len(tag.imu_data) == 0:
            return None
            
        imu = tag.imu_data
        accel = np.array([imu.acc_x[-1], imu.acc_y[-1], imu.acc_z[-1]])
        gyro = np.array([imu.gyro_x[-1], imu.gyro_y[-1], imu.gyro_z[-1]])
        
        if accel is None or gyro is None:
             return None
            
        self.accel_window.append(accel)
        self.gyro_window.append(gyro)
        
        vel = np.array([tag.velocity.x, tag.velocity.y, 0.0])
        
        if len(self.accel_window) == self.window_size:
            ground_truth = self.get_ground_truth_state(tag)
            
            result = "Unknown"
            
            if selected_method == "Method 1: Gyro Kinematic":
                result = self.method_1_gyro_kinematic(self.gyro_window)
            elif selected_method == "Method 2: PCA":
                result = self.method_2_pca(self.accel_window)
            elif selected_method == "Method 3: Centripetal":
                result = self.method_3_centripetal(accel, gyro, vel)
            elif selected_method == "Method 4: Variance/ZCR":
                result = self.method_4_variance_zcr(self.accel_window)
            
            return ground_truth, result
            
        return None
