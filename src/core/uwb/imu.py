import numpy as np
import navpy

class IMUSimulator:
    """
    Realistic IMU simulator that generates sensor data from tag kinematics.
    Simulates accelerometer and gyroscope with realistic noise and bias characteristics.
    """
    def __init__(self, sample_rate=100):
        """
        Initialize IMU simulator with realistic sensor parameters.
        
        Args:
            sample_rate: IMU sampling rate in Hz (default: 100 Hz)
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # --- Tunable Simulation Variables ---
        # Modify these to change the noise and bias characteristics.
        
        # White Noise (Standard Deviation)
        self.acc_noise_std = 0.02        # Accelerometer noise (m/s²)
        self.gyro_noise_std = 0.001      # Gyroscope noise (rad/s)
        
        # Constant / Initial Bias
        # acc_bias: constant offset in accelerometer readings
        # gyro_bias: constant offset in gyroscope readings
        self.acc_bias = np.random.normal(0, 0.05, 3)   # Accelerometer bias (m/s²)
        self.gyro_bias = np.random.normal(0, 0.01, 3)  # Gyroscope bias (rad/s)
        
        # Bias Instability (Random Walk)
        # Higher values cause the bias to drift faster over time.
        self.bias_instability = 1e-5     # Bias random walk coefficient
        
        # Gravity
        self.g = 9.81                    # m/s²
        # ------------------------------------
        
        # Previous state for differentiation
        self.prev_position = None
        self.prev_velocity = None
        self.prev_orientation = None
        
    def reset(self):
        """Reset the simulator state"""
        self.prev_position = None
        self.prev_velocity = None
        self.prev_orientation = None
        self.acc_bias = np.random.normal(0, 0.05, 3)
        self.gyro_bias = np.random.normal(0, 0.01, 3)
        
    def generate_imu_data(self, position, orientation, dt):
        """
        Generate realistic IMU measurements from tag kinematics.
        
        Args:
            position: Position object with x, y, z coordinates
            orientation: Tag orientation in radians (yaw angle)
            dt: Time step since last measurement
            
        Returns:
            tuple: (acceleration[3], angular_velocity[3]) in body frame
        """
        current_pos = np.array([position.x, position.y, position.z])
        
        # Initialize on first call
        if self.prev_position is None:
            self.prev_position = current_pos
            self.prev_velocity = np.zeros(3)
            self.prev_orientation = orientation
            # Return stationary measurements
            acc_body = np.array([0, 0, self.g]) + self.acc_bias + np.random.normal(0, self.acc_noise_std, 3)
            gyro_body = self.gyro_bias + np.random.normal(0, self.gyro_noise_std, 3)
            return acc_body, gyro_body
        
        # Calculate velocity and acceleration in world frame
        velocity = (current_pos - self.prev_position) / dt
        acceleration = (velocity - self.prev_velocity) / dt
        
        # Calculate angular velocity (yaw rate for 2D motion)
        delta_yaw = orientation - self.prev_orientation
        # Normalize to [-pi, pi]
        delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
        yaw_rate = delta_yaw / dt
        
        # Transform acceleration to body frame using navpy
        R_world_to_body = navpy.angle2dcm(orientation, 0, 0, input_unit='rad')
        
        # Accelerometer measures specific force (linear acceleration + gravity)
        gravity_world = np.array([0, 0, -self.g])
        specific_force_world = acceleration - gravity_world
        specific_force_body = R_world_to_body @ specific_force_world
        
        # Gyroscope measures angular velocity in body frame
        angular_velocity_body = np.array([0, 0, yaw_rate])  # Only yaw for 2D motion
        
        # Add sensor imperfections
        # Update bias (random walk)
        self.acc_bias += np.random.normal(0, self.bias_instability, 3)
        self.gyro_bias += np.random.normal(0, self.bias_instability / 10, 3)
        
        # Add noise and bias
        acc_measured = specific_force_body + self.acc_bias + np.random.normal(0, self.acc_noise_std, 3)
        gyro_measured = angular_velocity_body + self.gyro_bias + np.random.normal(0, self.gyro_noise_std, 3)
        
        # Update previous state
        self.prev_position = current_pos
        self.prev_velocity = velocity
        self.prev_orientation = orientation
        
        return acc_measured, gyro_measured


class IMUData:
    """High-performance ring-buffer storage for IMU measurements.
    
    Pre-allocates a fixed-size NumPy array and overwrites in a circular
    fashion, eliminating the O(n)-per-call cost of np.append().
    
    Columns: [timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """
    _COL_T  = 0
    _COL_AX = 1
    _COL_AY = 2
    _COL_AZ = 3
    _COL_GX = 4
    _COL_GY = 5
    _COL_GZ = 6
    _N_COLS = 7

    def __init__(self, max_samples: int = 1000):
        self._max = max_samples
        self._buf = np.zeros((max_samples, self._N_COLS), dtype=np.float64)
        self._idx = 0          # next write position
        self._count = 0        # number of stored samples (up to _max)

    # ── O(1) insertion ────────────────────────────────────────────────
    def add_measurement(self, t, ax, ay, az, gx, gy, gz):
        """Add a new IMU measurement in O(1) time."""
        self._buf[self._idx] = (t, ax, ay, az, gx, gy, gz)
        self._idx = (self._idx + 1) % self._max
        if self._count < self._max:
            self._count += 1

    # ── Chronologically-ordered views (read-only) ─────────────────────
    def _ordered(self, col: int) -> np.ndarray:
        """Return column *col* in chronological order."""
        if self._count < self._max:
            return self._buf[:self._count, col]
        # Buffer is full → wrap around _idx
        return np.concatenate((
            self._buf[self._idx:, col],
            self._buf[:self._idx, col],
        ))

    @property
    def timestamps(self) -> np.ndarray:
        return self._ordered(self._COL_T)

    @property
    def acc_x(self) -> np.ndarray:
        return self._ordered(self._COL_AX)

    @property
    def acc_y(self) -> np.ndarray:
        return self._ordered(self._COL_AY)

    @property
    def acc_z(self) -> np.ndarray:
        return self._ordered(self._COL_AZ)

    @property
    def gyro_x(self) -> np.ndarray:
        return self._ordered(self._COL_GX)

    @property
    def gyro_y(self) -> np.ndarray:
        return self._ordered(self._COL_GY)

    @property
    def gyro_z(self) -> np.ndarray:
        return self._ordered(self._COL_GZ)

    # ── Housekeeping ──────────────────────────────────────────────────
    def clear(self):
        """Clear all stored measurements."""
        self._buf[:] = 0.0
        self._idx = 0
        self._count = 0

    def __len__(self):
        return self._count

    def __str__(self):
        """Return string representation of IMU data"""
        ts = self.timestamps
        return (f"IMU Data:\n"
                f"  Timestamps: {ts[-5:] if len(ts) > 0 else []}\n"
                f"  Accelerometer (x,y,z): \n"
                f"    x: {self.acc_x[-5:] if self._count > 0 else []}\n" 
                f"    y: {self.acc_y[-5:] if self._count > 0 else []}\n"
                f"    z: {self.acc_z[-5:] if self._count > 0 else []}\n"
                f"  Gyroscope (x,y,z): \n"
                f"    x: {self.gyro_x[-5:] if self._count > 0 else []}\n"
                f"    y: {self.gyro_y[-5:] if self._count > 0 else []}\n"
                f"    z: {self.gyro_z[-5:] if self._count > 0 else []}")

    def __repr__(self):
        """Return string representation of IMU data object"""
        return f"IMUData(samples={self._count})"
