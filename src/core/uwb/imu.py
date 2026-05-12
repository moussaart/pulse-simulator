import numpy as np

class IMUSimulator:
    """
    IMU simulator that generates accelerometer and gyroscope measurements.

    Frame and unit conventions:
    - World frame: x/y are horizontal map axes, z is up, units are meters and seconds.
    - Body/sensor frame: yaw-only body frame attached to the tag. At yaw=0, body axes
      align with world axes. Positive yaw rotates body x toward world y.
    - Gravity in world frame is [0, 0, -9.81] m/s^2.
    - Accelerometer output is specific force in body frame, so a stationary level tag
      measures [0, 0, +9.81] m/s^2.
    - Gyroscope output is angular velocity in body frame, in rad/s.
    """
    def __init__(self, sample_rate=100, acc_noise_std=0.02, gyro_noise_std=0.001,
                 acc_bias=None, gyro_bias=None, bias_instability=1e-5, gravity=9.81,
                 acc_scale_factor=None, gyro_scale_factor=None, rng=None):
        """
        Initialize IMU simulator with realistic sensor parameters.
        
        Args:
            sample_rate: IMU sampling rate in Hz (default: 100 Hz)
            acc_noise_std: accelerometer white-noise standard deviation (m/s^2)
            gyro_noise_std: gyroscope white-noise standard deviation (rad/s)
            acc_bias: accelerometer bias vector (m/s^2). Random if None.
            gyro_bias: gyroscope bias vector (rad/s). Random if None.
            bias_instability: bias random-walk coefficient per sqrt(second)
            gravity: gravitational acceleration magnitude (m/s^2)
            acc_scale_factor: per-axis accelerometer multiplicative scale error
            gyro_scale_factor: per-axis gyroscope multiplicative scale error
            rng: optional NumPy random generator for reproducible simulations
        """
        if sample_rate <= 0 or not np.isfinite(sample_rate):
            raise ValueError("sample_rate must be finite and positive")

        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.rng = rng
        
        # --- Tunable Simulation Variables ---
        self.acc_noise_std = float(acc_noise_std)
        self.gyro_noise_std = float(gyro_noise_std)
        self.acc_bias = self._random_normal(0, 0.05, 3) if acc_bias is None else self._as_vector(acc_bias)
        self.gyro_bias = self._random_normal(0, 0.01, 3) if gyro_bias is None else self._as_vector(gyro_bias)
        self.bias_instability = float(bias_instability)
        self.g = float(gravity)
        self.acc_scale_factor = np.zeros(3) if acc_scale_factor is None else self._as_vector(acc_scale_factor)
        self.gyro_scale_factor = np.zeros(3) if gyro_scale_factor is None else self._as_vector(gyro_scale_factor)
        # ------------------------------------
        
        # Previous state for differentiation
        self.prev_position = None
        self.prev_velocity = None
        self.prev_orientation = None

    def _random_normal(self, mean, std, size):
        if self.rng is not None:
            return self.rng.normal(mean, std, size)
        return np.random.normal(mean, std, size)

    @staticmethod
    def _as_vector(value):
        vector = np.asarray(value, dtype=float).reshape(3)
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"IMU vector contains non-finite values: {vector}")
        return vector

    @staticmethod
    def _world_to_body_matrix(yaw):
        """Return DCM that rotates a world-frame vector into the yaw-only body frame."""
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ])

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
        
    def reset(self):
        """Reset the simulator state"""
        self.prev_position = None
        self.prev_velocity = None
        self.prev_orientation = None
        self.acc_bias = self._random_normal(0, 0.05, 3)
        self.gyro_bias = self._random_normal(0, 0.01, 3)
        
    def generate_imu_data(self, position, orientation, dt, velocity=None,
                          acceleration=None, angular_velocity=None):
        """
        Generate realistic IMU measurements from tag kinematics.
        
        Args:
            position: Position object with x, y, z coordinates
            orientation: Tag orientation in radians (yaw angle)
            dt: Time step since last measurement
            velocity: optional world-frame velocity vector (m/s)
            acceleration: optional world-frame linear acceleration vector (m/s^2)
            angular_velocity: optional world-frame angular velocity vector (rad/s)
            
        Returns:
            tuple: (specific_force_body[3], angular_velocity_body[3])
        """
        if dt is None or not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"IMU dt must be finite and positive, got {dt}")
        if not np.isfinite(orientation):
            raise ValueError(f"IMU orientation must be finite, got {orientation}")

        current_pos = np.array([position.x, position.y, position.z])
        if not np.all(np.isfinite(current_pos)):
            raise ValueError(f"IMU position contains non-finite values: {current_pos}")

        current_velocity = self._as_vector(velocity) if velocity is not None else None
        linear_acceleration_world = self._as_vector(acceleration) if acceleration is not None else None
        angular_velocity_world = self._as_vector(angular_velocity) if angular_velocity is not None else None
        
        # Initialize on first call. With no previous velocity, finite-difference
        # acceleration is intentionally zero rather than assuming the tag started
        # from rest, which is the source of startup acceleration spikes.
        if self.prev_position is None:
            self.prev_position = current_pos
            self.prev_velocity = current_velocity
            self.prev_orientation = orientation
            if linear_acceleration_world is None:
                linear_acceleration_world = np.zeros(3)
            if angular_velocity_world is None:
                angular_velocity_world = np.zeros(3)
            return self._measure(linear_acceleration_world, angular_velocity_world, orientation, dt)
        
        # Calculate velocity and acceleration in world frame when the caller does
        # not provide ground truth. The first differentiated velocity sample is
        # treated as initialization, not as evidence of acceleration from rest.
        if current_velocity is None:
            current_velocity = (current_pos - self.prev_position) / dt
        if linear_acceleration_world is None:
            if self.prev_velocity is None:
                linear_acceleration_world = np.zeros(3)
            else:
                linear_acceleration_world = (current_velocity - self.prev_velocity) / dt
        
        if angular_velocity_world is None:
            delta_yaw = self._wrap_angle(orientation - self.prev_orientation)
            angular_velocity_world = np.array([0.0, 0.0, delta_yaw / dt])
        
        # Update previous state
        self.prev_position = current_pos
        self.prev_velocity = current_velocity
        self.prev_orientation = orientation

        return self._measure(linear_acceleration_world, angular_velocity_world, orientation, dt)

    def _measure(self, linear_acceleration_world, angular_velocity_world, orientation, dt):
        R_world_to_body = self._world_to_body_matrix(orientation)

        # Accelerometers measure specific force: f = a - g. Since world z is up,
        # gravity is [0, 0, -g], making stationary level output [0, 0, +g].
        gravity_world = np.array([0.0, 0.0, -self.g])
        specific_force_world = linear_acceleration_world - gravity_world
        specific_force_body = R_world_to_body @ specific_force_world

        angular_velocity_body = R_world_to_body @ angular_velocity_world

        # Bias random walk is scaled by sqrt(dt), as a discrete-time random walk.
        if self.bias_instability > 0.0:
            walk_scale = self.bias_instability * np.sqrt(dt)
            self.acc_bias += self._random_normal(0, walk_scale, 3)
            self.gyro_bias += self._random_normal(0, walk_scale / 10.0, 3)

        acc_measured = (
            specific_force_body * (1.0 + self.acc_scale_factor)
            + self.acc_bias
            + self._random_normal(0, self.acc_noise_std, 3)
        )
        gyro_measured = (
            angular_velocity_body * (1.0 + self.gyro_scale_factor)
            + self.gyro_bias
            + self._random_normal(0, self.gyro_noise_std, 3)
        )

        if not np.all(np.isfinite(acc_measured)) or not np.all(np.isfinite(gyro_measured)):
            raise ValueError("IMU generated non-finite measurement")
        
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
