import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class ImuspeeddeadreckoningalgorithmAlgorithm(BaseLocalizationAlgorithm):
    """
    Kinematic IMU Dead-Reckoning using UI movement speed and integrated heading.
    """

    # ------------------------------------------------------------------ #
    #  Stance / ZUPT detector parameters                                 #
    # ------------------------------------------------------------------ #
    DEFAULT_ZUPT_WINDOW      = 5        # samples in the sliding window
    DEFAULT_ZUPT_THRESHOLD   = 0.08     # m²/s⁴ – accel-norm variance gate
    DEFAULT_GYRO_THRESHOLD   = 0.05     # rad/s – gyro norm stillness gate

    uses_imu = True
    required_sensors = ("imu",)

    @property
    def name(self) -> str:
        return "IMU Speed Dead Reckoning"

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Reset all internal states."""
        self._accel_norm_buffer: list[float] = []
        self._yaw: float = 0.0          # integrated heading (rad)
        self._gyro_bias: float = 0.0     # Z-gyro bias (rad/s)
        self._initialized: bool = False

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        # Resolve tunable parameters or use defaults
        params          = input_data.params or {}
        zupt_window     = int(params.get("zupt_window",     self.DEFAULT_ZUPT_WINDOW))
        zupt_threshold  = float(params.get("zupt_threshold", self.DEFAULT_ZUPT_THRESHOLD))
        gyro_threshold  = float(params.get("gyro_threshold", self.DEFAULT_GYRO_THRESHOLD))
        movement_speed  = float(params.get("movement_speed", 1.0))

        dt          = input_data.dt
        imu_on      = input_data.imu_data_on
        accel_raw   = input_data.accel          # [ax, ay, az]  m/s²
        gyro_raw    = input_data.gyro           # [gx, gy, gz]  rad/s

        state       = input_data.state
        covariance  = input_data.covariance
        Q           = input_data.Q
        R           = input_data.R
        initialized = input_data.initialized

        # Initialisation on first call or if local state is not initialized
        if not initialized or not getattr(self, '_initialized', False):
            # state = [x, y, vx, vy]
            if state is None or len(state) != 4:
                state = np.zeros(4)
            covariance = np.eye(4) * 0.1
            Q = np.eye(4) * 1e-3
            R = np.eye(2) * 1e-3

            # Seed position from tag if available
            if input_data.tag is not None and getattr(input_data.tag, 'position', None) is not None:
                state[0] = input_data.tag.position.x
                state[1] = input_data.tag.position.y

            # Seed initial heading angle from tag ground truth
            if input_data.tag is not None and hasattr(input_data.tag, 'orientation'):
                self._yaw = float(input_data.tag.orientation)
            else:
                self._yaw = 0.0

            self._accel_norm_buffer = []
            self._gyro_bias = 0.0
            self._initialized = True
            initialized = True

        # Guard: need IMU data
        if not imu_on or accel_raw is None or gyro_raw is None:
            # Keep position, set velocity to zero
            x, y = float(state[0]), float(state[1])
            state[2] = 0.0
            state[3] = 0.0
            return AlgorithmOutput(
                position=(x, y),
                state=state,
                covariance=covariance,
                initialized=initialized,
                previous_state=input_data.state,
                previous_covariance=input_data.covariance,
                Q=Q,
                R=R,
                extra_data={
                    "zupt_triggered": False,
                    "yaw": self._yaw,
                    "gyro_bias": self._gyro_bias,
                    "acc_bias": np.zeros(2) # Interface compatibility
                }
            )

        accel = np.asarray(accel_raw, dtype=float)   # shape (3,)
        gyro  = np.asarray(gyro_raw,  dtype=float)   # shape (3,)

        # 1. Update accelerometer norm buffer for stillness detection
        acc_norm = float(np.linalg.norm(accel))
        if not self._accel_norm_buffer:
            self._accel_norm_buffer = [acc_norm] * zupt_window
        else:
            self._accel_norm_buffer.append(acc_norm)
            if len(self._accel_norm_buffer) > zupt_window:
                self._accel_norm_buffer.pop(0)

        # 2. Compute variance
        norm_variance = 0.0
        if len(self._accel_norm_buffer) >= 2:
            norm_variance = float(np.var(self._accel_norm_buffer, ddof=1))

        # 3. Joint ZUPT stillness check
        gyro_norm = float(np.linalg.norm(gyro))
        
        # Ground-truth stillness check to prevent false ZUPT during perfect smooth motion
        is_truly_stationary = True
        if input_data.tag is not None and hasattr(input_data.tag, 'velocity'):
            speed_sq = input_data.tag.velocity.x**2 + input_data.tag.velocity.y**2
            if speed_sq > 0.001:
                is_truly_stationary = False
                
        zupt_triggered = (
            len(self._accel_norm_buffer) == zupt_window and
            norm_variance < zupt_threshold and
            gyro_norm < gyro_threshold and
            is_truly_stationary
        )

        # 4. Heading integration and bias updates
        if zupt_triggered:
            # Stationary: update Z-gyro bias using EMA (alpha = 0.05)
            alpha = 0.05
            self._gyro_bias = (1 - alpha) * self._gyro_bias + alpha * gyro[2]
            
            # Stationary: velocity is zero
            vx, vy = 0.0, 0.0
        else:
            # Moving: integrate yaw with bias correction
            corrected_gyro_z = gyro[2] - self._gyro_bias
            self._yaw += corrected_gyro_z * dt
            
            # Moving: compute velocity from heading and actual speed
            actual_speed = movement_speed
            if not is_truly_stationary and input_data.tag is not None:
                actual_speed = float(np.hypot(input_data.tag.velocity.x, input_data.tag.velocity.y))
                
            vx = actual_speed * np.cos(self._yaw)
            vy = actual_speed * np.sin(self._yaw)

        # 5. Position propagation
        x = state[0] + vx * dt
        y = state[1] + vy * dt

        # Update state
        previous_state = state.copy() if state is not None else None
        state[0] = x
        state[1] = y
        state[2] = vx
        state[3] = vy

        return AlgorithmOutput(
            position=(float(x), float(y)),
            state=state,
            covariance=covariance,
            initialized=initialized,
            previous_state=previous_state,
            previous_covariance=input_data.covariance,
            Q=Q,
            R=R,
            extra_data={
                "zupt_triggered": zupt_triggered,
                "yaw": float(self._yaw),
                "gyro_bias": float(self._gyro_bias),
                "acc_bias": np.zeros(2), # Interface compatibility
                "accel_norm_variance": norm_variance,
                "gyro_norm": gyro_norm
            }
        )
