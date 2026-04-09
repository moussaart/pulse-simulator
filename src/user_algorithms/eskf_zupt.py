import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class EskfZuptAlgorithm(BaseLocalizationAlgorithm):
    """
    Indoor localization using a full Error-State Kalman Filter (ESKF)
    with Zero Velocity Update (ZUPT) — IMU data only.

    State vector (9-DOF):
        [x, y, z,        ← position   (m)
         vx, vy, vz,     ← velocity   (m/s)
         roll, pitch, yaw] ← Euler angles (rad)

    Pipeline per step:
      1. Attitude propagation via gyro integration (Euler angles, small-angle).
      2. Gravity compensation: rotate sensor accel to world frame, subtract g.
      3. Double integration: accel → velocity → position.
      4. Covariance propagation with 9×9 process noise Q.
      5. ZUPT detector: sliding-window variance of ||accel|| norm.
      6. When stance detected → Kalman update that zeros vx, vy, vz
         and slightly corrects attitude drift.
    """

    # ── Defaults (all overridable via input_data.params) ──────────────
    DEFAULT_ZUPT_WINDOW     = 7
    DEFAULT_ZUPT_THRESHOLD  = 0.05      # m²/s⁴  accel-norm variance gate
    DEFAULT_GRAVITY         = 9.81      # m/s²
    DEFAULT_Q_POS           = 1e-6      # process noise – position
    DEFAULT_Q_VEL           = 1e-3      # process noise – velocity
    DEFAULT_Q_ATT           = 1e-5      # process noise – attitude
    DEFAULT_R_ZUPT          = 1e-4      # measurement noise – ZUPT velocity

    # ── Lifecycle ─────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "ESKF + ZUPT (IMU only)"

    def initialize(self) -> None:
        self._accel_norm_buffer: list[float] = []

    # ── Main update ───────────────────────────────────────────────────

    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:

        # ── 0. Resolve params ──────────────────────────────────────────
        p  = input_data.params or {}
        zupt_window    = int(p.get("zupt_window",    self.DEFAULT_ZUPT_WINDOW))
        zupt_threshold = float(p.get("zupt_threshold", self.DEFAULT_ZUPT_THRESHOLD))
        gravity        = float(p.get("gravity",        self.DEFAULT_GRAVITY))
        q_pos          = float(p.get("q_pos",          self.DEFAULT_Q_POS))
        q_vel          = float(p.get("q_vel",          self.DEFAULT_Q_VEL))
        q_att          = float(p.get("q_att",          self.DEFAULT_Q_ATT))
        r_zupt         = float(p.get("r_zupt",         self.DEFAULT_R_ZUPT))

        dt          = input_data.dt
        imu_on      = input_data.imu_data_on
        accel_raw   = input_data.accel
        gyro_raw    = input_data.gyro

        state       = input_data.state
        covariance  = input_data.covariance
        Q           = input_data.Q
        R           = input_data.R
        initialized = input_data.initialized

        # ── 1. First-call initialisation ───────────────────────────────
        if not initialized:
            # state = [x, y, z, vx, vy, vz, roll, pitch, yaw]
            state = np.zeros(9)

            # Seed XY from tag if available
            if input_data.tag is not None and input_data.tag.position is not None:
                state[0] = input_data.tag.position.x
                state[1] = input_data.tag.position.y

            covariance = np.eye(9) * 1.0

            Q = np.diag([
                q_pos, q_pos, q_pos,   # position
                q_vel, q_vel, q_vel,   # velocity
                q_att, q_att, q_att,   # attitude (roll, pitch, yaw)
            ])

            # ZUPT: we "measure" vx=vy=vz=0  →  3×3 noise matrix
            R = np.eye(3) * r_zupt

            self._accel_norm_buffer = []
            initialized = True

        # ── 2. Guard: IMU required ─────────────────────────────────────
        if not imu_on or accel_raw is None or gyro_raw is None:
            x, y = float(state[0]), float(state[1])
            return self._pack(x, y, state, covariance, Q, R,
                              initialized, input_data, False)

        accel = np.asarray(accel_raw, dtype=float)   # (3,)
        gyro  = np.asarray(gyro_raw,  dtype=float)   # (3,)

        # ── 3. Unpack state ────────────────────────────────────────────
        pos   = state[0:3].copy()   # [x, y, z]
        vel   = state[3:6].copy()   # [vx, vy, vz]
        euler = state[6:9].copy()   # [roll, pitch, yaw]

        roll, pitch, yaw = euler

        # ── 4. Attitude propagation (Euler angle integration) ──────────
        #
        #  ┌ roll  ┐   ┌1  sin(r)tan(p)  cos(r)tan(p)┐   ┌ gx ┐
        #  │ pitch │ = │0  cos(r)        -sin(r)      │ × │ gy │ × dt
        #  └ yaw   ┘   └0  sin(r)/cos(p)  cos(r)/cos(p)┘   └ gz ┘
        #
        cos_r, sin_r = np.cos(roll),  np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        tan_p = np.tan(pitch)

        # Guard against gimbal lock (|pitch| near 90°)
        if abs(cos_p) < 1e-6:
            cos_p = np.sign(cos_p) * 1e-6

        W = np.array([
            [1,  sin_r * tan_p,  cos_r * tan_p],
            [0,  cos_r,         -sin_r        ],
            [0,  sin_r / cos_p,  cos_r / cos_p],
        ])

        euler_dot = W @ gyro
        euler     = euler + euler_dot * dt

        # Wrap angles to (-π, π)
        euler = (euler + np.pi) % (2 * np.pi) - np.pi
        roll, pitch, yaw = euler

        # ── 5. Body → World rotation matrix (ZYX convention) ──────────
        #  R_bw rotates a vector from sensor/body frame into world frame.
        cos_y, sin_y = np.cos(yaw),   np.sin(yaw)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_r, sin_r = np.cos(roll),  np.sin(roll)

        R_bw = np.array([
            [cos_y*cos_p,
             cos_y*sin_p*sin_r - sin_y*cos_r,
             cos_y*sin_p*cos_r + sin_y*sin_r],

            [sin_y*cos_p,
             sin_y*sin_p*sin_r + cos_y*cos_r,
             sin_y*sin_p*cos_r - cos_y*sin_r],

            [-sin_p,
             cos_p*sin_r,
             cos_p*cos_r],
        ])

        # ── 6. Gravity compensation ────────────────────────────────────
        g_world     = np.array([0.0, 0.0, gravity])
        accel_world = R_bw @ accel - g_world   # specific force → linear accel

        # ── 7. Double integration (velocity & position) ────────────────
        pos = pos + vel * dt + 0.5 * accel_world * dt**2
        vel = vel + accel_world * dt

        # ── 8. State transition matrix F (9×9 linearised) ─────────────
        #
        #  Jacobian of f(state) w.r.t. state, evaluated at current euler.
        #  Only the pos←vel and vel←att cross-terms are non-trivial.
        #
        F = np.eye(9)

        # pos += vel * dt
        F[0:3, 3:6] = np.eye(3) * dt

        # vel += R_bw(euler) @ accel * dt
        # ∂(R_bw @ a)/∂(euler) is a 3×3 Jacobian – computed analytically below.
        dR_droll, dR_dpitch, dR_dyaw = self._dR_deuler(roll, pitch, yaw)
        J_vel_att        = np.column_stack([
            dR_droll  @ accel,
            dR_dpitch @ accel,
            dR_dyaw   @ accel,
        ]) * dt                                  # shape (3, 3)
        F[3:6, 6:9] = J_vel_att

        # euler += W @ gyro * dt  (attitude←attitude cross-coupling)
        dW_droll, dW_dpitch = self._dW_deuler(roll, pitch, gyro)
        F[6:9, 6:9] = np.eye(3) + np.column_stack([
            dW_droll,
            dW_dpitch,
            np.zeros(3),
        ]) * dt

        # ── 9. Covariance propagation ──────────────────────────────────
        covariance = F @ covariance @ F.T + Q

        # ── 10. ZUPT detector ──────────────────────────────────────────
        accel_norm = float(np.linalg.norm(accel))
        self._accel_norm_buffer.append(accel_norm)
        if len(self._accel_norm_buffer) > zupt_window:
            self._accel_norm_buffer.pop(0)

        norm_var      = float(np.var(self._accel_norm_buffer))
        zupt_ready    = len(self._accel_norm_buffer) == zupt_window
        zupt_triggered = zupt_ready and (norm_var < zupt_threshold)

        # ── 11. ZUPT Kalman measurement update ─────────────────────────
        #
        #  Observation model: z = H @ state + noise
        #  z = [0, 0, 0]   (measured velocities during stance)
        #  H picks out vx, vy, vz from the 9-DOF state.
        #
        if zupt_triggered:
            H = np.zeros((3, 9))
            H[0, 3] = 1.0   # vx
            H[1, 4] = 1.0   # vy
            H[2, 5] = 1.0   # vz

            S          = H @ covariance @ H.T + R          # (3×3) innovation cov
            K          = covariance @ H.T @ np.linalg.inv(S)  # (9×3) Kalman gain

            z          = np.zeros(3)
            innovation = z - np.array([vel[0], vel[1], vel[2]])

            correction = K @ innovation                     # (9,)

            # Apply correction to all states
            pos   = pos   + correction[0:3]
            vel   = vel   + correction[3:6]
            euler = euler + correction[6:9]
            euler = (euler + np.pi) % (2 * np.pi) - np.pi  # re-wrap

            # Joseph form for numerical stability:  P = (I-KH)P(I-KH)ᵀ + KRKᵀ
            IKH        = np.eye(9) - K @ H
            covariance = IKH @ covariance @ IKH.T + K @ R @ K.T

        # ── 12. Pack updated state ─────────────────────────────────────
        state[0:3] = pos
        state[3:6] = vel
        state[6:9] = euler

        x, y = float(state[0]), float(state[1])

        extra = {
            "zupt_triggered": zupt_triggered,
            "norm_variance":  norm_var,
            "zupt_threshold": zupt_threshold,
            "roll_deg":       float(np.degrees(euler[0])),
            "pitch_deg":      float(np.degrees(euler[1])),
            "yaw_deg":        float(np.degrees(euler[2])),
            "velocity":       vel.tolist(),
            "accel_world":    accel_world.tolist(),
        }

        return self._pack(x, y, state, covariance, Q, R,
                          initialized, input_data, zupt_triggered, extra)

    # ── Analytical Jacobians ──────────────────────────────────────────

    @staticmethod
    def _dR_deuler(roll, pitch, yaw):
        """
        Partial derivatives of R_bw w.r.t. roll, pitch, yaw.
        Each return value is a (3×3) matrix such that:
            dR/dangle @ accel  gives the velocity sensitivity column.
        """
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)

        dR_droll = np.array([
            [0,
             cy*sp*cr + sy*sr,
             -cy*sp*sr + sy*cr],
            [0,
             sy*sp*cr - cy*sr,
             -sy*sp*sr - cy*cr],
            [0,
             cp*cr,
             -cp*sr],
        ])

        dR_dpitch = np.array([
            [-cy*sp,
              cy*cp*sr,
              cy*cp*cr],
            [-sy*sp,
              sy*cp*sr,
              sy*cp*cr],
            [-cp,
             -sp*sr,
             -sp*cr],
        ])

        dR_dyaw = np.array([
            [-sy*cp,
             -sy*sp*sr - cy*cr,
             -sy*sp*cr + cy*sr],
            [ cy*cp,
              cy*sp*sr - sy*cr,
              cy*sp*cr + sy*sr],
            [0, 0, 0],
        ])

        return dR_droll, dR_dpitch, dR_dyaw

    @staticmethod
    def _dW_deuler(roll, pitch, gyro):
        """
        Partial derivatives of (W @ gyro) w.r.t. roll and pitch only.
        Yaw does not appear in W so its column is zero (handled at call site).
        Returns two (3,) vectors: d(W@g)/d_roll, d(W@g)/d_pitch.
        """
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        gx, gy, gz = gyro

        tan_p = np.tan(pitch)
        sec2_p = 1.0 / max(cp**2, 1e-12)

        dWg_droll = np.array([
            ( cr*tan_p)*gy + (-sr*tan_p)*gz,
            (-sr)*gy       + (-cr)*gz,
            ( cr/cp)*gy    + (-sr/cp)*gz,
        ])

        dWg_dpitch = np.array([
            sr*sec2_p*gy  + cr*sec2_p*gz,
            0.0,
            sr*sp/cp**2*gy + cr*sp/cp**2*gz,
        ])

        return dWg_droll, dWg_dpitch

    # ── Output helper ─────────────────────────────────────────────────

    @staticmethod
    def _pack(x, y, state, cov, Q, R, initialized,
              input_data, zupt_triggered, extra=None) -> AlgorithmOutput:
        ed = {"zupt_triggered": zupt_triggered}
        if extra:
            ed.update(extra)
        return AlgorithmOutput(
            position            = (x, y),
            state               = state,
            covariance          = cov,
            initialized         = initialized,
            previous_state      = input_data.state,
            previous_covariance = input_data.covariance,
            Q                   = Q,
            R                   = R,
            extra_data          = ed,
        )