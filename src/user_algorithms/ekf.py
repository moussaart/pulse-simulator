import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class EkfAlgorithm(BaseLocalizationAlgorithm):
    """
    Standard Extended Kalman Filter (EKF) for 2D tag localization.

    State vector : [x, y, vx, vy]
    Measurements : distances from the tag to each anchor

    Uses static process noise Q and measurement noise R without adaptive updating terms.
    """

    PROCESS_NOISE_POS = 0.1
    PROCESS_NOISE_VEL = 1.0
    MEASUREMENT_NOISE = 0.15

    # ------------------------------------------------------------------ #
    #  BaseLocalizationAlgorithm interface                                #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "Extended Kalman Filter"

    def initialize(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        measurements = input_data.measurements
        anchors      = input_data.anchors
        dt           = input_data.dt

        state       = input_data.state
        covariance  = input_data.covariance
        Q           = input_data.Q
        R           = input_data.R
        initialized = input_data.initialized

        # ── 1. Initialisation ───────────────────────────────────────────
        if not initialized or Q is None or R is None:
            s_init, c_init, q_init, r_init = self._initialise(input_data, anchors, measurements)
            if not initialized:
                state, covariance, Q, R = s_init, c_init, q_init, r_init
                initialized = True

        # Keep Q consistent with the current time step dt
        if dt is not None and dt > 0:
            Q = self._build_Q(dt)

        # ── 2. Prediction ───────────────────────────────────────────────
        state_pred, cov_pred = self._predict(state, covariance, dt, Q)

        # ── 3. Standard measurement update (no adaptive terms) ──────────
        state, covariance = self._update(state_pred, cov_pred, measurements, anchors, R)

        return AlgorithmOutput(
            position=(float(state[0]), float(state[1])),
            state=state,
            covariance=covariance,
            initialized=initialized,
            Q=Q,
            R=R,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _initialise(self, input_data, anchors, measurements):
        state      = np.array([0, 0, 0.0, 0.0], dtype=float)
        covariance = np.diag([5.0, 5.0, 10.0, 10.0])
        dt         = input_data.dt if (hasattr(input_data, 'dt') and input_data.dt) else 0.1
        Q          = self._build_Q(dt=dt)
        n_meas     = len(measurements) if measurements is not None else len(anchors)
        R          = np.eye(n_meas) * (self.MEASUREMENT_NOISE ** 2)
        return state, covariance, Q, R

    # ── Prediction ──────────────────────────────────────────────────────

    def _build_F(self, dt: float) -> np.ndarray:
        return np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

    def _build_Q(self, dt: float) -> np.ndarray:
        sp = self.PROCESS_NOISE_POS
        sv = self.PROCESS_NOISE_VEL
        q_1d = np.array([
            [dt**4 / 4 * sp**2,      dt**3 / 2 * sp * sv],
            [dt**3 / 2 * sp * sv,    dt**2 * sv**2],
        ])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = q_1d
        Q[np.ix_([1, 3], [1, 3])] = q_1d
        return Q

    def _predict(self, state, P, dt, Q):
        F          = self._build_F(dt)
        state_pred = F @ state
        P_pred     = F @ P @ F.T + Q
        return state_pred, P_pred

    # ── Measurement helpers ──────────────────────────────────────────────

    def _predicted_distance(self, state, anchor) -> float:
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        return float(np.sqrt(dx**2 + dy**2))

    def _distance_jacobian_row(self, state, anchor) -> np.ndarray:
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        d  = np.sqrt(dx**2 + dy**2)
        if d < 1e-6:
            return np.array([0.0, 0.0, 0.0, 0.0])
        return np.array([dx / d, dy / d, 0.0, 0.0])

    # ── Standard EKF Update ──────────────────────────────────────────────

    def _update(self, state, P_pred, measurements, anchors, R):
        """
        Standard EKF measurement update:
          - Builds measurement Jacobian matrix H and innovation vector y
          - Uses fixed measurement noise covariance R
          - Computes Kalman gain K = P_pred @ H.T @ inv(S)
          - Updates state and covariance
        """
        if measurements is None or anchors is None:
            return state, P_pred

        n = len(anchors)
        if n == 0:
            return state, P_pred

        if R is None or R.shape != (n, n):
            R = np.eye(n) * (self.MEASUREMENT_NOISE ** 2)

        # ── Build H (n×4) and innovation y (n,) ─────────────────────────
        H     = np.zeros((n, 4))
        y_vec = np.zeros(n)

        for i, (anchor, z_raw) in enumerate(zip(anchors, measurements)):
            z = float(z_raw)
            if np.isnan(z) or z <= 0:
                continue
            z_hat      = self._predicted_distance(state, anchor)
            H[i]       = self._distance_jacobian_row(state, anchor)
            y_vec[i]   = z - z_hat

        # ── EKF correction ───────────────────────────────────────────────
        S     = H @ P_pred @ H.T + R                           # innovation covariance (n×n)
        K     = P_pred @ H.T @ np.linalg.inv(S)               # Kalman gain (4×n)
        state = state + K @ y_vec                              # state update
        P     = (np.eye(4) - K @ H) @ P_pred                  # covariance update
        P     = 0.5 * (P + P.T)                                # preserve symmetry

        return state, P
