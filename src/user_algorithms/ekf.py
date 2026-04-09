import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class EkfTestAlgorithm(BaseLocalizationAlgorithm):
    """
    Extended Kalman Filter (EKF) for 2D tag localization.

    State vector : [x, y, vx, vy]  (position + velocity)
    Measurements : distances from the tag to each anchor  d_i = sqrt((x - ax_i)^2 + (y - ay_i)^2)

    The motion model is a constant-velocity model (linear), so the
    prediction step is a standard Kalman prediction.
    The measurement model is non-linear (Euclidean distance), so the
    EKF linearises it via the Jacobian H at each update step.
    """

    # ------------------------------------------------------------------ #
    #  Noise tuning – adjust to your hardware / environment               #
    # ------------------------------------------------------------------ #
    PROCESS_NOISE_POS   = 0.1   # [m]   std-dev of acceleration disturbance → position
    PROCESS_NOISE_VEL   = 1.0   # [m/s] std-dev of acceleration disturbance → velocity
    MEASUREMENT_NOISE   = 0.15  # [m]   std-dev of ranging measurement noise
    MIN_ANCHOR_DIST     = 1e-6  # guard against division by zero

    # ------------------------------------------------------------------ #
    #  BaseLocalizationAlgorithm interface                                #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "Extended Kalman Filter"

    def initialize(self) -> None:
        """Reset any persistent internal state."""
        pass  # all state lives in AlgorithmInput/Output – nothing extra needed

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        """
        Run one EKF cycle:
            1. Initialise on first call.
            2. Predict state forward by dt using the constant-velocity model.
            3. Update with all available distance measurements.
        """
        measurements = input_data.measurements   # list/array of distances to anchors
        anchors      = input_data.anchors         # list of anchor objects with .position
        dt           = input_data.dt

        state       = input_data.state
        covariance  = input_data.covariance
        initialized = input_data.initialized

        # ── 1. Initialisation ───────────────────────────────────────────
        if not initialized:
            state, covariance = self._initialise(input_data, anchors, measurements)
            initialized = True

        # ── 2. Prediction ───────────────────────────────────────────────
        state, covariance = self._predict(state, covariance, dt)

        # ── 3. Measurement update ────────────────────────────────────────
        state, covariance = self._update(state, covariance, measurements, anchors)

        x, y = float(state[0]), float(state[1])

        return AlgorithmOutput(
            position=(x, y),
            state=state,
            covariance=covariance,
            initialized=initialized,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _initialise(self, input_data, anchors, measurements):
        """
        Best-effort initial position.
        Priority:
            1. tag.position if already known
            2. Centroid of anchors weighted by inverse distance (closer anchor → more weight)
            3. Simple geometric centroid of all anchors
        """
        """ # Option 1 – use existing tag position
        if input_data.tag.position is not None:
            x0 = float(input_data.tag.position.x)
            y0 = float(input_data.tag.position.y)
        elif anchors and measurements is not None and len(measurements) == len(anchors):
            # Option 2 – inverse-distance weighted centroid
            weights = []
            for d in measurements:
                d = max(float(d), self.MIN_ANCHOR_DIST)
                weights.append(1.0 / d)
            w_sum = sum(weights)
            x0 = sum(w * float(a.position.x) for w, a in zip(weights, anchors)) / w_sum
            y0 = sum(w * float(a.position.y) for w, a in zip(weights, anchors)) / w_sum
        elif anchors:
            # Option 3 – simple centroid
            x0 = np.mean([float(a.position.x) for a in anchors])
            y0 = np.mean([float(a.position.y) for a in anchors])
        else:
            x0, y0 = 0.0, 0.0 """

        state = np.array([0, 0, 0.0, 0.0], dtype=float)

        # Initial covariance – high uncertainty on position, very high on velocity
        covariance = np.diag([5.0, 5.0, 10.0, 10.0])

        return state, covariance

    # ── Prediction (constant-velocity motion model) ─────────────────────

    def _build_F(self, dt: float) -> np.ndarray:
        """State transition matrix for constant-velocity model."""
        return np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

    def _build_Q(self, dt: float) -> np.ndarray:
        """
        Process noise covariance Q (discrete white-noise acceleration model).
        Assumes uncorrelated x / y axes.
        """
        sp = self.PROCESS_NOISE_POS
        sv = self.PROCESS_NOISE_VEL
        q_1d = np.array([
            [dt**4 / 4 * sp**2, dt**3 / 2 * sp * sv],
            [dt**3 / 2 * sp * sv, dt**2 * sv**2],
        ])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = q_1d   # x axis
        Q[np.ix_([1, 3], [1, 3])] = q_1d   # y axis
        return Q

    def _predict(self, state: np.ndarray, P: np.ndarray, dt: float):
        """EKF prediction step (linear model → no approximation needed)."""
        F = self._build_F(dt)
        Q = self._build_Q(dt)

        state_pred = F @ state
        P_pred     = F @ P @ F.T + Q
        return state_pred, P_pred

    # ── Measurement update ───────────────────────────────────────────────

    def _predicted_distance(self, state: np.ndarray, anchor) -> float:
        """Expected distance from current state to one anchor."""
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        return float(np.sqrt(dx**2 + dy**2))

    def _distance_jacobian_row(self, state: np.ndarray, anchor) -> np.ndarray:
        """
        Single row of the measurement Jacobian H for one anchor.
        ∂d/∂[x, y, vx, vy] = [(x-ax)/d, (y-ay)/d, 0, 0]
        """
        dx = state[0] - float(anchor.position.x)
        dy = state[1] - float(anchor.position.y)
        d  = max(np.sqrt(dx**2 + dy**2), self.MIN_ANCHOR_DIST)
        return np.array([dx / d, dy / d, 0.0, 0.0])

    def _update(self, state: np.ndarray, P: np.ndarray, measurements, anchors):
        """
        EKF measurement update step.
        Each valid (anchor, distance) pair is fused sequentially,
        which is equivalent to a joint update but numerically stabler.
        """
        if measurements is None or anchors is None:
            return state, P

        for anchor, z_raw in zip(anchors, measurements):
            z = float(z_raw)

            # Skip invalid measurements
            if np.isnan(z) or z <= 0:
                continue

            # Predicted measurement and Jacobian
            z_hat = self._predicted_distance(state, anchor)
            H     = self._distance_jacobian_row(state, anchor).reshape(1, -1)   # (1, 4)

            # Measurement noise (scalar)
            R = np.array([[self.MEASUREMENT_NOISE**2]])

            # Innovation
            y_innov = np.array([[z - z_hat]])                                   # (1, 1)

            # Innovation covariance
            S = H @ P @ H.T + R                                                 # (1, 1)

            # Kalman gain
            K = P @ H.T @ np.linalg.inv(S)                                     # (4, 1)

            # State & covariance update
            state = state + (K @ y_innov).flatten()
            P     = (np.eye(4) - K @ H) @ P

        return state, P