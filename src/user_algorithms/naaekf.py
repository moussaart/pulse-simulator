import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class NaaekfAlgorithm(BaseLocalizationAlgorithm):
    """
    NLOS-Aware Adaptive Extended Kalman Filter (NA-AEKF) for 2D tag localization.

    State vector : [x, y, vx, vy]
    Measurements : distances from the tag to each anchor

    Extends the AEKF with an NLOS-gated inflation of the adaptive R term: for
    any anchor flagged as NLOS, the freshly-estimated measurement-noise
    variance r_i is scaled up by LAMBDA_NLOS before the usual exponential
    smoothing is applied. This down-weights NLOS measurements in the
    subsequent Kalman gain without discarding them outright.
    """

    PROCESS_NOISE_POS = 0.1
    PROCESS_NOISE_VEL = 1.0
    MEASUREMENT_NOISE = 0.15

    ALPHA = 0.3   # smoothing factor for R
    BETA  = 0.5   # smoothing factor for Q

    LAMBDA_NLOS = 2.0   # inflation factor applied to r_i when anchor i is NLOS

    # ------------------------------------------------------------------ #
    #  BaseLocalizationAlgorithm interface                                #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "NLOS-Aware Adaptive Extended Kalman Filter"

    def initialize(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    #  Main update                                                         #
    # ------------------------------------------------------------------ #

    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        measurements = input_data.measurements
        anchors      = input_data.anchors
        dt           = input_data.dt
        # NLOS Status (0=LOS, 1=NLOS)
        is_los = input_data.is_los

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

        # ── 2. Prediction ───────────────────────────────────────────────
        state, covariance = self._predict(state, covariance, dt, Q)

        # ── 3. NLOS-aware adaptive measurement update ───────────────────
        state, covariance, Q, R = self._update(
            state, covariance, measurements, anchors, covariance, Q, R, is_los
        )

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
        state      = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        covariance = np.diag([5.0, 5.0, 10.0, 10.0])
        Q          = self._build_Q(dt=input_data.dt)
        R          = np.eye(len(measurements)) * self.MEASUREMENT_NOISE**2
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
        return np.array([dx / d, dy / d, 0.0, 0.0])

    # ── NLOS gating helper ────────────────────────────────────────────────

    def _nlos_mask(self, is_los, n) -> np.ndarray:
        """
        Returns a boolean array of length n, True where the measurement is NLOS.

        `is_los` is expected to follow the same convention as the comment in
        `update()`: 0 = LOS, 1 = NLOS, despite the `is_los` name. If it is
        missing or malformed, every measurement is treated as LOS (i.e. no
        inflation is applied) so the filter degrades gracefully to the
        original AEKF behaviour rather than failing.
        """
        if is_los is None:
            return np.zeros(n, dtype=bool)
        flags = np.asarray(is_los).reshape(-1)
        if flags.shape[0] != n:
            return np.zeros(n, dtype=bool)
        return flags.astype(bool)

    # ── Adaptive update ──────────────────────────────────────────────────

    def _update(self, state, P, measurements, anchors, P_pred, Q, R, is_los):
        """
        Joint NLOS-aware adaptive update:
          - Builds full H and innovation vector y
          - Computes r_i,new per anchor from innovation statistics
          - Inflates r_i,new by LAMBDA_NLOS wherever the anchor is flagged NLOS
          - Smooths the resulting R via exponential averaging
          - Adapts Q (unchanged from AEKF)
          - Applies standard EKF correction
        """
        if measurements is None or anchors is None:
            return state, P, Q, R

        n = len(anchors)
        nlos_mask = self._nlos_mask(is_los, n)

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

        # ── Adaptive R update with NLOS gating ──────────────────────────
        C_innov  = np.outer(y_vec, y_vec)                       # y·yᵀ
        diag_new = np.abs(np.diag(C_innov) - np.diag(H @ P_pred @ H.T))  # r_i,new per anchor

        # Per-measurement NLOS inflation: r_i,new *= LAMBDA_NLOS where NLOS
        diag_new = np.where(nlos_mask, self.LAMBDA_NLOS * diag_new, diag_new)

        R_new = np.diag(diag_new)                               # PSD by construction
        R     = self.ALPHA * R + (1 - self.ALPHA) * R_new       # exponential smoothing

        # ── Adaptive Q update (unchanged) ────────────────────────────────
        norm_y  = np.linalg.norm(y_vec)
        gamma   = max(1.0, norm_y / n)                          # scaling coefficient
        Q_new   = gamma * np.eye(4)                             # process noise magnitude
        Q       = self.BETA * Q + (1 - self.BETA) * Q_new      # exponential smoothing

        # ── EKF correction ───────────────────────────────────────────────
        S     = H @ P_pred @ H.T + R                            # innovation covariance (n×n)
        K     = P_pred @ H.T @ np.linalg.inv(S)                # Kalman gain (4×n)
        state = state + K @ y_vec                               # state update
        P     = (np.eye(4) - K @ H) @ P_pred                   # covariance update

        return state, P, Q, R