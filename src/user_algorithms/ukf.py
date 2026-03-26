import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class UkfAlgorithm(BaseLocalizationAlgorithm):
    """
    Unscented Kalman Filter (UKF) for 2D tag localization.

    State vector : [x, y, vx, vy]
    Measurements : distances from the tag to each anchor  d_i = sqrt((x-ax_i)^2 + (y-ay_i)^2)
    """

    # ------------------------------------------------------------------ #
    #  Noise tuning                                                        #
    # ------------------------------------------------------------------ #
    PROCESS_NOISE_POS = 0.1
    PROCESS_NOISE_VEL = 1.0
    MEASUREMENT_NOISE = 0.15
    MIN_ANCHOR_DIST   = 1e-6

    # ------------------------------------------------------------------ #
    #  Unscented transform parameters                                      #
    # ------------------------------------------------------------------ #
    ALPHA = 1e-3
    BETA  = 2.0
    KAPPA = 0.0

    # ------------------------------------------------------------------ #
    #  BaseLocalizationAlgorithm interface                                #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "UKF"

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

        n = 4  # state dimension

        # ── 1. Initialisation ───────────────────────────────────────────
        if not initialized:
            state, covariance, Q, R = self._initialise(measurements)
            initialized = True

        # ── 2. Compute UKF weights ───────────────────────────────────────
        Wm, Wc, lam = self._compute_weights(n)

        # ── 3. Prediction ───────────────────────────────────────────────
        state, covariance = self._predict(state, covariance, dt, Q, Wm, Wc, lam, n)

        # ── 4. Measurement update ────────────────────────────────────────
        state, covariance = self._update(
            state, covariance, measurements, anchors, R, Wm, Wc, lam, n
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

    def _initialise(self, measurements):
        state      = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        covariance = np.diag([5.0, 5.0, 10.0, 10.0])
        Q          = self._build_Q(dt=0.1)
        R          = np.eye(len(measurements)) * self.MEASUREMENT_NOISE**2
        return state, covariance, Q, R

    # ── Unscented transform weights ──────────────────────────────────────

    def _compute_weights(self, n: int):
        lam   = self.ALPHA**2 * (n + self.KAPPA) - n

        Wm    = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        Wc    = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1.0 - self.ALPHA**2 + self.BETA)

        return Wm, Wc, lam

    # ── Sigma point generation ───────────────────────────────────────────

    def _sigma_points(self, state: np.ndarray, P: np.ndarray, lam: float, n: int) -> np.ndarray:
        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = state

        try:
            L = np.linalg.cholesky((n + lam) * P)
        except np.linalg.LinAlgError:
            P_reg = P + np.eye(n) * 1e-6
            L = np.linalg.cholesky((n + lam) * P_reg)

        for i in range(n):
            sigma[i + 1]     = state + L[:, i]
            sigma[n + i + 1] = state - L[:, i]

        return sigma   # shape (2n+1, n)

    # ── Process model ────────────────────────────────────────────────────

    def _build_Q(self, dt: float) -> np.ndarray:
        sp   = self.PROCESS_NOISE_POS
        sv   = self.PROCESS_NOISE_VEL
        q_1d = np.array([
            [dt**4 / 4 * sp**2,   dt**3 / 2 * sp * sv],
            [dt**3 / 2 * sp * sv, dt**2 * sv**2],
        ])
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = q_1d
        Q[np.ix_([1, 3], [1, 3])] = q_1d
        return Q

    def _f(self, state: np.ndarray, dt: float) -> np.ndarray:
        F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)
        return F @ state

    def _predict(self, state, P, dt, Q, Wm, Wc, lam, n):
        sigma      = self._sigma_points(state, P, lam, n)              # (2n+1, n)
        sigma_pred = np.array([self._f(s, dt) for s in sigma])         # (2n+1, n)
        x_pred     = np.sum(Wm[:, None] * sigma_pred, axis=0)          # (n,)
        diff       = sigma_pred - x_pred                                # (2n+1, n)
        P_pred     = np.einsum('i,ij,ik->jk', Wc, diff, diff) + Q     # (n, n)
        return x_pred, P_pred

    # ── Measurement model ────────────────────────────────────────────────

    def _h(self, state: np.ndarray, anchors) -> np.ndarray:
        z = np.zeros(len(anchors))
        for i, anchor in enumerate(anchors):
            dx   = state[0] - float(anchor.position.x)
            dy   = state[1] - float(anchor.position.y)
            z[i] = np.sqrt(dx**2 + dy**2)
        return z

    # ── Measurement update ───────────────────────────────────────────────

    def _update(self, state, P, measurements, anchors, R, Wm, Wc, lam, n):
        if measurements is None or anchors is None:
            return state, P

        z_meas = np.array([float(z) for z in measurements])

        # Sigma points from predicted state
        sigma  = self._sigma_points(state, P, lam, n)              # (2n+1, n)

        # Propagate through h
        Z_sigma = np.array([self._h(s, anchors) for s in sigma])   # (2n+1, m)

        # Predicted measurement mean
        z_pred = np.sum(Wm[:, None] * Z_sigma, axis=0)             # (m,)

        # Innovation
        y = z_meas - z_pred                                        # (m,)

        # Predicted measurement covariance Pzz
        dZ   = Z_sigma - z_pred                                    # (2n+1, m)
        P_zz = np.einsum('i,ij,ik->jk', Wc, dZ, dZ) + R          # (m, m)

        # Cross-covariance Pxz
        dX   = sigma - state                                       # (2n+1, n)
        P_xz = np.einsum('i,ij,ik->jk', Wc, dX, dZ)              # (n, m)

        # Kalman gain
        K = P_xz @ np.linalg.inv(P_zz)                            # (n, m)

        # State & covariance update
        state = state + K @ y
        P     = P - K @ P_zz @ K.T

        # Symmetrise P
        P = 0.5 * (P + P.T)

        return state, P