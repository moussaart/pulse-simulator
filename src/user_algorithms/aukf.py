import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class AukfAlgorithm(BaseLocalizationAlgorithm):
    """
    Unscented Kalman Filter (UKF) for 2D tag localization.

    State vector : [x, y, vx, vy]
    Measurements : distances from the tag to each anchor  d_i = sqrt((x-ax_i)^2 + (y-ay_i)^2)

    Instead of linearising the measurement function via a Jacobian (EKF),
    the UKF propagates a set of deterministically chosen sigma points through
    the non-linear measurement function, then recovers mean and covariance
    from the transformed points — no derivatives required.
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
    # n  = state dimension = 4
    # α  controls spread of sigma points around mean  (1e-3 … 1)
    # β  encodes prior knowledge of distribution (2 optimal for Gaussian)
    # κ  secondary scaling parameter (0 or 3-n)
    ALPHA = 1e-3
    BETA  = 2.0
    KAPPA = 0.0

    # ------------------------------------------------------------------ #
    #  BaseLocalizationAlgorithm interface                                #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "AUKF"

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
        state, covariance, Q, R = self._update(
            state, covariance, measurements, anchors, Q, R, Wm, Wc, lam, n
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
        """
        Compute mean (Wm) and covariance (Wc) weights for 2n+1 sigma points.

        λ = α²(n + κ) - n
        Wm_0 = λ / (n + λ)
        Wc_0 = λ / (n + λ) + (1 - α² + β)
        Wm_i = Wc_i = 1 / (2(n + λ))   for i = 1 … 2n
        """
        lam = self.ALPHA**2 * (n + self.KAPPA) - n

        Wm    = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        Wc    = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1.0 - self.ALPHA**2 + self.BETA)

        return Wm, Wc, lam

    # ── Sigma point generation ───────────────────────────────────────────

    def _sigma_points(self, state: np.ndarray, P: np.ndarray, lam: float, n: int) -> np.ndarray:
        """
        Generate 2n+1 sigma points around the mean state.

        X_0     = state
        X_i     = state + sqrt((n+λ) P)_i      i = 1 … n
        X_{n+i} = state - sqrt((n+λ) P)_i      i = 1 … n
        """
        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = state

        # Cholesky of scaled covariance  →  numerically stable square root
        try:
            L = np.linalg.cholesky((n + lam) * P)
        except np.linalg.LinAlgError:
            # fallback: regularise P if not positive-definite
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
        """Constant-velocity process model  x_{k+1} = F · x_k."""
        F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)
        return F @ state

    def _predict(self, state, P, dt, Q, Wm, Wc, lam, n):
        """
        UKF prediction step:
          1. Generate sigma points from current state/covariance.
          2. Propagate each through the process model f(·).
          3. Recover predicted mean and covariance.
        """
        # Sigma points
        sigma = self._sigma_points(state, P, lam, n)   # (2n+1, n)

        # Propagate through f
        sigma_pred = np.array([self._f(s, dt) for s in sigma])  # (2n+1, n)

        # Predicted mean
        x_pred = np.sum(Wm[:, None] * sigma_pred, axis=0)       # (n,)

        # Predicted covariance
        diff   = sigma_pred - x_pred                             # (2n+1, n)
        P_pred = np.einsum('i,ij,ik->jk', Wc, diff, diff) + Q  # (n, n)

        return x_pred, P_pred

    # ── Measurement model ────────────────────────────────────────────────

    def _h(self, state: np.ndarray, anchors) -> np.ndarray:
        """
        Non-linear measurement function.
        Returns expected distances to all anchors for a given state.
        """
        z = np.zeros(len(anchors))
        for i, anchor in enumerate(anchors):
            dx   = state[0] - float(anchor.position.x)
            dy   = state[1] - float(anchor.position.y)
            z[i] = np.sqrt(dx**2 + dy**2)
        return z

    # ── Measurement update ───────────────────────────────────────────────

    def _update(self, state, P, measurements, anchors, Q, R, Wm, Wc, lam, n):
        """
        UKF measurement update step:
          1. Generate sigma points from predicted state/covariance.
          2. Propagate each through measurement model h(·).
          3. Recover predicted measurement mean and covariance.
          4. Compute cross-covariance and Kalman gain.
          5. Update state and covariance.
          6. Adapt R and Q from innovation statistics.
        """
        if measurements is None or anchors is None:
            return state, P, Q, R

        m     = len(anchors)
        z_meas = np.array([float(z) for z in measurements])

        # ── Sigma points from predicted state ────────────────────────────
        sigma = self._sigma_points(state, P, lam, n)             # (2n+1, n)

        # ── Propagate sigma points through h ─────────────────────────────
        Z_sigma = np.array([self._h(s, anchors) for s in sigma]) # (2n+1, m)

        # ── Predicted measurement mean ────────────────────────────────────
        z_pred = np.sum(Wm[:, None] * Z_sigma, axis=0)           # (m,)

        # ── Innovation ────────────────────────────────────────────────────
        y = z_meas - z_pred                                       # (m,)

        # ── Predicted measurement covariance Pzz ─────────────────────────
        dZ    = Z_sigma - z_pred                                  # (2n+1, m)
        P_zz  = np.einsum('i,ij,ik->jk', Wc, dZ, dZ) + R        # (m, m)

        # ── Cross-covariance Pxz ──────────────────────────────────────────
        dX    = sigma - state                                     # (2n+1, n)
        P_xz  = np.einsum('i,ij,ik->jk', Wc, dX, dZ)            # (n, m)

        # ── Kalman gain ───────────────────────────────────────────────────
        K = P_xz @ np.linalg.inv(P_zz)                           # (n, m)

        # ── State & covariance update ─────────────────────────────────────
        state = state + K @ y
        P     = P - K @ P_zz @ K.T

        # ── Symmetrise P to avoid numerical drift ─────────────────────────
        P = 0.5 * (P + P.T)

        # ── Adaptive R update (Section 5.1) ──────────────────────────────
        C_innov = np.outer(y, y)
        R_new   = C_innov - (P_zz - R)          # C_innov - H·P·Hᵀ equivalent
        R_new   = np.diag(np.abs(np.diag(R_new)))
        R       = 0.5 * R + 0.5 * R_new

        # ── Adaptive Q update (Section 5.2) ──────────────────────────────
        norm_y = np.linalg.norm(y)
        gamma  = max(1.0, norm_y / m)
        Q_new  = gamma * np.eye(n)
        Q      = 0.5 * Q + 0.5 * Q_new

        return state, P, Q, R