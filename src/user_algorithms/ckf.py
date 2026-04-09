import numpy as np
from src.core.localization.base_algorithm import BaseLocalizationAlgorithm, AlgorithmInput, AlgorithmOutput


class CkfAlgorithm(BaseLocalizationAlgorithm):
    """
    Cubature Kalman Filter (CKF) for 2-D tag localization.

    State vector : [x, y, vx, vy]  (position + velocity)
    Measurements : distances from the tag to each anchor
                   z_i = sqrt((x - ax_i)^2 + (y - ay_i)^2) + noise
    """

    # ------------------------------------------------------------------ #
    #  Tuning knobs – adjust to your deployment environment               #
    # ------------------------------------------------------------------ #
    _Q_DIAG = np.array([0.01, 0.01, 0.1, 0.1])   # process noise  (x,y,vx,vy)
    _R_STD  = 0.15                                  # ranging std-dev [m]

    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "Cubature Kalman Filter"

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Reset any persistent internal state."""
        pass

    # ------------------------------------------------------------------
    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        measurements = input_data.measurements   # list/dict of anchor distances
        anchors      = input_data.anchors         # anchor positions
        dt           = input_data.dt

        state      = input_data.state
        covariance = input_data.covariance
        initialized = input_data.initialized

        # ── 1. First-call initialisation ─────────────────────────────────
        if not initialized:
            x0, y0 = 0.0, 0.0
            state      = np.array([x0, y0, 0.0, 0.0], dtype=float)
            covariance = np.diag([1.0, 1.0, 1.0, 1.0])
            initialized = True

        # ── 2. Build anchor array & measurement vector ───────────────────
        anchor_positions, z_meas = self._parse_measurements(measurements, anchors)

        if anchor_positions is None or len(anchor_positions) == 0:
            # No valid ranging data – propagate only
            state, covariance = self._predict(state, covariance, dt)
            return AlgorithmOutput(
                position=(float(state[0]), float(state[1])),
                state=state,
                covariance=covariance,
                initialized=initialized,
            )

        # ── 3. CKF predict ───────────────────────────────────────────────
        state_pred, cov_pred = self._predict(state, covariance, dt)

        # ── 4. CKF update ────────────────────────────────────────────────
        state_upd, cov_upd = self._update(state_pred, cov_pred,
                                           z_meas, anchor_positions)

        return AlgorithmOutput(
            position=(float(state_upd[0]), float(state_upd[1])),
            state=state_upd,
            covariance=cov_upd,
            initialized=initialized,
        )

    # ================================================================== #
    #  CKF internals                                                       #
    # ================================================================== #

    # --- state dimension -----------------------------------------------
    _n = 4

    @property
    def _cubature_points(self):
        """Return the 2n cubature weights (all equal = 1/(2n)) and offsets."""
        n = self._n
        W = 1.0 / (2 * n)
        # Unit cubature directions: ±√n · e_i
        xi = np.sqrt(n) * np.vstack([np.eye(n), -np.eye(n)])   # (2n, n)
        return xi, W

    # --- process model -------------------------------------------------
    def _f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Constant-velocity motion model."""
        F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)
        return F @ x

    def _Q(self, dt: float) -> np.ndarray:
        """Discrete process-noise covariance (white-noise acceleration)."""
        q = self._Q_DIAG.copy()
        q[0] *= dt**2
        q[1] *= dt**2
        q[2] *= dt**2
        q[3] *= dt**2
        return np.diag(q)

    # --- measurement model ---------------------------------------------
    @staticmethod
    def _h(x: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        """
        Range measurement function.
        anchors : (M, 2) array of [ax, ay]
        returns : (M,) predicted distances
        """
        dx = anchors[:, 0] - x[0]
        dy = anchors[:, 1] - x[1]
        return np.sqrt(dx**2 + dy**2)

    def _R(self, m: int) -> np.ndarray:
        return np.eye(m) * (self._R_STD ** 2)

    # --- predict step --------------------------------------------------
    def _predict(self, x: np.ndarray, P: np.ndarray,
                 dt: float):
        n = self._n
        xi, W = self._cubature_points
        sqrt_P = np.linalg.cholesky(P)

        # Propagate cubature points
        sigma = (sqrt_P @ xi.T).T + x          # (2n, n)
        sigma_f = np.array([self._f(s, dt) for s in sigma])

        x_pred = W * 2 * n * sigma_f.mean(axis=0)
        diff   = sigma_f - x_pred
        P_pred = W * (diff.T @ diff) + self._Q(dt)

        return x_pred, P_pred

    # --- update step ---------------------------------------------------
    def _update(self, x: np.ndarray, P: np.ndarray,
                z: np.ndarray, anchors: np.ndarray):
        n  = self._n
        m  = len(z)
        xi, W = self._cubature_points
        sqrt_P = np.linalg.cholesky(P)

        # Propagate sigma points through h
        sigma   = (sqrt_P @ xi.T).T + x          # (2n, n)
        sigma_z = np.array([self._h(s, anchors) for s in sigma])  # (2n, m)

        z_pred = W * 2 * n * sigma_z.mean(axis=0)

        # Innovation covariance
        dz = sigma_z - z_pred
        dx = sigma   - x
        Pzz = W * (dz.T @ dz) + self._R(m)
        Pxz = W * (dx.T @ dz)

        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)

        x_upd = x + K @ (z - z_pred)
        P_upd = P - K @ Pzz @ K.T

        # Symmetrise to avoid numerical drift
        P_upd = 0.5 * (P_upd + P_upd.T)

        return x_upd, P_upd

    # ================================================================== #
    #  Helpers                                                             #
    # ================================================================== #

    @staticmethod
    def _parse_measurements(measurements, anchors):
        """
        Extract paired (anchor_position, distance) from whatever container
        the framework passes in.

        Supports:
          • dict  {anchor_id: distance}
          • list  [distance_0, distance_1, ...]  (same order as anchors)
          • list  of objects with .anchor_id / .distance attributes
        """
        if measurements is None or anchors is None:
            return None, None

        anchor_pos = []
        z_list     = []

        # Build a lookup id→position from anchors
        anchor_lookup = {}
        for a in anchors:
            aid = getattr(a, 'id', None)
            pos = getattr(a, 'position', None)
            if aid is not None and pos is not None:
                anchor_lookup[aid] = np.array([pos.x, pos.y], dtype=float)

        if isinstance(measurements, dict):
            for aid, dist in measurements.items():
                if aid in anchor_lookup and dist is not None:
                    anchor_pos.append(anchor_lookup[aid])
                    z_list.append(float(dist))

        elif isinstance(measurements, (list, tuple)):
            for i, item in enumerate(measurements):
                # Object with attributes
                dist = None
                aid  = None
                if hasattr(item, 'distance'):
                    dist = item.distance
                    aid  = getattr(item, 'anchor_id', i)
                else:
                    # Plain scalar – match by index
                    dist = float(item)
                    aid  = i

                apos = anchor_lookup.get(aid)
                if apos is None and isinstance(aid, int) and aid < len(anchors):
                    a = anchors[aid]
                    pos = getattr(a, 'position', None)
                    if pos:
                        apos = np.array([pos.x, pos.y], dtype=float)

                if apos is not None and dist is not None:
                    anchor_pos.append(apos)
                    z_list.append(float(dist))

        if not z_list:
            return None, None

        return np.array(anchor_pos), np.array(z_list)