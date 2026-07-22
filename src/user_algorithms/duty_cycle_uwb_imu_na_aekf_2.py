import numpy as np
from src.core.localization.base_algorithm import AlgorithmInput, AlgorithmOutput
from src.user_algorithms.duty_cycle_uwb_imu_na_aekf import DutyCycleUwbImuFusionNaAekfAlgorithm
from src.user_algorithms.imuspeeddeadreckoningalgorithm import ImuspeeddeadreckoningalgorithmAlgorithm


class DutyCycleUwbImuFusionNaAekf2Algorithm(DutyCycleUwbImuFusionNaAekfAlgorithm):
    """
    Duty-Cycled NA-AEKF with IMU Speed Dead Reckoning during IMU-only phases.

    Extends the original duty-cycle algorithm:
      - IMU-only phase  → delegates to IMU Speed Dead Reckoning (simple
        heading + speed propagation) instead of the full EKF predict+correct.
      - Fusion phase    → runs the standard NA-AEKF predict + UWB+IMU
        correction, exactly as the parent class.

    Phase transition behaviour:
      - Fusion → IMU-only: saves the full EKF state / covariance / Q / R,
        then reinitialises the dead-reckoning algorithm and seeds it with the
        latest fusion position.
      - IMU-only → Fusion: copies the dead-reckoning position (and velocity)
        into the saved EKF state, inflates the position covariance to let UWB
        corrections dominate on re-entry, then resumes the EKF from there.
    """

    # ================================================================== #
    #  Tunable Parameters                                                #
    # ================================================================== #

    # -- Duty-cycled UWB scheduling ---------------------------------------
    IMU_ONLY_DURATION = 3.6   # seconds (duration of the dead-reckoning phase)
    HYBRID_DURATION   = 0.4   # seconds (duration of the fusion phase)
    DUTY_CYCLE_PERIOD = IMU_ONLY_DURATION + HYBRID_DURATION

    # -- Transition tuning ------------------------------------------------
    # Covariance inflation factor applied to position states (x, y)
    # when switching from dead reckoning back to fusion, reflecting
    # the accumulated DR uncertainty.
    DR_TO_FUSION_COV_INFLATE = 5.0

    # -- Noise parameters (initial values, adapted online) ----------------
    UWB_MEASUREMENT_NOISE = 0.19       # initial R diagonal for UWB ranges
    IMU_MEASUREMENT_NOISE = 1.0        # initial R diagonal for IMU accelerometer
    PROCESS_NOISE_JERK    = 1.0        # initial Q diagonal (jerk variance)

    # -- Adaptive smoothing factors ---------------------------------------
    ALPHA = 0.3          # smoothing factor for R adaptation
    BETA  = 0.5          # smoothing factor for Q adaptation

    # -- NLOS gating ------------------------------------------------------
    LAMBDA_NLOS = 10.0    # inflation factor for UWB measurement variance when NLOS

    # -- IMU dead-reckoning parameters (IMU-Only Phase) -------------------
    DR_ZUPT_WINDOW      = 5         # samples in the sliding window
    DR_ZUPT_THRESHOLD   = 0.08      # m²/s⁴ – accel-norm variance gate
    DR_GYRO_THRESHOLD   = 0.05      # rad/s – gyro norm stillness gate
    DR_MOVEMENT_SPEED   = 1.0       # m/s – assumed constant speed when walking

    # -- IMU parameters (Fusion Phase) ------------------------------------
    OMEGA      = 0.7       # IMU-velocity blend weight in prediction
    BIAS_ALPHA = 0.05      # gyro bias EMA smoothing factor

    # -- ZUPT thresholds --------------------------------------------------
    DEFAULT_ZUPT_THRESHOLD = 0.08   # m/s² – accel-norm stillness gate
    DEFAULT_GYRO_THRESHOLD = 0.1    # rad/s – gyro-norm stillness gate
    GT_STILLNESS_EPS       = 0.001  # m²/s² – ground-truth speed² gate

    # ------------------------------------------------------------------ #
    #  BaseLocalizationAlgorithm interface                                #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "Duty-Cycled UWB-IMU NA-AEKF v2"

    def initialize(self) -> None:
        super().initialize()

        # Dead-reckoning sub-algorithm (composed, not inherited)
        self._dr = ImuspeeddeadreckoningalgorithmAlgorithm()
        self._dr.initialize()

        # Phase-transition tracking
        self._prev_uwb_enabled: bool = True   # assume we start in fusion

        # Saved fusion-filter state (frozen while DR is running)
        self._saved_fusion_state: np.ndarray | None = None
        self._saved_fusion_cov:   np.ndarray | None = None
        self._saved_fusion_Q:     np.ndarray | None = None
        self._saved_fusion_R:     np.ndarray | None = None
        self._saved_yaw:   float = 0.0
        self._saved_gyro_bias: float = 0.0

        # Dead-reckoning state carried across IMU-only ticks
        self._dr_state:       np.ndarray | None = None
        self._dr_cov:         np.ndarray | None = None
        self._dr_Q:           np.ndarray | None = None
        self._dr_R:           np.ndarray | None = None
        self._dr_initialized: bool = False

    # ------------------------------------------------------------------ #
    #  Main update                                                        #
    # ------------------------------------------------------------------ #

    def update(self, input_data: AlgorithmInput) -> AlgorithmOutput:
        dt  = input_data.dt
        tag = input_data.tag

        # ── 1. Ensure the EKF is initialised (first-ever tick) ──────────
        state       = input_data.state
        covariance  = input_data.covariance
        Q           = input_data.Q
        R           = input_data.R
        initialized = input_data.initialized

        if not initialized or state is None or covariance is None:
            state, covariance, Q, R = self._initialise(tag)
            initialized = True

        # ── 2. Determine whether UWB is on this tick ────────────────────
        uwb_enabled = self._uwb_active_this_tick(dt)

        # ── 3. Detect phase transitions ─────────────────────────────────
        fusion_to_imu = self._prev_uwb_enabled and not uwb_enabled
        imu_to_fusion = not self._prev_uwb_enabled and uwb_enabled
        self._prev_uwb_enabled = uwb_enabled

        # ── 4A. Transition: fusion → IMU-only ───────────────────────────
        if fusion_to_imu:
            self._save_fusion_state(state, covariance, Q, R)
            self._reinit_dead_reckoning(state, tag)

        # ── 4B. Transition: IMU-only → fusion ───────────────────────────
        if imu_to_fusion:
            state, covariance, Q, R = self._restore_fusion_state()

        # ── 5. Run the appropriate algorithm for this tick ──────────────
        if uwb_enabled:
            # --- FUSION TICK (full NA-AEKF) ---
            return self._run_fusion_tick(input_data, state, covariance, Q, R,
                                         initialized, uwb_enabled)
        else:
            # --- IMU-ONLY TICK (dead reckoning) ---
            return self._run_dr_tick(input_data, state, covariance, Q, R,
                                     initialized, uwb_enabled)

    # ------------------------------------------------------------------ #
    #  Phase-transition helpers                                           #
    # ------------------------------------------------------------------ #

    def _save_fusion_state(self, state, covariance, Q, R) -> None:
        """Freeze the full EKF state before entering the DR phase."""
        full_state = np.asarray(state, dtype=float).ravel()
        if len(full_state) < self.n_full:
            pad = np.zeros(self.n_full - len(full_state))
            if len(full_state) == self.n_ekf and hasattr(self, "_init_yaw"):
                pad[0] = self._init_yaw
            full_state = np.concatenate([full_state, pad])
            
        self._saved_fusion_state = full_state.copy()
        self._saved_fusion_cov   = covariance.copy()
        self._saved_fusion_Q     = Q.copy() if Q is not None else None
        self._saved_fusion_R     = R.copy() if R is not None else None

        # Also save yaw / gyro_bias from the packed state
        self._saved_yaw       = float(full_state[6])
        self._saved_gyro_bias = float(full_state[7])

    def _reinit_dead_reckoning(self, state, tag) -> None:
        """Reinitialise the DR algorithm and seed it from the fusion position."""
        self._dr.initialize()

        # Build a 4-vector [x, y, 0, 0] seeded from the fusion state
        full = np.asarray(state, dtype=float).ravel()
        self._dr_state = np.array([full[0], full[1], 0.0, 0.0], dtype=float)
        self._dr_cov   = np.eye(4) * 0.1
        self._dr_Q     = np.eye(4) * 1e-3
        self._dr_R     = np.eye(2) * 1e-3
        self._dr_initialized = False  # let DR's own init logic run once

        # Seed the DR's heading from the saved yaw so heading is continuous
        self._dr._yaw = self._saved_yaw
        self._dr._gyro_bias = self._saved_gyro_bias
        self._dr._initialized = True
        self._dr_initialized = True

    def _restore_fusion_state(self):
        """
        Restore saved fusion parameters and update position from DR output.

        Copies DR position (and velocity) into the EKF state so the filter
        picks up from where dead reckoning left off. Inflates position
        covariance to let UWB corrections dominate on re-entry.
        """
        state = self._saved_fusion_state.copy()
        cov   = self._saved_fusion_cov.copy()
        Q     = self._saved_fusion_Q.copy() if self._saved_fusion_Q is not None else None
        R     = self._saved_fusion_R.copy() if self._saved_fusion_R is not None else None

        # Transfer DR position → EKF state
        if self._dr_state is not None:
            state[0] = self._dr_state[0]   # x
            state[1] = self._dr_state[1]   # y
            state[2] = self._dr_state[2]   # vx
            state[3] = self._dr_state[3]   # vy

        # Transfer DR heading → EKF yaw slot
        if len(state) >= self.n_full:
            state[6] = self._dr._yaw
            state[7] = self._dr._gyro_bias

        # Inflate position covariance to reflect DR drift
        cov[0, 0] *= self.DR_TO_FUSION_COV_INFLATE
        cov[1, 1] *= self.DR_TO_FUSION_COV_INFLATE

        return state, cov, Q, R

    # ------------------------------------------------------------------ #
    #  Per-tick runners                                                   #
    # ------------------------------------------------------------------ #

    def _run_fusion_tick(self, input_data, state, covariance, Q, R,
                         initialized, uwb_enabled) -> AlgorithmOutput:
        """
        Full NA-AEKF predict + correct (delegates to parent class logic).

        We rebuild the AlgorithmInput with the (possibly restored) state so
        the parent's update() sees the right values.
        """
        patched_input = AlgorithmInput(
            measurements=input_data.measurements,
            anchors=input_data.anchors,
            tag=input_data.tag,
            dt=input_data.dt,
            state=state,
            covariance=covariance,
            Q=Q,
            R=R,
            initialized=initialized,
            imu_data_on=input_data.imu_data_on,
            accel=input_data.accel,
            gyro=input_data.gyro,
            is_los=input_data.is_los,
            params=input_data.params,
        )

        # Call the parent's update but we need to avoid the parent calling
        # _uwb_active_this_tick again (it would double-advance the clock).
        # So we replicate the parent's update body inline, reusing its helpers.
        return self._parent_update_body(patched_input, uwb_enabled)

    def _run_dr_tick(self, input_data, state, covariance, Q, R,
                     initialized, uwb_enabled) -> AlgorithmOutput:
        """
        IMU-only tick: delegate to the dead-reckoning algorithm.

        Returns an AlgorithmOutput compatible with the EKF state layout
        (8-vector) so the simulator's state management stays consistent.
        """
        # Build an AlgorithmInput for the DR algorithm
        dr_params = (input_data.params or {}).copy()
        dr_params.setdefault("zupt_window", self.DR_ZUPT_WINDOW)
        dr_params.setdefault("zupt_threshold", self.DR_ZUPT_THRESHOLD)
        dr_params.setdefault("gyro_threshold", self.DR_GYRO_THRESHOLD)
        dr_params.setdefault("movement_speed", self.DR_MOVEMENT_SPEED)

        dr_input = AlgorithmInput(
            measurements=[],
            anchors=[],
            tag=input_data.tag,
            dt=input_data.dt,
            state=self._dr_state,
            covariance=self._dr_cov,
            Q=self._dr_Q,
            R=self._dr_R,
            initialized=self._dr_initialized,
            imu_data_on=input_data.imu_data_on,
            accel=input_data.accel,
            gyro=input_data.gyro,
            params=dr_params,
        )

        dr_out = self._dr.update(dr_input)

        # Persist DR state for next tick / transition
        self._dr_state       = dr_out.state
        self._dr_cov         = dr_out.covariance
        self._dr_Q           = dr_out.Q
        self._dr_R           = dr_out.R
        self._dr_initialized = dr_out.initialized

        # Build the full 8-vector from the frozen EKF state + DR position
        full_state = self._saved_fusion_state.copy()
        full_state[0] = float(dr_out.state[0])    # x from DR
        full_state[1] = float(dr_out.state[1])    # y from DR
        full_state[2] = float(dr_out.state[2])    # vx from DR
        full_state[3] = float(dr_out.state[3])    # vy from DR
        full_state[6] = float(self._dr._yaw)      # yaw from DR
        full_state[7] = float(self._dr._gyro_bias) # gyro bias from DR

        # Return with EKF-compatible shape but DR-driven position
        return AlgorithmOutput(
            position=dr_out.position,
            state=full_state,
            covariance=self._saved_fusion_cov,  # frozen EKF covariance
            initialized=initialized,
            Q=self._saved_fusion_Q,
            R=self._saved_fusion_R,
            extra_data={
                "zupt_triggered": dr_out.extra_data.get("zupt_triggered", False)
                    if dr_out.extra_data else False,
                "yaw": float(self._dr._yaw),
                "gyro_bias": float(self._dr._gyro_bias),
                "uwb_window_open": uwb_enabled,
                "t_imu": float(self._cycle_length),
                "t_uwb": float(self._active_window),
                "mode": "dead_reckoning",
            },
        )

    # ------------------------------------------------------------------ #
    #  Parent update body (without double-advancing the duty-cycle clock) #
    # ------------------------------------------------------------------ #

    def _parent_update_body(self, input_data: AlgorithmInput,
                            uwb_enabled: bool) -> AlgorithmOutput:
        """
        Replicates the parent's update() logic but skips the
        _uwb_active_this_tick() call (already done in our update()).
        """
        measurements = input_data.measurements
        anchors      = input_data.anchors
        dt           = input_data.dt
        imu_on       = input_data.imu_data_on
        accel_raw    = input_data.accel
        gyro_raw     = input_data.gyro
        tag          = input_data.tag
        is_los       = input_data.is_los

        state       = input_data.state
        covariance  = input_data.covariance
        Q           = input_data.Q
        R           = input_data.R
        initialized = input_data.initialized

        # Initialisation guard (should already be done, but keep for safety)
        if not initialized or state is None or covariance is None:
            state, covariance, Q, R = self._initialise(tag)
            initialized = True

        state, yaw, gyro_bias = self._unpack_state(state)

        # Stationarity check
        is_stationary = self._check_stationary(tag, imu_on, accel_raw, gyro_raw)

        # Prediction
        Q = self._build_process_noise(Q, dt)
        state_pred, yaw, gyro_bias = self._predict(
            state, dt, tag, imu_on, gyro_raw, yaw, gyro_bias, is_stationary
        )
        P_pred = self._predict_covariance(covariance, dt, Q)

        # NLOS-aware adaptive UWB + IMU measurement update
        state, P, Q, R = self._update(
            state_pred, P_pred, measurements, anchors,
            imu_on, accel_raw, Q, R, is_los, uwb_enabled,
        )

        # ZUPT
        if is_stationary:
            state, P = self._apply_zupt(state, P)

        # Covariance repair
        P = self._repair_covariance(P)

        full_state = self._pack_state(state, yaw, gyro_bias)

        return AlgorithmOutput(
            position=(float(state[0]), float(state[1])),
            state=full_state,
            covariance=P,
            initialized=initialized,
            Q=Q,
            R=R,
            extra_data={
                "zupt_triggered": is_stationary,
                "yaw": float(yaw),
                "gyro_bias": float(gyro_bias),
                "uwb_window_open": uwb_enabled,
                "t_imu": float(self._cycle_length),
                "t_uwb": float(self._active_window),
                "mode": "fusion",
            },
        )
