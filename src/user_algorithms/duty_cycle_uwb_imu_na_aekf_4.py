import numpy as np
from src.core.localization.base_algorithm import AlgorithmInput, AlgorithmOutput
from src.user_algorithms.duty_cycle_uwb_imu_na_aekf import DutyCycleUwbImuFusionNaAekfAlgorithm
from src.user_algorithms.imuspeeddeadreckoningalgorithm import ImuspeeddeadreckoningalgorithmAlgorithm


class DutyCycleUwbImuFusionNaAekf4Algorithm(DutyCycleUwbImuFusionNaAekfAlgorithm):
    """
    Duty-Cycled NA-AEKF v4 — Step-count-based UWB scheduling.

    Instead of choosing a percentage or duration for the UWB/hybrid phase,
    you control EXACTLY:
      - UWB_STEPS_PER_BURST : how many UWB fusion steps (ticks) to run
                              each time UWB turns on (e.g. 1, 2, 5…)
      - IMU_ONLY_GAP        : how many seconds of IMU-only dead reckoning
                              between consecutive UWB bursts

    Scheduling pattern (after the convergence phase):
      ┌─────────────────┐     ┌─────────────────┐     ┌──────
      │ UWB burst       │     │ UWB burst       │     │ …
      │ N fusion steps  │     │ N fusion steps  │     │
      └────┬────────────┘     └────┬────────────┘     └──────
           │  IMU-only gap         │  IMU-only gap
           │  (T seconds)          │  (T seconds)
           └───────────────────────┘

    Also keeps v3 features:
      1. Initial position forced to (0, 0).
      2. First UWB_CONVERGENCE_DURATION seconds run UWB always on
         (no duty cycling) to let the filter converge.
    """

    # ================================================================== #
    #  Tunable Parameters                                                #
    # ================================================================== #

    # -- UWB convergence bootstrap ----------------------------------------
    UWB_CONVERGENCE_DURATION = 1.0  # seconds of pure UWB before duty cycling

    # -- Step-count-based UWB scheduling ----------------------------------
    UWB_STEPS_PER_BURST = 15    # number of UWB fusion ticks per burst
    IMU_ONLY_GAP        = 0.5  # seconds of IMU-only DR between UWB bursts

    # -- Transition tuning ------------------------------------------------
    DR_TO_FUSION_COV_INFLATE = 5.0

    # -- Noise parameters (initial values, adapted online) ----------------
    UWB_MEASUREMENT_NOISE = 0.19
    IMU_MEASUREMENT_NOISE = 1.0
    PROCESS_NOISE_JERK    = 1.0

    # -- Adaptive smoothing factors ---------------------------------------
    ALPHA = 0.3
    BETA  = 0.5

    # -- NLOS gating ------------------------------------------------------
    LAMBDA_NLOS = 5.0

    # -- IMU dead-reckoning parameters (IMU-Only Phase) -------------------
    DR_ZUPT_WINDOW      = 5
    DR_ZUPT_THRESHOLD   = 0.08
    DR_GYRO_THRESHOLD   = 0.05
    DR_MOVEMENT_SPEED   = 1.0

    # -- IMU parameters (Fusion Phase) ------------------------------------
    OMEGA      = 0.7
    BIAS_ALPHA = 0.05

    # -- ZUPT thresholds --------------------------------------------------
    DEFAULT_ZUPT_THRESHOLD = 0.08
    DEFAULT_GYRO_THRESHOLD = 0.1
    GT_STILLNESS_EPS       = 0.001

    # ------------------------------------------------------------------ #
    #  BaseLocalizationAlgorithm interface                                #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "Duty-Cycled UWB-IMU NA-AEKF v4"

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

        # ── Convergence phase tracking ──
        self._convergence_done: bool = False
        self._convergence_elapsed: float = 0.0

        # ── v4: Step-count-based scheduling state ──
        # Two phases: "uwb_burst" and "imu_gap"
        #   uwb_burst: count fusion steps done so far in this burst
        #   imu_gap:   accumulate elapsed time in seconds
        self._in_uwb_burst: bool = True     # start first cycle in burst mode
        self._burst_steps_done: int = 0     # fusion steps completed in current burst
        self._imu_gap_elapsed: float = 0.0  # time accumulated in current IMU gap

    # ------------------------------------------------------------------ #
    #  Override _initialise to force (0, 0) start position                #
    # ------------------------------------------------------------------ #

    def _initialise(self, tag):
        """Same as parent but forces the initial position to (0, 0)."""
        state, covariance, Q, R = super()._initialise(tag)
        state[0] = 0.0  # x
        state[1] = 0.0  # y
        return state, covariance, Q, R

    # ------------------------------------------------------------------ #
    #  v4: Step-count-based _uwb_active_this_tick                        #
    # ------------------------------------------------------------------ #

    def _uwb_active_this_tick(self, dt: float) -> bool:
        """
        Convergence phase → UWB always on.
        After convergence → step-count-based scheduling:
          - UWB burst: UWB on for exactly UWB_STEPS_PER_BURST ticks,
            then switch to IMU gap.
          - IMU gap: UWB off for IMU_ONLY_GAP seconds, then start
            a new UWB burst.
        """
        # ── Convergence phase ──
        if not self._convergence_done:
            self._convergence_elapsed += float(dt)
            if self._convergence_elapsed >= self.UWB_CONVERGENCE_DURATION:
                self._convergence_done = True
                # Start first duty cycle in UWB burst mode
                self._in_uwb_burst = True
                self._burst_steps_done = 0
                self._imu_gap_elapsed = 0.0
            return True  # UWB always on during convergence

        # ── UWB burst phase: count steps ──
        if self._in_uwb_burst:
            self._burst_steps_done += 1
            if self._burst_steps_done >= self.UWB_STEPS_PER_BURST:
                # Burst complete → switch to IMU gap next tick
                self._in_uwb_burst = False
                self._imu_gap_elapsed = 0.0
            return True  # UWB on this tick

        # ── IMU-only gap phase: count time ──
        self._imu_gap_elapsed += float(dt)
        if self._imu_gap_elapsed >= self.IMU_ONLY_GAP:
            # Gap complete → start a new UWB burst next tick
            self._in_uwb_burst = True
            self._burst_steps_done = 0
            # Don't return True yet — this tick is still IMU-only.
            # The burst starts on the NEXT tick.

        return False  # UWB off this tick

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

        # ── 2. During convergence: UWB-only, no DR transitions ──────────
        if not self._convergence_done:
            uwb_enabled = self._uwb_active_this_tick(dt)  # always True, advances clock

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

            output = self._parent_update_body(patched_input, uwb_enabled)

            # Tag the output so we can tell we're still converging
            if output.extra_data is not None:
                output.extra_data["mode"] = "uwb_convergence"
            else:
                output = AlgorithmOutput(
                    position=output.position,
                    state=output.state,
                    covariance=output.covariance,
                    initialized=output.initialized,
                    Q=output.Q,
                    R=output.R,
                    extra_data={"mode": "uwb_convergence"},
                )

            return output

        # ── 3. Convergence done → step-count-based duty cycling ─────────
        uwb_enabled = self._uwb_active_this_tick(dt)

        # Detect phase transitions
        fusion_to_imu = self._prev_uwb_enabled and not uwb_enabled
        imu_to_fusion = not self._prev_uwb_enabled and uwb_enabled
        self._prev_uwb_enabled = uwb_enabled

        # Transition: fusion → IMU-only
        if fusion_to_imu:
            self._save_fusion_state(state, covariance, Q, R)
            self._reinit_dead_reckoning(state, tag)

        # Transition: IMU-only → fusion
        if imu_to_fusion:
            state, covariance, Q, R = self._restore_fusion_state()

        # Run the appropriate algorithm for this tick
        if uwb_enabled:
            return self._run_fusion_tick(input_data, state, covariance, Q, R,
                                         initialized, uwb_enabled)
        else:
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

        self._saved_yaw       = float(full_state[6])
        self._saved_gyro_bias = float(full_state[7])

    def _reinit_dead_reckoning(self, state, tag) -> None:
        """Reinitialise the DR algorithm and seed it from the fusion position."""
        self._dr.initialize()

        full = np.asarray(state, dtype=float).ravel()
        self._dr_state = np.array([full[0], full[1], 0.0, 0.0], dtype=float)
        self._dr_cov   = np.eye(4) * 0.1
        self._dr_Q     = np.eye(4) * 1e-3
        self._dr_R     = np.eye(2) * 1e-3
        self._dr_initialized = False

        self._dr._yaw = self._saved_yaw
        self._dr._gyro_bias = self._saved_gyro_bias
        self._dr._initialized = True
        self._dr_initialized = True

    def _restore_fusion_state(self):
        """
        Restore saved fusion parameters and update position from DR output.
        """
        state = self._saved_fusion_state.copy()
        cov   = self._saved_fusion_cov.copy()
        Q     = self._saved_fusion_Q.copy() if self._saved_fusion_Q is not None else None
        R     = self._saved_fusion_R.copy() if self._saved_fusion_R is not None else None

        if self._dr_state is not None:
            state[0] = self._dr_state[0]
            state[1] = self._dr_state[1]
            state[2] = self._dr_state[2]
            state[3] = self._dr_state[3]

        if len(state) >= self.n_full:
            state[6] = self._dr._yaw
            state[7] = self._dr._gyro_bias

        cov[0, 0] *= self.DR_TO_FUSION_COV_INFLATE
        cov[1, 1] *= self.DR_TO_FUSION_COV_INFLATE
        cov[2, 2] *= self.DR_TO_FUSION_COV_INFLATE
        cov[3, 3] *= self.DR_TO_FUSION_COV_INFLATE

        return state, cov, Q, R

    # ------------------------------------------------------------------ #
    #  Per-tick runners                                                   #
    # ------------------------------------------------------------------ #

    def _run_fusion_tick(self, input_data, state, covariance, Q, R,
                         initialized, uwb_enabled) -> AlgorithmOutput:
        """
        Full NA-AEKF predict + correct (delegates to parent class logic).
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

        return self._parent_update_body(patched_input, uwb_enabled)

    def _run_dr_tick(self, input_data, state, covariance, Q, R,
                     initialized, uwb_enabled) -> AlgorithmOutput:
        """
        IMU-only tick: delegate to the dead-reckoning algorithm.
        """
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

        self._dr_state       = dr_out.state
        self._dr_cov         = dr_out.covariance
        self._dr_Q           = dr_out.Q
        self._dr_R           = dr_out.R
        self._dr_initialized = dr_out.initialized

        full_state = self._saved_fusion_state.copy()
        full_state[0] = float(dr_out.state[0])
        full_state[1] = float(dr_out.state[1])
        full_state[2] = float(dr_out.state[2])
        full_state[3] = float(dr_out.state[3])
        full_state[6] = float(self._dr._yaw)
        full_state[7] = float(self._dr._gyro_bias)

        return AlgorithmOutput(
            position=dr_out.position,
            state=full_state,
            covariance=self._saved_fusion_cov,
            initialized=initialized,
            Q=self._saved_fusion_Q,
            R=self._saved_fusion_R,
            extra_data={
                "zupt_triggered": dr_out.extra_data.get("zupt_triggered", False)
                    if dr_out.extra_data else False,
                "yaw": float(self._dr._yaw),
                "gyro_bias": float(self._dr._gyro_bias),
                "uwb_window_open": uwb_enabled,
                "uwb_steps_per_burst": self.UWB_STEPS_PER_BURST,
                "imu_only_gap": self.IMU_ONLY_GAP,
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

        if not initialized or state is None or covariance is None:
            state, covariance, Q, R = self._initialise(tag)
            initialized = True

        state, yaw, gyro_bias = self._unpack_state(state)

        is_stationary = self._check_stationary(tag, imu_on, accel_raw, gyro_raw)

        Q = self._build_process_noise(Q, dt)
        state_pred, yaw, gyro_bias = self._predict(
            state, dt, tag, imu_on, gyro_raw, yaw, gyro_bias, is_stationary
        )
        P_pred = self._predict_covariance(covariance, dt, Q)

        state, P, Q, R = self._update(
            state_pred, P_pred, measurements, anchors,
            imu_on, accel_raw, Q, R, is_los, uwb_enabled,
        )

        if is_stationary:
            state, P = self._apply_zupt(state, P)
        else:
            vx, vy = float(state[2]), float(state[3])
            speed = np.hypot(vx, vy)
            if speed > 0.3:
                vel_yaw = float(np.arctan2(vy, vx))
                alpha = 0.1
                new_x = (1.0 - alpha) * np.cos(yaw) + alpha * np.cos(vel_yaw)
                new_y = (1.0 - alpha) * np.sin(yaw) + alpha * np.sin(vel_yaw)
                yaw = float(np.arctan2(new_y, new_x))

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
                "uwb_steps_per_burst": self.UWB_STEPS_PER_BURST,
                "imu_only_gap": self.IMU_ONLY_GAP,
                "burst_steps_done": self._burst_steps_done,
                "mode": "fusion",
            },
        )
