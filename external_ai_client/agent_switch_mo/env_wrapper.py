import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any
from pulse_ai_client import PulseRLEnv, PulseState
from .config import (
    W_ERROR,
    W_STD,
    W_ENERGY,
    ERROR_TARGET_MEAN,
    ERROR_TARGET_STD,
    DEFAULT_IMU_POWER_MW,
    DEFAULT_FUSION_POWER_MW,
)

class MacroSwitchWrapper(gym.Wrapper):
    """
    Wraps PulseRLEnv into a Time-Based Macro-Step environment for the
    sensor switching task.

    Time-Based Architecture:
      - Each macro-step spans TAU_SECONDS of simulated time.
      - IMU updates at IMU_FREQ_HZ (e.g. 100 Hz → sub-step every 10 ms).
      - UWB updates at UWB_FREQ_HZ (e.g.  10 Hz → every 10th sub-step).
      - The agent controls the fraction of τ spent in IMU-only mode
        via incremental actions.

    Incremental Action Space (Markov Modulated):
      - 0: Decrease IMU duty-cycle ratio by one discrete step
      - 1: Keep same
      - 2: Increase IMU duty-cycle ratio by one discrete step

    Augmented Observation Space:
      - Appends current IMU duty-cycle ratio (∈ [0, 1]) to the base
        state vector.
    """

    def __init__(
        self,
        env: PulseRLEnv,
        tau_seconds: float = 1.0,
        imu_freq_hz: float = 100.0,
        uwb_freq_hz: float = 10.0,
    ):
        super().__init__(env)
        self.tau_seconds = tau_seconds
        self.imu_freq_hz = imu_freq_hz
        self.uwb_freq_hz = uwb_freq_hz
        self.num_agents = env.num_agents
        self.actual_num_anchors = 8  # default, updated after reset

        # ── Derived timing constants (initialized with defaults, updated from API) ────
        self.imu_period = 1.0 / imu_freq_hz       # seconds per IMU tick
        self.uwb_period = 1.0 / uwb_freq_hz       # seconds per UWB tick
        # Number of IMU-rate sub-steps per macro-step (will be dynamic)
        self.sub_steps_per_macro = max(1, int(round(self.tau_seconds * imu_freq_hz)))
        # UWB fires every N-th IMU sub-step (integer ratio for stability)
        self.uwb_every_n = max(1, int(round(imu_freq_hz / uwb_freq_hz)))

        # Action Space: Tuple of (T_imu, T_fusion) choices
        self.t_imu_choices = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0])
        self.t_fusion_choices = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
        self.num_imu_choices = len(self.t_imu_choices)
        self.num_fusion_choices = len(self.t_fusion_choices)
        self.total_actions = self.num_imu_choices * self.num_fusion_choices

        # MultiDiscrete for each agent mapped to a flattened discrete space for DQN
        self.action_space = spaces.MultiDiscrete([self.total_actions] * self.num_agents)

        # Observation Space: base + 6 augmented fields + 6 IMU fields
        # (t_imu_choice, t_fusion_choice, t_imu_state, t_uwb_state, cycle_time, uwb_window_open) + (ax, ay, az, wx, wy, wz)
        orig_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(orig_shape[0] + 12,), dtype=np.float32
        )

        # Per-agent timing selections
        self.current_t_imu = np.ones(self.num_agents, dtype=np.float32) * 0.5
        self.current_t_fusion = np.ones(self.num_agents, dtype=np.float32) * 0.5

        # Track previous macro-step mean error per agent for the Δe term
        self.prev_mean_errors = np.zeros(self.num_agents, dtype=np.float32)
        
        # Track previous cycle total energy for discrete comparisons
        self.prev_energies = np.zeros(self.num_agents, dtype=np.float32)

        # ── Dynamic energy bounds (µJ) ────────────────────────────────
        # Computed from power defaults initially, but will dynamically adapt
        # using the empirical step energies received from the server.
        self.E_MIN = DEFAULT_IMU_POWER_MW * tau_seconds * 1000.0       # µJ
        self.E_MAX = DEFAULT_FUSION_POWER_MW * tau_seconds * 1000.0    # µJ
        
        self._empirical_imu_step_energy = 0.0
        self._empirical_both_step_energy = 0.0

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_macro_reward(
        self,
        all_micro_states: List[List[PulseState]],
    ) -> np.ndarray:
        """Compute reward over one τ-second macro-step for each agent.
        
        Continuous Multi-Objective Reward:
        R = W_ERROR * r_error_mean + W_STD * r_error_std + W_ENERGY * r_energy
        
        Where:
          - r_error_mean:
             1.0 if e_mean <= ERROR_TARGET_MEAN
             1.0 - (e_mean - ERROR_TARGET_MEAN) / ERROR_TARGET_MEAN if e_mean > ERROR_TARGET_MEAN (clamped to [-1.0, 1.0])
          - r_error_std:
             1.0 if e_std <= ERROR_TARGET_STD
             1.0 - (e_std - ERROR_TARGET_STD) / ERROR_TARGET_STD if e_std > ERROR_TARGET_STD (clamped to [-1.0, 1.0])
          - r_energy:
             +1 if E_t < E_{t-1} (energy decreased / better)
             -1 if E_t > E_{t-1} (energy increased / worse)
              0 otherwise
        """
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        N = len(all_micro_states)
        if N == 0:
            return rewards

        for a_idx in range(self.num_agents):
            errors = []
            total_energy_uJ = 0.0

            for micro_states in all_micro_states:
                if a_idx >= len(micro_states):
                    continue
                state = micro_states[a_idx]
                e_step = state.energy.step_energy_uJ
                src = state.environment.measurement_source if hasattr(state.environment, 'measurement_source') else ""
                
                # Empirically learn the step energies to dynamically scale bounds (retaining original environment logic)
                if src == "IMU" and e_step > 0:
                    if self._empirical_imu_step_energy == 0:
                        self._empirical_imu_step_energy = e_step
                    else:
                        self._empirical_imu_step_energy = 0.9 * self._empirical_imu_step_energy + 0.1 * e_step
                elif src == "Both" and e_step > 0:
                    if self._empirical_both_step_energy == 0:
                        self._empirical_both_step_energy = e_step
                    else:
                        self._empirical_both_step_energy = 0.9 * self._empirical_both_step_energy + 0.1 * e_step
                
                err = state.precision.localization_error
                errors.append(err)
                total_energy_uJ += e_step

            if not errors:
                continue
                
            # Dynamically recalculate E_MIN and E_MAX if we have reliable empirical data
            if self._empirical_imu_step_energy > 0 and self._empirical_both_step_energy > 0:
                num_uwb = self.sub_steps_per_macro // self.uwb_every_n
                num_imu = self.sub_steps_per_macro - num_uwb
                self.E_MAX = (num_uwb * self._empirical_both_step_energy) + (num_imu * self._empirical_imu_step_energy)
                self.E_MIN = self.sub_steps_per_macro * self._empirical_imu_step_energy
            else:
                self.E_MIN = DEFAULT_IMU_POWER_MW * self.tau_seconds * 1000.0
                self.E_MAX = DEFAULT_FUSION_POWER_MW * self.tau_seconds * 1000.0

            # ── 1. Average Error Continuous Reward ────────────────────────────
            e_mean = np.mean(errors)
            if e_mean <= ERROR_TARGET_MEAN:
                r_error_mean = 1.0
            else:
                r_error_mean = 1.0 - (e_mean - ERROR_TARGET_MEAN) / ERROR_TARGET_MEAN
                r_error_mean = max(-1.0, float(r_error_mean))

            # ── 2. Standard Deviation Continuous Reward ───────────────────────
            e_std = np.std(errors)
            if e_std <= ERROR_TARGET_STD:
                r_error_std = 1.0
            else:
                r_error_std = 1.0 - (e_std - ERROR_TARGET_STD) / ERROR_TARGET_STD
                r_error_std = max(-1.0, float(r_error_std))

            # ── 3. Energy Comparison Discrete Reward (t vs t-1) ───────────────
            E_prev = self.prev_energies[a_idx]
            if E_prev == 0.0:
                r_energy = 0.0
            elif total_energy_uJ < E_prev:
                r_energy = 1.0
            elif total_energy_uJ > E_prev:
                r_energy = -1.0
            else:
                r_energy = 0.0

            # Update historical energy
            self.prev_energies[a_idx] = total_energy_uJ

            # Combined reward formula
            r_mo = W_ERROR * r_error_mean + W_STD * r_error_std + W_ENERGY * r_energy

            rewards[a_idx] = r_mo
            self.prev_mean_errors[a_idx] = e_mean

        return rewards

    # ── Step (time-based) ─────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one macro-step (dynamic duration) of time-based simulation.

        The agent directly selects T_imu and T_fusion for the EKF.
        The environment steps forward by max(T_cycle) = max(T_imu + T_fusion) across all agents,
        ensuring each agent completes at least one full cycle before the next decision.
        """
        # Decode actions to (T_imu, T_fusion)
        for a_idx in range(self.num_agents):
            act = int(action[a_idx])
            imu_idx = act // self.num_fusion_choices
            fusion_idx = act % self.num_fusion_choices
            
            self.current_t_imu[a_idx] = self.t_imu_choices[imu_idx]
            self.current_t_fusion[a_idx] = self.t_fusion_choices[fusion_idx]

        # Dynamically determine the macro step duration based on the max cycle time
        max_cycle_time = np.max(self.current_t_imu + self.current_t_fusion)
        self.tau_seconds = float(max_cycle_time)
        self.sub_steps_per_macro = max(1, int(round(self.tau_seconds * self.imu_freq_hz)))

        final_obs = None
        final_info = None
        terminated_all = np.zeros(self.num_agents, dtype=bool)
        truncated_all = np.zeros(self.num_agents, dtype=bool)

        all_micro_states: List[List[PulseState]] = []

        for k in range(self.sub_steps_per_macro):
            backend_action = []
            for a_idx in range(self.num_agents):
                # The duty-cycled EKF handles IMU/UWB gating internally based on t_imu and t_uwb.
                # We always send measurement_source='Both' so the server
                # provides both IMU and UWB data; the EKF's cycle timer
                # decides when to fuse UWB measurements.
                backend_action.append({
                    "filter": "Duty-Cycled IMU-UWB Adaptive EKF",
                    "measurement_source": "Both",
                    "anchors": list(range(self.actual_num_anchors)),
                    "t_imu": float(self.current_t_imu[a_idx]),
                    "t_uwb": float(self.current_t_fusion[a_idx]),
                })

            obs, _micro_reward, terminated, truncated, info = self.env.step(
                backend_action
            )
            final_obs = obs
            final_info = info

            micro_states = info.get("states", [])
            all_micro_states.append(micro_states)

            terminated_all = np.logical_or(terminated_all, terminated)
            truncated_all = np.logical_or(truncated_all, truncated)

            if terminated_all.all() or truncated_all.all():
                break

        macro_reward = self._compute_macro_reward(all_micro_states)
        final_info["all_micro_states"] = all_micro_states

        # Augment observation: append chosen t_imu, t_fusion, duty-cycle state, and explicit IMU data
        duty_cycle_state = np.zeros((self.num_agents, 4), dtype=np.float32)
        imu_state = np.zeros((self.num_agents, 6), dtype=np.float32)
        final_micro_states = all_micro_states[-1] if all_micro_states else []
        for a_idx in range(self.num_agents):
            if a_idx < len(final_micro_states):
                ms = final_micro_states[a_idx]
                dc = getattr(ms, 'duty_cycle', None)
                if dc is not None:
                    duty_cycle_state[a_idx, 0] = getattr(dc, 't_imu', 0.0)
                    duty_cycle_state[a_idx, 1] = getattr(dc, 't_uwb', 0.0)
                    duty_cycle_state[a_idx, 2] = getattr(dc, 'cycle_time', 0.0)
                    duty_cycle_state[a_idx, 3] = 1.0 if getattr(dc, 'uwb_window_open', False) else 0.0
                
                imu = getattr(ms, 'imu_data', None)
                if imu is not None:
                    imu_state[a_idx, 0] = imu.acceleration[0]
                    imu_state[a_idx, 1] = imu.acceleration[1]
                    imu_state[a_idx, 2] = imu.acceleration[2]
                    imu_state[a_idx, 3] = imu.angular_velocity[0]
                    imu_state[a_idx, 4] = imu.angular_velocity[1]
                    imu_state[a_idx, 5] = imu.angular_velocity[2]

        # Update timing constants dynamically from the latest received states
        self._update_timing_from_states(final_micro_states)

        augmented_obs = np.hstack([
            final_obs,
            self.current_t_imu[:, np.newaxis],
            self.current_t_fusion[:, np.newaxis],
            duty_cycle_state,
            imu_state,
        ])

        return augmented_obs, macro_reward, terminated_all, truncated_all, final_info

    def _update_timing_from_states(self, states: List[PulseState]) -> None:
        """Update IMU/UWB period and frequency constants dynamically from the simulator state."""
        if not states or len(states) == 0:
            return
        state = states[0]
        env_cfg = getattr(state, "environment", None)
        if env_cfg is not None:
            imu_period = getattr(env_cfg, "imu_period", getattr(env_cfg, "dt", None))
            if imu_period is not None and imu_period > 0:
                self.imu_period = float(imu_period)
                self.imu_freq_hz = 1.0 / self.imu_period

            uwb_period = getattr(env_cfg, "uwb_period", None)
            if uwb_period is not None and uwb_period > 0:
                self.uwb_period = float(uwb_period)
                self.uwb_freq_hz = 1.0 / self.uwb_period

            # Re-derive ratios
            self.uwb_every_n = max(1, int(round(self.imu_freq_hz / self.uwb_freq_hz)))

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        """Reset previous errors and duty-cycle parameters on episode boundary."""
        self.prev_mean_errors = np.zeros(self.num_agents, dtype=np.float32)
        self.prev_energies = np.zeros(self.num_agents, dtype=np.float32)
        self.current_t_imu = np.ones(self.num_agents, dtype=np.float32) * 0.5
        self.current_t_fusion = np.ones(self.num_agents, dtype=np.float32) * 0.5
        obs, info = super().reset(**kwargs)

        # Update actual anchor count from server state
        initial_states = info.get("states", [])
        if (
            initial_states
            and hasattr(initial_states[0], "num_anchors")
            and initial_states[0].num_anchors > 0
        ):
            self.actual_num_anchors = initial_states[0].num_anchors

        # Update timing constants from the API state
        self._update_timing_from_states(initial_states)

        # Augment observation with t_imu, t_fusion, server state, and explicit IMU state (zeros on reset)
        augmented_obs = np.hstack([
            obs,
            self.current_t_imu[:, np.newaxis],
            self.current_t_fusion[:, np.newaxis],
            np.zeros((self.num_agents, 4), dtype=np.float32),  # t_imu, t_uwb, cycle_time, uwb_window_open
            np.zeros((self.num_agents, 6), dtype=np.float32),  # ax, ay, az, wx, wy, wz
        ])
        return augmented_obs, info
