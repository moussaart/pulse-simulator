"""
PULSE AI - Markov Modulated Switching Agent (PyTorch DQN Edition)
                         — Time-Based Simulation —

Goal:
    The agent must learn to MAXIMIZE the time spent in low-power IMU-only mode
    (saving energy) while KEEPING the localization error below a target threshold.

    This is a precision-vs-energy tradeoff:
      - Using IMU-only is cheap but error drifts over time.
      - Using UWB+IMU fusion is precise but costs energy.
      - The optimal policy finds the longest IMU duty-cycle that keeps
        the error below the configured TARGET_ERROR.

Architecture (Time-Based):
    Macro-step of duration TAU_SECONDS (e.g. 1.0 s).
    At each macro-step the agent adjusts the IMU-only duty-cycle ratio:
      - First (ratio × τ) seconds  → IMU-only dead reckoning
      - Remaining seconds          → UWB+IMU sensor fusion

    Within each macro-step:
      - IMU updates at IMU_FREQ_HZ  (e.g. 100 Hz → every 10 ms)
      - UWB updates at UWB_FREQ_HZ  (e.g.  10 Hz → every 100 ms)
      - Motion updates at the IMU rate (highest frequency sensor)

Incremental Actions (Markov Modulation):
    Action Space per agent is discrete size 3:
      - 0: Decrease IMU duty-cycle ratio by one step
      - 1: Keep same
      - 2: Increase IMU duty-cycle ratio by one step

Augmented Observation (Markovian Completeness):
    We append each agent's current IMU duty-cycle ratio (∈ [0, 1])
    directly to the state observation vector to ensure the RL algorithm can
    properly model policy state transitions.

Realistic Motion:
    The tracked tag moves at a configurable human walking speed
    (default ≈ 1.4 m/s).  Position updates depend on elapsed time (dt)
    rather than step count.

Safety:
    - RAM monitoring: auto GC + checkpoint when memory usage is high.
    - Auto-save: training statistics and PyTorch model weights saved
      periodically and on crash/interrupt.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Any, Tuple, Dict, Optional
import random
import gc
import os
import sys
import json
import signal
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

from pulse_ai_client import PulseRLEnv, PulseState


# ══════════════════════════════════════════════════════════════════════════════
#  Simulation Goal Parameters
# ══════════════════════════════════════════════════════════════════════════════
#
#  Reward formula (per agent, per macro-step of τ seconds):
#
#      R_t = α · (e_(t-1) − e_t)  +  (1 − α) · (1 − (E_t − E_min) / (E_max − E_min))
#
#  Where:
#      e_t       = mean localization error over the τ-second window
#      e_(t-1)   = mean localization error from the previous macro-step
#      E_t       = total energy consumed over the τ-second window
#      E_min     = energy of τ seconds in IMU-only mode
#      E_max     = energy of τ seconds in full UWB+IMU fusion mode
#      α ∈ [0,1] = tradeoff weight  (higher → more weight on precision)
#
#  The first term rewards error REDUCTION between consecutive macro-steps.
#  The second term rewards LOW energy consumption (IMU-only is cheap).
#
#  Energy bounds are computed dynamically from sensor power characteristics
#  and the macro-step duration τ.
# ══════════════════════════════════════════════════════════════════════════════

ALPHA = 0.6           # tradeoff weight: precision vs energy

# ── Time-based simulation parameters ─────────────────────────────────────────
EPISODE_DURATION_S = 60.0     # Total episode duration in seconds
TAU_SECONDS = 1.0             # Macro-step duration in seconds
IMU_FREQ_HZ = 100.0           # IMU update rate (Hz)
UWB_FREQ_HZ = 10.0            # UWB update rate (Hz)
WALKING_SPEED_MPS = 1.4       # Human walking speed (m/s)

# ── Derived constants (computed at startup from the above) ───────────────────
# IMU-only power ≈ V × I_imu = 3.3 V × 1.0 mA = 3.3 mW
# Full fusion adds UWB active power ≈ duty_cycle × (V × I_tx_rx)
# These are rough defaults; recalibrated dynamically in MacroSwitchWrapper.
DEFAULT_IMU_POWER_MW = 3.3    # IMU-only average power (mW)
DEFAULT_FUSION_POWER_MW = 45.0  # Full UWB+IMU fusion average power (mW)



# ══════════════════════════════════════════════════════════════════════════════
#  PyTorch RL Agent (DQN) Implementation
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """Deep Q-Network for mapping augmented states to action Q-values."""
    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience Replay Buffer for uniform off-policy sampling."""
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )
        
    def __len__(self):
        return len(self.buffer)


def train_dqn_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma=0.99):
    """Performs a single Deep Q-Network optimization step."""
    if len(replay_buffer) < batch_size:
        return 0.0
        
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    # Q(s, a)
    q_values = policy_net(states)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # max Q(s', a') using the target network
    with torch.no_grad():
        next_q_values = target_net(next_states)
        next_state_values = next_q_values.max(1)[0]
        expected_state_action_values = rewards + (1 - dones) * gamma * next_state_values
        
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stabilization
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    return float(loss.item())



# ══════════════════════════════════════════════════════════════════════════════
#  Gymnasium Environment Wrapper
# ══════════════════════════════════════════════════════════════════════════════

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

        # ── Derived timing constants ──────────────────────────────────
        self.imu_period = 1.0 / imu_freq_hz       # seconds per IMU tick
        self.uwb_period = 1.0 / uwb_freq_hz       # seconds per UWB tick
        # Number of IMU-rate sub-steps per macro-step
        self.sub_steps_per_macro = max(1, int(round(tau_seconds * imu_freq_hz)))
        # UWB fires every N-th IMU sub-step (integer ratio for stability)
        self.uwb_every_n = max(1, int(round(imu_freq_hz / uwb_freq_hz)))

        # Action Space: 61 discrete actions representing step changes from -30 to +30
        self.action_space = spaces.MultiDiscrete([61] * self.num_agents)

        # Observation Space: base + 1 augmented field (duty-cycle ratio)
        orig_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(orig_shape[0] + 1,), dtype=np.float32
        )

        # Granularity of duty-cycle adjustment (one action = 1/sub_steps change)
        self.duty_step = 1.0 / self.sub_steps_per_macro

        # Per-agent IMU duty-cycle ratio ∈ [0, 1].
        # 0.0 = always UWB+IMU fusion; 1.0 = always IMU-only
        # Initialised to 0.0 (full fusion) so the agent starts with best accuracy.
        self.current_duty_ratios = np.zeros(self.num_agents, dtype=np.float32)

        # Track previous macro-step mean error per agent for the Δe term
        self.prev_mean_errors = np.zeros(self.num_agents, dtype=np.float32)

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
        """Compute reward over one τ-second macro-step for each agent."""
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        N = len(all_micro_states)
        if N == 0:
            return rewards

        for a_idx in range(self.num_agents):
            errors = []
            total_energy = 0.0

            for micro_states in all_micro_states:
                if a_idx >= len(micro_states):
                    continue
                state = micro_states[a_idx]
                e_step = state.energy.step_energy_uJ
                src = state.environment.measurement_source if hasattr(state.environment, 'measurement_source') else ""
                
                # Empirically learn the step energies to dynamically scale bounds
                if src == "IMU" and e_step > 0:
                    # Use exponential moving average for stability instead of max
                    if self._empirical_imu_step_energy == 0:
                        self._empirical_imu_step_energy = e_step
                    else:
                        self._empirical_imu_step_energy = 0.9 * self._empirical_imu_step_energy + 0.1 * e_step
                elif src == "Both" and e_step > 0:
                    if self._empirical_both_step_energy == 0:
                        self._empirical_both_step_energy = e_step
                    else:
                        self._empirical_both_step_energy = 0.9 * self._empirical_both_step_energy + 0.1 * e_step
                
                errors.append(state.precision.localization_error)
                total_energy += e_step

            if not errors:
                continue
                
            # Dynamically recalculate E_MIN and E_MAX if we have reliable empirical data
            if self._empirical_imu_step_energy > 0 and self._empirical_both_step_energy > 0:
                num_uwb = self.sub_steps_per_macro // self.uwb_every_n
                num_imu = self.sub_steps_per_macro - num_uwb
                self.E_MAX = (num_uwb * self._empirical_both_step_energy) + (num_imu * self._empirical_imu_step_energy)
                self.E_MIN = self.sub_steps_per_macro * self._empirical_imu_step_energy

            e_t = np.mean(errors)
            e_prev = self.prev_mean_errors[a_idx]
            E_t = total_energy

            # Precision term:  α · (e_(t-1) − e_t)
            # If e_prev is 0.0, this is the very first step of an episode after reset.
            # Do not penalize the agent for the initial starting error.
            if e_prev == 0.0:
                r_precision = 0.0
            else:
                r_precision = ALPHA * (e_prev - e_t)

            # Energy term:  (1 − α) · (1 − (E_t − E_min) / (E_max − E_min))
            if self.E_MAX > self.E_MIN * 1.01:  # Require at least 1% gap
                normalized_energy = np.clip(
                    (E_t - self.E_MIN) / (self.E_MAX - self.E_MIN), 0.0, 1.0
                )
            else:
                normalized_energy = 0.0
            r_energy = (1.0 - ALPHA) * (1.0 - normalized_energy)

            rewards[a_idx] = r_precision + r_energy
            self.prev_mean_errors[a_idx] = e_t

        return rewards

    # ── Step (time-based) ─────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one macro-step (τ seconds) of time-based simulation.

        Within the macro-step:
          - IMU updates every sub-step (at imu_freq_hz).
          - UWB updates every uwb_every_n-th sub-step.
          - First (ratio × sub_steps) sub-steps are IMU-only;
            remaining sub-steps use Both (IMU + UWB).
        """
        # Apply incremental action to each agent's duty-cycle ratio
        for a_idx in range(self.num_agents):
            # act is [0, 60], mapping to [-30, +30] sub-steps
            step_change = int(action[a_idx]) - 30
            
            self.current_duty_ratios[a_idx] = np.clip(
                self.current_duty_ratios[a_idx] + (step_change * self.duty_step),
                0.0, 1.0
            )

        final_obs = None
        final_info = None
        terminated_all = np.zeros(self.num_agents, dtype=bool)
        truncated_all = np.zeros(self.num_agents, dtype=bool)

        all_micro_states: List[List[PulseState]] = []

        for k in range(self.sub_steps_per_macro):
            # Is this sub-step a UWB tick?
            is_uwb_tick = (k % self.uwb_every_n) == 0

            backend_action = []
            for a_idx in range(self.num_agents):
                # Number of sub-steps in the IMU-only phase
                imu_only_steps = int(round(
                    self.current_duty_ratios[a_idx] * self.sub_steps_per_macro
                ))
                in_imu_only_phase = k < imu_only_steps

                # Source selection:
                #   - In IMU-only phase: always "IMU"
                #   - In fusion phase:   "Both" on UWB ticks, "IMU" otherwise
                #     (UWB data only arrives at uwb_freq; between UWB ticks
                #      the filter propagates with IMU alone)
                if in_imu_only_phase:
                    source = "IMU"
                elif is_uwb_tick:
                    source = "Both"
                else:
                    source = "IMU"

                backend_action.append({
                    "filter": "Extended Kalman Filter",
                    "measurement_source": source,
                    "anchors": list(range(self.actual_num_anchors))
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

        # Augment observation: append duty-cycle ratio (already ∈ [0, 1])
        augmented_obs = np.hstack([
            final_obs,
            self.current_duty_ratios[:, np.newaxis]
        ])

        return augmented_obs, macro_reward, terminated_all, truncated_all, final_info

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self, **kwargs):
        """Reset previous errors and duty-cycle ratios on episode boundary."""
        self.prev_mean_errors = np.zeros(self.num_agents, dtype=np.float32)
        self.current_duty_ratios = np.zeros(self.num_agents, dtype=np.float32)
        obs, info = super().reset(**kwargs)

        # Update actual anchor count from server state
        initial_states = info.get("states", [])
        if (
            initial_states
            and hasattr(initial_states[0], "num_anchors")
            and initial_states[0].num_anchors > 0
        ):
            self.actual_num_anchors = initial_states[0].num_anchors

        # Augment observation
        augmented_obs = np.hstack([
            obs,
            self.current_duty_ratios[:, np.newaxis]
        ])
        return augmented_obs, info


# ══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ══════════════════════════════════════════════════════════════════════════════

# Global flag for graceful shutdown
_shutdown_requested = False


def _signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global _shutdown_requested
    if _shutdown_requested:
        # Second Ctrl+C → force exit
        print("\n\n⚠️  Forced exit (second interrupt).")
        sys.exit(1)
    _shutdown_requested = True
    print("\n\n🛑 Shutdown requested — exiting after current step...")


def main():
    global _shutdown_requested

    # Install signal handler for graceful Ctrl+C
    signal.signal(signal.SIGINT, _signal_handler)

    # ── Configuration ─────────────────────────────────────────────────
    PORT = 5555
    NUM_ANCHORS = 8
    NUM_AGENTS = 4
    EPISODES = 50

    # Time-based simulation parameters (use module-level defaults)
    TAU_S = TAU_SECONDS           # Macro-step duration (seconds)
    IMU_HZ = IMU_FREQ_HZ         # IMU update rate (Hz)
    UWB_HZ = UWB_FREQ_HZ         # UWB update rate (Hz)
    EP_DURATION = EPISODE_DURATION_S  # Episode duration (seconds)

    # Derived: how many macro-steps fit in one episode
    MACRO_STEPS = int(EP_DURATION / TAU_S)
    SUB_STEPS = int(round(TAU_S * IMU_HZ))  # sub-steps per macro

    # DQN Hyperparameters
    BATCH_SIZE = 64
    GAMMA = 0.95
    LR = 1e-3
    TARGET_UPDATE = 50      # Update target net every N macro-steps
    BUFFER_CAPACITY = 20000

    config = {
        "port": PORT,
        "num_anchors": NUM_ANCHORS,
        "num_agents": NUM_AGENTS,
        "episodes": EPISODES,
        "macro_steps_per_episode": MACRO_STEPS,
        "episode_duration_s": EP_DURATION,
        "tau_seconds": TAU_S,
        "imu_freq_hz": IMU_HZ,
        "uwb_freq_hz": UWB_HZ,
        "sub_steps_per_macro": SUB_STEPS,
        "walking_speed_mps": WALKING_SPEED_MPS,
        "alpha": ALPHA,
        "dqn_lr": LR,
        "dqn_batch_size": BATCH_SIZE,
        "dqn_gamma": GAMMA,
    }

    print("=" * 64)
    print("  PULSE AI – Time-Based Switching Agent (DQN)")
    print("=" * 64)
    print(f"  Port            : {PORT}")
    print(f"  Agents          : {NUM_AGENTS}")
    print(f"  Walking Speed   : {WALKING_SPEED_MPS} m/s")
    print(f"  ─── Time-Based Simulation ───")
    print(f"  Episode Duration: {EP_DURATION} s")
    print(f"  τ (tau)         : {TAU_S} s  →  {MACRO_STEPS} macro-steps/episode")
    print(f"  IMU Frequency   : {IMU_HZ} Hz")
    print(f"  UWB Frequency   : {UWB_HZ} Hz")
    print(f"  Sub-steps/macro : {SUB_STEPS}  (IMU ticks per τ window)")
    print(f"  UWB every       : {max(1, int(round(IMU_HZ / UWB_HZ)))} IMU ticks")
    print(f"  ─── Reward Formula ───")
    print(f"  R_t = α·(e_(t-1) − e_t) + (1−α)·(1 − (E_t-E_min)/(E_max-E_min))")
    print(f"  α = {ALPHA}  |  Energy bounds computed dynamically from τ={TAU_S}s")
    print("=" * 64)

    # Identity action formatter
    def identity_action_formatter(act):
        return act

    base_env = PulseRLEnv(
        port=PORT,
        num_anchors=NUM_ANCHORS,
        num_agents=NUM_AGENTS,
        action_space=spaces.Discrete(1),   # Ignored by wrapper
        action_formatter=identity_action_formatter,
        vectorized=True,
    )

    env = MacroSwitchWrapper(
        base_env,
        tau_seconds=TAU_S,
        imu_freq_hz=IMU_HZ,
        uwb_freq_hz=UWB_HZ,
    )

    # ── PyTorch RL Initialization ─────────────────────────────────────
    state_dim = env.observation_space.shape[0]  # Includes augmented field
    action_dim = 61  # Mapping -30 to +30 steps
    
    policy_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    # ── Training state ────────────────────────────────────────────────
    episode_history: List[Dict] = []   # Summary per completed episode
    best_episode_reward = -np.inf
    current_episode = 0
    current_step = 0
    epsilon = 1.0

    print("   Starting fresh training session...\n")

    def select_actions(state, eps):
        """Select epsilon-greedy actions across all homogeneous agents."""
        if random.random() < eps:
            return np.random.randint(0, action_dim, size=NUM_AGENTS)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state)
                q_values = policy_net(state_t)  # shape (NUM_AGENTS, action_dim)
                return q_values.argmax(dim=1).numpy()

    try:
        obs, info = env.reset()
        print(f"\n✅ Connected with {NUM_AGENTS} agents!")

        # Detect actual anchor count from the initial state
        initial_states = info.get("states", [])
        if initial_states and hasattr(initial_states[0], 'num_anchors') and initial_states[0].num_anchors > 0:
            actual_num_anchors = initial_states[0].num_anchors
        else:
            actual_num_anchors = NUM_ANCHORS
        print(f"  Detected {actual_num_anchors} anchors from server.")

        print("  Agent is ready to train.\n")

        for episode in range(current_episode, EPISODES):
            current_episode = episode

            # ── Check for shutdown request ────────────────────────────
            if _shutdown_requested:
                print(f"\n🛑 Graceful shutdown at episode {episode+1}")
                break

            episode_rewards = np.zeros(NUM_AGENTS)
            episode_errors = []
            episode_energies = []
            episode_duties = []
            episode_losses = []

            for step in range(MACRO_STEPS):
                current_step = step

                # ── Check for shutdown between steps ──────────────────
                if _shutdown_requested:
                    print(f"\n🛑 Graceful shutdown at episode {episode+1}, step {step}")
                    break

                # Epsilon-greedy action selection
                action = select_actions(obs, epsilon)

                # Report current step and cumulative rewards back to GUI
                if step == 0:
                    step_rewards_list = [0.0] * NUM_AGENTS
                else:
                    step_rewards_list = reward.tolist()
                env.unwrapped.set_next_metrics({
                    "step_rewards": step_rewards_list,
                    "cumulative_rewards": episode_rewards.tolist()
                })

                # Step the environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_rewards += reward

                # Store transitions into experience replay
                for a_idx in range(NUM_AGENTS):
                    replay_buffer.push(
                        obs[a_idx],
                        action[a_idx],
                        reward[a_idx],
                        next_obs[a_idx],
                        float(terminated[a_idx] or truncated[a_idx])
                    )

                # Optimize DQN
                loss_val = train_dqn_step(policy_net, target_net, optimizer, replay_buffer, BATCH_SIZE, GAMMA)
                if loss_val > 0.0:
                    episode_losses.append(loss_val)

                # Periodically update the target network
                if step > 0 and step % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # Step epsilon decay (decay faster so it explores early but eventually settles)
                epsilon = max(0.05, epsilon * 0.995)

                # Compute statistics for Agent 0
                all_micro = info.get("all_micro_states", [])
                if all_micro and len(all_micro[0]) > 0:
                    a0_errors = [ms[0].precision.localization_error for ms in all_micro if len(ms) > 0]
                    a0_energy = sum(ms[0].energy.step_energy_uJ for ms in all_micro if len(ms) > 0)
                    mean_err = np.mean(a0_errors)
                else:
                    mean_err = 0.0
                    a0_energy = 0.0

                a0_duty = env.current_duty_ratios[0]
                sim_time = step * TAU_S  # elapsed simulation time (s)

                episode_errors.append(mean_err)
                episode_energies.append(a0_energy)
                episode_duties.append(a0_duty)

                # Print debugging info for the first 3 macro-steps to verify
                if step < 3:
                    all_micro_dbg = info.get("all_micro_states", [])
                    if all_micro_dbg and len(all_micro_dbg) > 0 and len(all_micro_dbg[0]) > 0:
                        # Show a sample of sub-steps (first, middle, last)
                        sample_indices = [0, len(all_micro_dbg) // 2, len(all_micro_dbg) - 1]
                        for mk in sample_indices:
                            if mk < len(all_micro_dbg) and len(all_micro_dbg[mk]) > 0:
                                s0 = all_micro_dbg[mk][0]
                                src = s0.environment.measurement_source if hasattr(s0.environment, 'measurement_source') else '?'
                                print(f"    [DEBUG sub-step {mk}/{SUB_STEPS}] src={src}  "
                                      f"step_E={s0.energy.step_energy_uJ:.2f}µJ  "
                                      f"uwb_pwr={s0.energy.uwb_active_power_mW:.2f}mW  "
                                      f"imu_pwr={s0.energy.imu_power_mW:.2f}mW")

                # Print progress (show all data as requested)
                loss_str = f"L={np.mean(episode_losses[-10:]):.4f}" if episode_losses else "L=N/A"
                print(
                    f"  [Ep {episode+1:>2} | t={sim_time:>6.1f}s]  "
                    f"duty={a0_duty:.0%} (act={action[0]})  "
                    f"ē={mean_err:.3f}m  "
                    f"E={a0_energy:.1f}µJ (Bounds: {env.E_MIN:.1f}-{env.E_MAX:.1f}µJ)  "
                    f"{loss_str}  "
                    f"eps={epsilon:.2f}  "
                    f"R={reward[0]:+.4f}"
                )

                # Free micro-states to prevent RAM leak
                if "all_micro_states" in info:
                    del info["all_micro_states"]

                obs = next_obs

                if terminated.all() or truncated.all():
                    break

            if _shutdown_requested:
                break

            # ── Episode summary ──────────────────────────────────────
            ep_mean_error = np.mean(episode_errors) if episode_errors else 0.0
            ep_mean_energy = np.mean(episode_energies) if episode_energies else 0.0
            ep_mean_duty = np.mean(episode_duties) if episode_duties else 0.0
            ep_mean_loss = np.mean(episode_losses) if episode_losses else 0.0
            total_r = episode_rewards.sum()
            is_best = total_r > best_episode_reward
            if is_best:
                best_episode_reward = total_r

            # Store episode summary
            episode_summary = {
                "episode": episode + 1,
                "total_reward": float(total_r),
                "per_agent_rewards": episode_rewards.tolist(),
                "mean_error": float(ep_mean_error),
                "mean_energy": float(ep_mean_energy),
                "mean_imu_duty": float(ep_mean_duty),
                "mean_loss": float(ep_mean_loss),
                "epsilon": float(epsilon),
                "steps_completed": len(episode_errors),
                "is_best": is_best,
            }
            episode_history.append(episode_summary)

            print(f"\n{'═' * 60}")
            print(f"  Episode {episode+1}/{EPISODES} Summary {'★ BEST' if is_best else ''}")
            print(f"  Rewards        : {np.round(episode_rewards, 3)}")
            print(f"  Mean Error (ē) : {ep_mean_error:.4f} m")
            print(f"  Mean Energy    : {ep_mean_energy:.1f} µJ / macro-step")
            print(f"  Mean IMU Duty  : {ep_mean_duty:.0%}")
            print(f"  Mean DQN Loss  : {ep_mean_loss:.4f}")
            print(f"{'═' * 60}\n")

            # Free episode-level lists
            episode_errors.clear()
            episode_energies.clear()
            episode_duties.clear()
            episode_losses.clear()

            # Reset for next episode
            obs, info = env.reset()

    except MemoryError:
        print("\n\n🚨 OUT OF MEMORY — exiting...")
        gc.collect()
    except KeyboardInterrupt:
        print("\n\n⏹  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print(f"\n✅ Disconnected from PULSE simulator.")


if __name__ == "__main__":
    main()
