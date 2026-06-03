"""
Gymnasium environment that wraps the PulseClient for RL training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List, Callable

from .client import PulseClient
from .state import PulseState


class PulseRLEnv(gym.Env):
    """
    OpenAI Gymnasium environment for RL-based training in PULSE.

    This generic environment allows users to define custom action and observation
    spaces via callback formatters, supporting both single and multi-agent
    vectorized structures.

    Args:
        host: Server hostname
        port: Server TCP port (must match the PULSE GUI port spinner)
        num_anchors: Expected number of anchors in the scenario
        num_agents: Number of simultaneous agents
        action_space: User-defined action space
        observation_space: User-defined observation space
        action_formatter: Function mapping gym action to generic list/dict for server
        obs_formatter: Function mapping List[PulseState] to gym observation
        reward_formatter: Function mapping List[PulseState] to rewards
        vectorized: If True, returns batched (obs, rew, done, info) for all agents
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        num_anchors: int = 8,
        num_agents: int = 1,
        action_space: Optional[spaces.Space] = None,
        observation_space: Optional[spaces.Space] = None,
        action_formatter: Optional[Callable] = None,
        obs_formatter: Optional[Callable] = None,
        reward_formatter: Optional[Callable] = None,
        vectorized: bool = False,
    ):
        super().__init__()

        self.host = host
        self.port = port
        self.num_anchors = num_anchors
        self.num_agents = num_agents
        self.vectorized = vectorized

        self.client = PulseClient(host=host, port=port)

        # Allow user to inject custom formatting logic; fallback to anchor selection defaults
        self.action_space = action_space or spaces.MultiBinary(num_anchors)
        obs_size = num_anchors * 2 + 16
        self.observation_space = observation_space or spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        self.action_formatter = action_formatter or self._default_action_formatter
        self.obs_formatter = obs_formatter or self._default_obs_formatter
        self.reward_formatter = reward_formatter or self._default_reward_formatter

        self._current_states: List[PulseState] = []
        self._connected = False

    def _default_action_formatter(self, action: np.ndarray) -> List[Any]:
        # Legacy single-agent binary mask to anchors list
        selected = [int(i) for i in range(len(action)) if action[i]]
        if not selected:
            selected = list(range(min(3, self.num_anchors)))
        return [selected] * self.num_agents

    def _default_obs_formatter(self, states: List[PulseState]) -> np.ndarray:
        obs_list = []
        for state in states[:self.num_agents]:
            uwb = np.pad(state.measurements.uwb_ranges, (0, max(0, self.num_anchors - len(state.measurements.uwb_ranges))))[:self.num_anchors]
            los = np.pad([1.0 if l else 0.0 for l in state.nlos_info.is_los], (0, max(0, self.num_anchors - len(state.nlos_info.is_los))))[:self.num_anchors]
            imu_acc = state.imu_data.acceleration
            imu_gyro = state.imu_data.angular_velocity
            tag_gt = state.tag_position_gt
            tag_est = state.tag_position_estimated or [0.0, 0.0]
            
            flat = np.concatenate([
                uwb, los, imu_acc, imu_gyro, tag_gt, tag_est,
                [state.precision.localization_error, state.precision.prev_localization_error,
                 state.energy.step_energy_uJ, state.nlos_info.nlos_count, state.precision.gdop or 0.0]
            ], dtype=np.float32)
            obs_list.append(flat)
            
        # Pad if there are fewer states returned than num_agents
        while len(obs_list) < self.num_agents:
            obs_list.append(np.zeros_like(obs_list[0] if obs_list else np.zeros(self.observation_space.shape[0])))
            
        if self.vectorized:
            return np.array(obs_list, dtype=np.float32)
        return obs_list[0]

    def _default_reward_formatter(self, states: List[PulseState]) -> Any:
        # Client-side fallback if server rewards aren't sufficient
        rews = []
        for state in states[:self.num_agents]:
            error_imp = state.precision.prev_localization_error - state.precision.localization_error
            nlos_pen = -0.1 * state.nlos_info.nlos_count
            egy_pen = -0.001 * state.energy.step_energy_uJ
            rews.append(float(error_imp + nlos_pen + egy_pen))
            
        # Pad if there are fewer states returned than num_agents
        while len(rews) < self.num_agents:
            rews.append(0.0)
            
        return np.array(rews, dtype=np.float32) if self.vectorized else rews[0]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed)

        if not self._connected:
            self.client.connect()
            self._connected = True
            self._current_states = self.client.receive_state()
        elif not self._current_states:
            # Fallback if somehow connected but no states
            self._current_states = self.client.receive_state()
            
        # We DO NOT call receive_state() if already connected and we have states, 
        # because the server is currently blocked waiting for our next action.
        
        obs = self.obs_formatter(self._current_states)
        
        info = {"states": self._current_states}
        return obs, info

    def set_next_metrics(self, metrics: Dict[str, float]) -> None:
        """Set metrics to be sent to the server on the next step call."""
        self._next_metrics = metrics

    def step(self, action: Any) -> Tuple[Any, Any, Any, Any, Dict[str, Any]]:
        formatted_action = self.action_formatter(action)
        
        # Include any queued metrics
        metrics = getattr(self, '_next_metrics', None)
        self.client.send_action(formatted_action, metrics=metrics)
        self._next_metrics = None  # Clear after sending

        try:
            self._current_states = self.client.receive_state()
        except ConnectionError:
            dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            if self.vectorized:
                return np.array([dummy_obs]*self.num_agents), np.zeros(self.num_agents), np.ones(self.num_agents, dtype=bool), np.zeros(self.num_agents, dtype=bool), {"reason": "disconnected"}
            return dummy_obs, 0.0, True, False, {"reason": "disconnected"}

        obs = self.obs_formatter(self._current_states)
        reward = self.reward_formatter(self._current_states)
        
        if self.vectorized:
            terminated = np.zeros(self.num_agents, dtype=bool)
            truncated = np.zeros(self.num_agents, dtype=bool)
        else:
            terminated = False
            truncated = False

        info = {"states": self._current_states}
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._connected:
            self.client.close()
            self._connected = False
