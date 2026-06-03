"""
PULSE AI Training Agent — Multi-Agent Generic Script

Demonstrates connecting to the PULSE simulator and training an RL agent
using task-agnostic dictionaries for actions (filter selection, sensors, etc.)
and vectorized outputs for multiple simultaneous agents.

Usage:
    1. Start the PULSE simulator and open the AI Training Window.
    2. Ensure the port matches (default 5555).
    3. Click "Start Training" in the simulator.
    4. Run this script:
        python train_agent.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Any
import random

from pulse_ai_client import PulseRLEnv, PulseState


# The 5 filters we want the agents to choose between
AVAILABLE_FILTERS = [
    "Improved Adaptive EKF",
    "Extended Kalman Filter",
    "NLOS-Aware",
    "Trilateration",
    "Min-Max"
]

def format_multi_agent_actions(action: np.ndarray, num_anchors: int) -> List[dict]:
    """
    Map the raw RL action to the generic task-agnostic dictionary expected by the server.
    
    action: array of shape (NUM_AGENTS, 2)
        - Column 0: Filter index (0-4)
        - Column 1: IMU on/off (0 or 1)
    """
    actions = []
    for a in action:
        filter_idx = int(a[0])
        use_imu = bool(a[1])
        
        # We can dynamically pick anchors, or just use all available
        anchors_to_use = list(range(num_anchors))
        
        agent_action = {
            "filter": AVAILABLE_FILTERS[filter_idx],
            "measurement_source": "Both" if use_imu else "UWB",
            "anchors": anchors_to_use
        }
        actions.append(agent_action)
        
    return actions


def main():
    # ── Configuration ─────────────────────────────────────────────────
    PORT = 5555
    NUM_ANCHORS = 8
    NUM_AGENTS = 3
    EPISODES = 100
    MAX_STEPS = 500

    print("=" * 60)
    print("  PULSE AI Multi-Agent Training (Generic Architecture)")
    print("=" * 60)
    print(f"  Port: {PORT} | Anchors: {NUM_ANCHORS} | Agents: {NUM_AGENTS}")
    print("=" * 60)

    # ── Environment Definition ─────────────────────────────────────────
    # Action Space: Per agent, [filter_index (0-4), use_imu (0-1)]
    multi_action_space = spaces.MultiDiscrete([len(AVAILABLE_FILTERS), 2] * NUM_AGENTS)
    
    # We will use the built-in obs_formatter but define our own action formatter
    def custom_action_formatter(act: np.ndarray):
        # Reshape to (NUM_AGENTS, 2)
        act = act.reshape((NUM_AGENTS, 2))
        return format_multi_agent_actions(act, NUM_ANCHORS)

    env = PulseRLEnv(
        port=PORT,
        num_anchors=NUM_ANCHORS,
        num_agents=NUM_AGENTS,
        action_space=multi_action_space,
        action_formatter=custom_action_formatter,
        vectorized=True
    )

    try:
        obs, info = env.reset()
        print(f"\n✅ Connected with {NUM_AGENTS} agents!")

        # ── Training Loop ─────────────────────────────────────────────
        for episode in range(EPISODES):
            episode_rewards = np.zeros(NUM_AGENTS)

            for step in range(MAX_STEPS):
                # Random multi-agent action (replace with your RL model)
                action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_rewards += reward
                
                states: List[PulseState] = info["states"]

                # Print state for Agent 0 every 50 steps
                if step % 50 == 0:
                    a0_state = states[0]
                    a0_act = action[:2]
                    a0_filter = AVAILABLE_FILTERS[a0_act[0]]
                    
                    print(f"\n  [Ep {episode+1} | Step {step}] -> Agent 0 Overview")
                    print(f"    Filter Used: {a0_filter}")
                    print(f"    Sensors: {'UWB+IMU' if a0_act[1] else 'UWB Only'}")
                    print(f"    Error: {a0_state.precision.localization_error:.3f}m")
                    print(f"    Energy: {a0_state.energy.step_energy_uJ:.1f} µJ/step")
                    print(f"    Agent 0 Reward: {reward[0]:.4f}")

                if terminated.all() or truncated.all():
                    break

            print(f"\n── Episode {episode+1}/{EPISODES} ──")
            print(f"   Rewards: {np.round(episode_rewards, 2)}")

            # Reset for next episode
            obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n\n⏹  Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\n✅ Disconnected from PULSE simulator.")


if __name__ == "__main__":
    main()
