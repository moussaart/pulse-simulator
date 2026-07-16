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
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


def save_policy(policy_net, prefix="policy_switch_mo"):
    """Save the policy weights in PyTorch and JSON format for external hardware deployment."""
    os.makedirs("checkpoints", exist_ok=True)
    pth_path = os.path.join("checkpoints", f"{prefix}.pth")
    json_path = os.path.join("checkpoints", f"{prefix}.json")
    
    try:
        # Save PyTorch state dict
        torch.save(policy_net.state_dict(), pth_path)
        print(f"\n💾 PyTorch policy saved to: {pth_path}")
        
        # Save weights in JSON format for easy deployment on external hardware (C/C++, MicroPython, etc.)
        model_weights = {}
        for name, param in policy_net.state_dict().items():
            model_weights[name] = param.cpu().numpy().tolist()
            
        with open(json_path, 'w') as f:
            json.dump(model_weights, f, indent=4)
        print(f"💾 JSON policy weights saved to: {json_path} (for embedded/external hardware implementation)")
        
    except Exception as e:
        print(f"\n⚠️ Failed to save policy: {e}")

from pulse_ai_client import PulseRLEnv

from .config import (
    W_ERROR,
    W_STD,
    W_ENERGY,
    ERROR_TARGET_MEAN,
    ERROR_TARGET_STD,
    EPISODE_DURATION_S,
    TAU_SECONDS,
    IMU_FREQ_HZ,
    UWB_FREQ_HZ,
    WALKING_SPEED_MPS,
)
from .model import QNetwork, ReplayBuffer, train_dqn_step
from .env_wrapper import MacroSwitchWrapper

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
    PORT = 5555  # Default port for PULSE simulator
    NUM_ANCHORS = 8
    NUM_AGENTS = 4
    EPISODES = 1  # Run for one episode only
    # Time-based simulation parameters (use module-level defaults)
    TAU_S = TAU_SECONDS           # Macro-step duration (seconds)
    IMU_HZ = IMU_FREQ_HZ         # IMU update rate (Hz)
    UWB_HZ = UWB_FREQ_HZ         # UWB update rate (Hz)
    EP_DURATION = EPISODE_DURATION_S  # Episode duration (seconds)

    # Derived: how many macro-steps fit in one episode (approximate for logging)
    if EP_DURATION == float('inf'):
        MACRO_STEPS_APPROX = 999999999
    else:
        MACRO_STEPS_APPROX = int(EP_DURATION / TAU_S)

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
        "macro_steps_per_episode": MACRO_STEPS_APPROX,
        "episode_duration_s": EP_DURATION,
        "tau_seconds": "Dynamic",
        "imu_freq_hz": IMU_HZ,
        "uwb_freq_hz": UWB_HZ,
        "walking_speed_mps": WALKING_SPEED_MPS,
        "w_error": W_ERROR,
        "w_std": W_STD,
        "w_energy": W_ENERGY,
        "error_target_mean": ERROR_TARGET_MEAN,
        "error_target_std": ERROR_TARGET_STD,
        "dqn_lr": LR,
        "dqn_batch_size": BATCH_SIZE,
        "dqn_gamma": GAMMA,
    }

    print("================================================================")
    print("  PULSE AI - Multi-Objective Discrete Switching Agent (DQN)")
    print("================================================================")
    print(f"  Port            : {PORT}")
    print(f"  Agents          : {NUM_AGENTS}")
    print(f"  Walking Speed   : {WALKING_SPEED_MPS} m/s")
    print(f"  --- Time-Based Simulation ---")
    print(f"  Episode Duration: {EP_DURATION} s")
    print(f"  tau             : Dynamic (T_imu + T_fusion)")
    print(f"  IMU Frequency   : {IMU_HZ} Hz")
    print(f"  UWB Frequency   : {UWB_HZ} Hz")
    print(f"  UWB Ratio       : 1 UWB tick every {max(1, int(round(IMU_HZ / UWB_HZ)))} IMU ticks")
    print(f"  Action Space    : Tuple of (T_imu, T_fusion)")
    print("================================================================")
    print(f"  Reward Formula (Continuous Multi-Objective):")
    print(f"  R = w_error * r_error_mean + w_std * r_error_std + w_energy * r_energy")
    print(f"  Targets         : error_mean <= {ERROR_TARGET_MEAN}m, error_std <= {ERROR_TARGET_STD}m")
    print(f"  Penalty         : Continuous Piecewise Linear when exceeding targets")
    print(f"  Weights         : w_error={W_ERROR}, w_std={W_STD}, w_energy={W_ENERGY}")
    print(f"  Mode            : Infinite Single Episode (Ctrl+C to stop)")
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

    # ── PyTorch RL Initialization & GPU Configuration ─────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"👉 Using device for RL training: {device}")
    
    state_dim = env.observation_space.shape[0]  # Includes augmented field
    action_dim = env.total_actions  # Matrix of (T_imu, T_fusion) pairs
    
    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    # ── Training state ────────────────────────────────────────────────
    episode_history = deque(maxlen=1000)   # Cap to prevent unbounded RAM growth
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
                state_t = torch.FloatTensor(state).to(device)
                q_values = policy_net(state_t)  # shape (NUM_AGENTS, action_dim)
                return q_values.argmax(dim=1).cpu().numpy()

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

        episode = current_episode
        while episode < EPISODES:
            current_episode = episode

            # ── Check for shutdown request ────────────────────────────
            if _shutdown_requested:
                print(f"\n🛑 Graceful shutdown at episode {episode+1}")
                break

            episode_rewards = np.zeros(NUM_AGENTS)
            episode_errors = []
            episode_energies = []
            episode_t_imus = []
            episode_t_fusions = []
            episode_losses = []

            step = 0
            sim_time = 0.0
            while sim_time < EP_DURATION:
                current_step = step

                # ── Check for shutdown between steps ──────────────────
                if _shutdown_requested:
                    print(f"\n🛑 Graceful shutdown at episode {episode+1}, step {step}")
                    break

                # Epsilon-greedy action selection
                t_inf_start = time.perf_counter()
                action = select_actions(obs, epsilon)
                t_nn_inference = time.perf_counter() - t_inf_start

                # Report current step and cumulative rewards back to GUI
                if step == 0:
                    step_rewards_list = [0.0] * NUM_AGENTS
                else:
                    step_rewards_list = reward.tolist()
                
                # Fetch optimizer time from the previous loop iteration (if initialized)
                opt_time_val = locals().get('t_optimizer', 0.0)
                env.unwrapped.set_next_metrics({
                    "step_rewards": step_rewards_list,
                    "cumulative_rewards": episode_rewards.tolist(),
                    "nn_inference_time": t_nn_inference,
                    "optimizer_time": opt_time_val
                })

                # Step the environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                episode_rewards += reward
                sim_time += env.tau_seconds

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
                t_opt_start = time.perf_counter()
                loss_val = train_dqn_step(policy_net, target_net, optimizer, replay_buffer, BATCH_SIZE, GAMMA, device=device)
                t_optimizer = time.perf_counter() - t_opt_start
                if loss_val > 0.0:
                    episode_losses.append(loss_val)

                # Periodically update the target network
                if step > 0 and step % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # Step epsilon decay
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

                a0_t_imu = env.current_t_imu[0]
                a0_t_fusion = env.current_t_fusion[0]
                a0_tau = env.tau_seconds

                episode_errors.append(mean_err)
                episode_energies.append(a0_energy)
                episode_t_imus.append(a0_t_imu)
                episode_t_fusions.append(a0_t_fusion)

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
                                print(f"    [DEBUG sub-step {mk}/{env.sub_steps_per_macro}] src={src}  "
                                      f"step_E={s0.energy.step_energy_uJ:.2f}µJ  "
                                      f"uwb_pwr={s0.energy.uwb_active_power_mW:.2f}mW  "
                                      f"imu_pwr={s0.energy.imu_power_mW:.2f}mW")

                # Print progress
                loss_str = f"L={np.mean(episode_losses[-10:]):.4f}" if episode_losses else "L=N/A"
                print(
                    f"  [Time: {sim_time:>5.1f}s] "
                    f"T_imu: {a0_t_imu:.2f}s | T_fusion: {a0_t_fusion:.2f}s | "
                    f"Err: {mean_err:.2f}m | Energy: {a0_energy/1000:.1f}mJ | "
                    f"Reward: {reward[0]:+.2f}"
                )

                # Free micro-states to prevent RAM leak
                if "all_micro_states" in info:
                    del info["all_micro_states"]

                obs = next_obs
                step += 1

                if terminated.all() or truncated.all():
                    break

            if _shutdown_requested:
                break

            # ── Episode summary ──────────────────────────────────────
            ep_mean_error = np.mean(episode_errors) if episode_errors else 0.0
            ep_mean_energy = np.mean(episode_energies) if episode_energies else 0.0
            ep_mean_t_imu = np.mean(episode_t_imus) if episode_t_imus else 0.0
            ep_mean_t_fusion = np.mean(episode_t_fusions) if episode_t_fusions else 0.0
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
                "mean_t_imu": float(ep_mean_t_imu),
                "mean_t_fusion": float(ep_mean_t_fusion),
                "mean_loss": float(ep_mean_loss),
                "epsilon": float(epsilon),
                "steps_completed": len(episode_errors),
                "is_best": is_best,
            }
            episode_history.append(episode_summary)

            print(f"\n{'═' * 60}")
            print(f"  Episode {episode+1} Summary {'★ BEST' if is_best else ''}")
            print(f"  Rewards        : {np.round(episode_rewards, 3)}")
            print(f"  Mean Error (ē) : {ep_mean_error:.4f} m")
            print(f"  Mean Energy    : {ep_mean_energy:.1f} µJ / macro-step")
            print(f"  Mean T_imu     : {ep_mean_t_imu:.2f} s")
            print(f"  Mean T_fusion  : {ep_mean_t_fusion:.2f} s")
            print(f"  Mean DQN Loss  : {ep_mean_loss:.4f}")
            print(f"{'═' * 60}\n")

            # Free episode-level lists
            episode_errors.clear()
            episode_energies.clear()
            episode_t_imus.clear()
            episode_t_fusions.clear()
            episode_losses.clear()

            # Reset for next episode
            obs, info = env.reset()
            episode += 1

    except MemoryError:
        print("\n\n🚨 OUT OF MEMORY — exiting...")
        gc.collect()
    except KeyboardInterrupt:
        print("\n\n⏹  Training interrupted by user.")
    except (TimeoutError, ConnectionError, socket.timeout) as se:
        print(f"\n📡 Simulator connection lost or timed out ({type(se).__name__}). Stopping training session gracefully...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'policy_net' in locals():
            save_policy(policy_net)
        env.close()
        print(f"\n✅ Disconnected from PULSE simulator.")
