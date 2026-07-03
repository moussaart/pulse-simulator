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

import sys
import os

# Append the directory containing agent_switch to sys.path to ensure correct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_switch.train import main

if __name__ == "__main__":
    main()
