"""
PULSE AI - Markov Modulated Switching Agent (PyTorch DQN Edition)
                         — Time-Based Simulation —
                     MULTI-OBJECTIVE REWARD EDITION (Revision 4)

Goal:
    The agent learns a control policy using a continuous multi-objective reward
    function to keep error below 0.2 m and standard deviation below 0.15 m,
    and to minimize energy consumption by comparing the current step's energy
    consumption against the previous step's energy consumption.
"""

import sys
import os

# Append the directory containing agent_switch_mo and its parent to sys.path to ensure correct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from agent_switch_mo.train import main

if __name__ == "__main__":
    main()
