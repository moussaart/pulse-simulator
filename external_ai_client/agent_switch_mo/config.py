# ══════════════════════════════════════════════════════════════════════════════
#  Simulation Goal Parameters (Discrete Multi-Objective Reward)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Reward formula (per agent, per macro-step/cycle of τ seconds):
#
#      R = W_ERROR * r_error_mean + W_STD * r_error_std + W_ENERGY * r_energy
#
#  Where:
#      r_error_mean:
#          +1 if e_mean < ERROR_TARGET_MEAN (0.1 m)
#          -1 if e_mean > ERROR_TARGET_MEAN (0.1 m)
#           0 otherwise
#      r_error_std:
#          +1 if e_std < ERROR_TARGET_STD (0.15 m)
#          -1 if e_std > ERROR_TARGET_STD (0.15 m)
#           0 otherwise
#      r_energy:
#          +1 if E_t < E_{t-1} (energy decreased)
#          -1 if E_t > E_{t-1} (energy increased)
#           0 if E_t == E_{t-1}
# ══════════════════════════════════════════════════════════════════════════════

# ── Multi-Objective Reward Weights & Targets ─────────────────────────────────
W_ERROR = 1.0               # Weight for average tracking error reward
W_STD = 0.5                 # Weight for error standard deviation reward
W_ENERGY = 1.0              # Weight for energy comparison reward

ERROR_TARGET_MEAN = 0.2     # Target mean error (meters)
ERROR_TARGET_STD = 0.15     # Target error standard deviation (meters)

# ── Time-based simulation parameters ─────────────────────────────────────────
EPISODE_DURATION_S = float('inf')  # Infinite episode duration for continuous training
TAU_SECONDS = 1.0                  # Macro-step duration in seconds
IMU_FREQ_HZ = 100.0                # IMU update rate (Hz)
UWB_FREQ_HZ = 10.0                 # UWB update rate (Hz)
WALKING_SPEED_MPS = 1.4            # Human walking speed (m/s)

# ── Derived constants (computed at startup from the above) ───────────────────
DEFAULT_IMU_POWER_MW = 3.3         # IMU-only average power (mW)
DEFAULT_FUSION_POWER_MW = 45.0     # Full UWB+IMU fusion average power (mW)
