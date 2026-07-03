# ══════════════════════════════════════════════════════════════════════════════
#  Simulation Goal Parameters
# ══════════════════════════════════════════════════════════════════════════════
#
#  Reward formula (per agent, per macro-step of τ seconds):
#
#      R_t = α · (e_shadow_imu − e_fusion)  +  (1 − α) · (1 − (E_t − E_min) / (E_max − E_min))
#
#  Where:
#      e_shadow_imu = mean error of a shadow IMU-only filter running in parallel
#                     during the fusion window (counterfactual: what would IMU-only
#                     have produced if fusion never started)
#      e_fusion     = mean localization error of the real fused filter during
#                     the UWB fusion window
#      E_t          = total energy consumed over the τ-second window
#      E_min        = energy of τ seconds in IMU-only mode
#      E_max        = energy of τ seconds in full UWB+IMU fusion mode
#      α ∈ [0,1]    = tradeoff weight  (higher → more weight on precision)
#
#  The first term rewards the VALUE of fusion: how much error reduction fusion
#  achieves compared to continuing IMU-only dead reckoning.
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
