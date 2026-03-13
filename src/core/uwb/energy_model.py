"""
UWB Tag Energy Consumption Model

Estimates the energy consumption of a UWB tag during ranging operations.
Supports SS-TWR and DS-TWR protocols, integrates IMU power consumption,
and provides battery life estimation.

Power values are based on the Decawave DW1000 datasheet defaults,
but all parameters are user-configurable.

Usage:
    from src.core.uwb.energy_model import EnergyCalculator, EnergyConfig

    config = EnergyConfig()
    calc = EnergyCalculator(config)
    result = calc.calculate()
    print(result.total_power_mW, result.battery_life_hours)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RangingMode(Enum):
    """Supported TWR ranging protocols."""
    SS_TWR = "SS-TWR"   # Single-Sided TWR – 2 messages (Poll + Response)
    DS_TWR = "DS-TWR"   # Double-Sided TWR – 4 messages (Poll + Resp + Final + DS-Resp)


# Mapping: protocol → (tag TX messages, tag RX messages)
_PROTOCOL_MSG_COUNT: Dict[RangingMode, tuple] = {
    RangingMode.SS_TWR: (1, 1),   # Tag sends Poll, receives Response
    RangingMode.DS_TWR: (2, 2),   # Tag sends Poll+Final, receives Resp+DS-Resp
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnergyConfig:
    """
    All configurable parameters for UWB tag energy estimation.

    Current values (mA) are based on the Decawave DW1000 datasheet.
    Timing values (µs) represent typical frame durations for IEEE 802.15.4a.
    """

    # ── Supply ────────────────────────────────────────────────────────────
    voltage: float = 3.3                # Supply voltage (V)

    # ── UWB Radio Currents (mA) ──────────────────────────────────────────
    tx_current_mA: float = 70.0         # Transmit current
    rx_current_mA: float = 110.0        # Receive current
    idle_current_mA: float = 12.0       # Idle / listen current
    sleep_current_mA: float = 0.001     # Deep-sleep current

    # ── UWB Timing per Message (µs) ──────────────────────────────────────
    tx_duration_us: float = 200.0       # Duration of one TX frame
    rx_duration_us: float = 300.0       # Duration of one RX window
    processing_duration_us: float = 10.0  # MCU processing per message

    # ── Ranging Protocol ──────────────────────────────────────────────────
    ranging_mode: str = "SS-TWR"        # "SS-TWR" or "DS-TWR"
    uwb_frequency_hz: float = 10.0     # Ranging rate (rangings per second)
    num_anchors: int = 4               # Number of anchors being ranged

    # ── IMU ───────────────────────────────────────────────────────────────
    imu_enabled: bool = True
    uwb_disabled: bool = False          # E.g., for "IMU Only" algorithm
    imu_active_current_mA: float = 6.5  # Typical MEMS IMU active current
    imu_sleep_current_mA: float = 0.006
    imu_sample_rate_hz: float = 100.0   # IMU sampling rate

    # ── Battery ───────────────────────────────────────────────────────────
    battery_capacity_mAh: float = 225.0  # Typical coin-cell / small LiPo

    device_name: str = "Custom / DW1000 Default"

    def apply_hardware_profile(self, profile_name: str):
        """Update current consumption values based on a predefined hardware profile."""
        from src.core.uwb.hardware_profiles import HardwareProfileManager
        profile = HardwareProfileManager.get_profile(profile_name)
        if profile:
            self.device_name = profile.name
            self.tx_current_mA = profile.tx_current_mA
            self.rx_current_mA = profile.rx_current_mA
            self.idle_current_mA = profile.idle_current_mA

    def get_ranging_mode(self) -> RangingMode:
        """Convert the string ranging_mode to the RangingMode enum."""
        for mode in RangingMode:
            if mode.value == self.ranging_mode:
                return mode
        return RangingMode.SS_TWR


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class EnergyResult:
    """
    Complete energy estimation output.

    All energy values are in micro-Joules (µJ).
    All power values are in milli-Watts (mW).
    All current values are in milli-Amperes (mA).
    """

    # ── Per-message ───────────────────────────────────────────────────────
    energy_per_tx_message_uJ: float = 0.0
    energy_per_rx_message_uJ: float = 0.0

    # ── Per-ranging exchange ──────────────────────────────────────────────
    energy_per_ranging_uJ: float = 0.0
    messages_per_ranging: int = 0
    tx_messages_per_ranging: int = 0
    rx_messages_per_ranging: int = 0

    # ── Continuous power breakdown ────────────────────────────────────────
    uwb_active_power_mW: float = 0.0   # Average UWB active power (TX+RX)
    tag_idle_power_mW: float = 0.0     # Idle / sleep power
    imu_power_mW: float = 0.0          # IMU contribution

    # ── Totals ────────────────────────────────────────────────────────────
    total_power_mW: float = 0.0        # Sum of all contributors
    total_current_mA: float = 0.0      # total_power_mW / voltage
    total_energy_consumed_J: float = 0.0 # Cumulative energy consumed over simulation

    # ── Battery ───────────────────────────────────────────────────────────
    battery_life_hours: float = 0.0
    battery_life_days: float = 0.0

    # ── Duty cycle ────────────────────────────────────────────────────────
    duty_cycle_percent: float = 0.0    # Fraction of time the radio is active

    # ── Protocol info ─────────────────────────────────────────────────────
    ranging_mode: str = ""
    uwb_frequency_hz: float = 0.0
    num_anchors: int = 0
    device_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a flat dictionary, ready for JSON / API."""
        return {
            "energy_per_tx_message_uJ": round(self.energy_per_tx_message_uJ, 4),
            "energy_per_rx_message_uJ": round(self.energy_per_rx_message_uJ, 4),
            "energy_per_ranging_uJ": round(self.energy_per_ranging_uJ, 4),
            "messages_per_ranging": self.messages_per_ranging,
            "tx_messages_per_ranging": self.tx_messages_per_ranging,
            "rx_messages_per_ranging": self.rx_messages_per_ranging,
            "uwb_active_power_mW": round(self.uwb_active_power_mW, 4),
            "tag_idle_power_mW": round(self.tag_idle_power_mW, 4),
            "imu_power_mW": round(self.imu_power_mW, 4),
            "total_power_mW": round(self.total_power_mW, 4),
            "total_current_mA": round(self.total_current_mA, 4),
            "total_energy_consumed_J": round(self.total_energy_consumed_J, 6),
            "battery_life_hours": round(self.battery_life_hours, 2),
            "battery_life_days": round(self.battery_life_days, 2),
            "duty_cycle_percent": round(self.duty_cycle_percent, 4),
            "ranging_mode": self.ranging_mode,
            "uwb_frequency_hz": self.uwb_frequency_hz,
            "num_anchors": self.num_anchors,
            "device_name": self.device_name,
        }


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class EnergyCalculator:
    """
    Computes the energy / power consumption of a UWB tag.

    The model is *duty-cycle based*:
        1.  Compute energy per single TX / RX message.
        2.  Sum per-ranging exchange based on protocol message count.
        3.  Scale by frequency and number of anchors to get average power.
        4.  Add idle and (optionally) IMU contributions.
        5.  Derive battery life from total current draw.
    """

    def __init__(self, config: EnergyConfig | None = None):
        self.config = config or EnergyConfig()
        
        # Cumulative tracking state
        self.cumulative_energy_uJ: float = 0.0
        self.total_simulation_time_s: float = 0.0
        self.step_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────

    def reset_accumulator(self):
        """Reset the cumulative energy tracking state."""
        self.cumulative_energy_uJ = 0.0
        self.total_simulation_time_s = 0.0
        self.step_count = 0
        
    def calculate_step(self, dt: float) -> EnergyResult:
        """
        Calculate energy consumption for a single simulation timestep dt,
        accumulate it, and return the instantaneous EnergyResult.
        """
        result = self.calculate()
        
        # Energy = Power * Time (uJ = mW * ms)
        # Power is in mW, dt is in s. Therefore dt * 1000 is in ms.
        # So mW * (dt * 1000) = uJ
        step_energy_uJ = result.total_power_mW * (dt * 1000.0)
        
        self.cumulative_energy_uJ += step_energy_uJ
        self.total_simulation_time_s += dt
        self.step_count += 1
        
        # Update result with the cumulative state
        result.total_energy_consumed_J = self.cumulative_energy_uJ * 1e-6
        return result

    def calculate(self) -> EnergyResult:
        """Run full energy estimation and return an EnergyResult."""
        cfg = self.config
        mode = cfg.get_ranging_mode()

        # --- Per-message energy (µJ) ---
        # E = V × I × t   (V in V, I in mA → mW, t in µs → µJ)
        e_tx = cfg.voltage * cfg.tx_current_mA * cfg.tx_duration_us * 1e-6  # µJ
        e_rx = cfg.voltage * cfg.rx_current_mA * cfg.rx_duration_us * 1e-6  # µJ
        # Convert from mW·s to µJ:  1 mW·µs = 1e-6 mW·s = 1e-3 µJ  … actually
        # V * mA = mW;  mW * µs = µW·s = nJ? Let me be precise.
        # V(V) * I(mA) = P(mW);  P(mW) * t(µs) = E(mW·µs) = E(nJ) = E(µJ)*1e-3
        # So: E(µJ) = V * I_mA * t_us * 1e-3
        e_tx = cfg.voltage * cfg.tx_current_mA * cfg.tx_duration_us * 1e-3  # µJ
        e_rx = cfg.voltage * cfg.rx_current_mA * cfg.rx_duration_us * 1e-3  # µJ

        # --- Messages per ranging ---
        tx_msgs, rx_msgs = _PROTOCOL_MSG_COUNT.get(mode, (1, 1))
        total_msgs = tx_msgs + rx_msgs

        # --- Energy per single ranging exchange (one anchor) ---
        e_ranging = e_tx * tx_msgs + e_rx * rx_msgs  # µJ

        # --- Active time per ranging (seconds) ---
        t_active_per_ranging_s = (
            (cfg.tx_duration_us * tx_msgs +
             cfg.rx_duration_us * rx_msgs +
             cfg.processing_duration_us * total_msgs) * 1e-6
        )

        # --- Duty cycle ---
        # Total active time per second = per-ranging active time × frequency × anchors
        t_active_per_second = t_active_per_ranging_s * cfg.uwb_frequency_hz * cfg.num_anchors
        duty_cycle = min(t_active_per_second, 1.0)  # cap at 100 %

        uwb_active_power = 0.0
        if not cfg.uwb_disabled:
            # --- Average UWB active power (mW) ---
            #   = (energy per ranging per anchor, in µJ) × freq × anchors → µJ/s = µW → /1000 → mW
            uwb_active_power = (e_ranging * cfg.uwb_frequency_hz * cfg.num_anchors) * 1e-3  # mW
        else:
            duty_cycle = 0.0

        # --- Idle / sleep power ---
        idle_fraction = 1.0 - duty_cycle
        tag_idle_power = cfg.voltage * cfg.idle_current_mA * idle_fraction  # mW

        # --- IMU power ---
        imu_power = 0.0
        if cfg.imu_enabled:
            # IMU is always on when enabled (continuous sampling)
            imu_power = cfg.voltage * cfg.imu_active_current_mA  # mW

        # --- Totals ---
        total_power = uwb_active_power + tag_idle_power + imu_power  # mW
        total_current = total_power / cfg.voltage if cfg.voltage > 0 else 0.0  # mA

        # --- Battery life ---
        if total_current > 0:
            battery_life_h = cfg.battery_capacity_mAh / total_current
        else:
            battery_life_h = float("inf")
        battery_life_d = battery_life_h / 24.0

        return EnergyResult(
            energy_per_tx_message_uJ=e_tx,
            energy_per_rx_message_uJ=e_rx,
            energy_per_ranging_uJ=e_ranging,
            messages_per_ranging=total_msgs,
            tx_messages_per_ranging=tx_msgs,
            rx_messages_per_ranging=rx_msgs,
            uwb_active_power_mW=uwb_active_power,
            tag_idle_power_mW=tag_idle_power,
            imu_power_mW=imu_power,
            total_power_mW=total_power,
            total_current_mA=total_current,
            total_energy_consumed_J=self.cumulative_energy_uJ * 1e-6,
            battery_life_hours=battery_life_h,
            battery_life_days=battery_life_d,
            duty_cycle_percent=duty_cycle * 100.0,
            ranging_mode=mode.value,
            uwb_frequency_hz=cfg.uwb_frequency_hz,
            num_anchors=cfg.num_anchors,
            device_name=cfg.device_name,
        )

    # ── Convenience setters ───────────────────────────────────────────────

    def set_ranging_mode(self, mode_str: str):
        """Set ranging mode from a string like 'SS-TWR' or 'DS-TWR'."""
        self.config.ranging_mode = mode_str

    def set_frequency(self, freq_hz: float):
        """Set UWB ranging frequency in Hz."""
        self.config.uwb_frequency_hz = max(0.1, freq_hz)

    def set_num_anchors(self, n: int):
        """Set number of anchors."""
        self.config.num_anchors = max(1, n)

    def set_imu_enabled(self, enabled: bool):
        """Enable or disable IMU power contribution."""
        self.config.imu_enabled = enabled

    def get_messages_per_ranging(self) -> int:
        """Return total message count for the current protocol."""
        mode = self.config.get_ranging_mode()
        tx, rx = _PROTOCOL_MSG_COUNT.get(mode, (1, 1))
        return tx + rx
