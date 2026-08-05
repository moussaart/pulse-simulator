"""
UWB Tag Energy Consumption Model

Estimates the energy consumption of a UWB tag during ranging operations.
Supports SS-TWR and DS-TWR protocols, integrates IMU power consumption,
and provides battery life estimation.

Energy values are user-configurable or loaded from device_profiles.json.

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
    """

    # ── UWB Energy Per Operation (µJ) ────────────────────────────
    energy_tx_uJ: float = 46.2       # Energy per TX frame
    energy_rx_uJ: float = 108.9      # Energy per RX window
    power_idle_mW: float = 19.8      # Idle power between rangings (mW)
    power_sleep_mW: float = 0.00033  # Deep sleep power (mW)

    # ── Ranging Protocol ─────────────────────────────────────────
    ranging_mode: str = "SS-TWR"
    uwb_frequency_hz: float = 10.0
    num_anchors: int = 4

    # ── IMU Energy ───────────────────────────────────────────────
    imu_enabled: bool = True
    uwb_disabled: bool = False
    imu_energy_active_uJ_per_sample: float = 0.033
    imu_power_sleep_mW: float = 0.0198
    imu_sample_rate_hz: float = 100.0

    # ── Battery ──────────────────────────────────────────────────
    battery_capacity_mAh: float = 225.0
    voltage: float = 3.3  # Kept ONLY for battery life estimation (V = I * R etc., P = V * I)

    # ── Profile Names ────────────────────────────────────────────
    uwb_profile_name: str = "DW1000"
    imu_profile_name: str = "Generic MEMS IMU"

    def apply_uwb_profile(self, profile_name: str):
        """Update energy consumption values based on a predefined hardware profile."""
        from src.core.uwb.hardware_profiles import DeviceProfileManager
        profile = DeviceProfileManager.get_uwb_profile(profile_name)
        if profile:
            self.uwb_profile_name = profile.name
            self.energy_tx_uJ = profile.energy_tx_uJ
            self.energy_rx_uJ = profile.energy_rx_uJ
            self.power_idle_mW = profile.power_idle_mW
            self.power_sleep_mW = profile.power_sleep_mW

    def apply_imu_profile(self, profile_name: str):
        from src.core.uwb.hardware_profiles import DeviceProfileManager
        profile = DeviceProfileManager.get_imu_profile(profile_name)
        if profile:
            self.imu_profile_name = profile.name
            self.imu_energy_active_uJ_per_sample = profile.energy_active_uJ_per_sample
            self.imu_power_sleep_mW = profile.power_sleep_mW
            self.imu_sample_rate_hz = profile.sample_rate_hz

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
    tag_idle_power_mW: float = 0.0     # Idle / standby power between rangings
    tag_sleep_power_mW: float = 0.0    # Deep sleep power
    imu_power_mW: float = 0.0          # IMU contribution

    # ── Totals ────────────────────────────────────────────────────────────
    total_power_mW: float = 0.0        # Sum of all contributors
    total_current_mA: float = 0.0      # total_power_mW / voltage
    total_energy_consumed_J: float = 0.0 # Cumulative energy consumed over simulation

    # ── Averages ──────────────────────────────────────────────────────────
    average_power_mW: float = 0.0      # Average power over simulation
    average_current_mA: float = 0.0    # Average current over simulation

    # ── Battery ───────────────────────────────────────────────────────────
    battery_life_hours: float = 0.0
    battery_life_days: float = 0.0

    # ── Protocol info ─────────────────────────────────────────────────────
    ranging_mode: str = ""
    uwb_frequency_hz: float = 0.0
    num_anchors: int = 0
    device_name: str = ""
    imu_name: str = ""

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
            "average_power_mW": round(self.average_power_mW, 4),
            "average_current_mA": round(self.average_current_mA, 4),
            "total_energy_consumed_J": round(self.total_energy_consumed_J, 6),
            "battery_life_hours": round(self.battery_life_hours, 2),
            "battery_life_days": round(self.battery_life_days, 2),
            "ranging_mode": self.ranging_mode,
            "uwb_frequency_hz": self.uwb_frequency_hz,
            "num_anchors": self.num_anchors,
            "device_name": self.device_name,
            "imu_name": self.imu_name,
        }


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

class EnergyCalculator:
    """
    Computes the energy / power consumption of a UWB tag using energy values.
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
        
        if self.total_simulation_time_s > 0:
            result.average_power_mW = (self.cumulative_energy_uJ / 1000.0) / self.total_simulation_time_s
        else:
            result.average_power_mW = result.total_power_mW
            
        result.average_current_mA = result.average_power_mW / self.config.voltage if self.config.voltage > 0 else 0.0
        
        # Re-evaluate battery life using average current
        if result.average_current_mA > 0:
            result.battery_life_hours = self.config.battery_capacity_mAh / result.average_current_mA
        else:
            result.battery_life_hours = float("inf")
        result.battery_life_days = result.battery_life_hours / 24.0
        
        return result

    def calculate(self) -> EnergyResult:
        """Run full energy estimation and return an EnergyResult."""
        cfg = self.config
        mode = cfg.get_ranging_mode()

        # --- Per-message energy (µJ) ---
        e_tx = cfg.energy_tx_uJ
        e_rx = cfg.energy_rx_uJ

        # --- Messages per ranging ---
        tx_msgs, rx_msgs = _PROTOCOL_MSG_COUNT.get(mode, (1, 1))
        total_msgs = tx_msgs + rx_msgs

        # --- Energy per single ranging exchange (one anchor) ---
        e_ranging = (e_tx * tx_msgs + e_rx * rx_msgs)  # µJ

        uwb_active_power = 0.0
        if not cfg.uwb_disabled:
            # --- Average UWB active power (mW) ---
            #   = (energy per ranging per anchor, in µJ) × freq × anchors → µJ/s = µW → /1000 → mW
            uwb_active_power = (e_ranging * cfg.uwb_frequency_hz * cfg.num_anchors) * 1e-3  # mW


        # --- IMU power ---
        imu_power = 0.0
        if cfg.imu_enabled:
            # IMU is always on when enabled (continuous sampling)
            imu_power = (cfg.imu_energy_active_uJ_per_sample * cfg.imu_sample_rate_hz) * 1e-3  # mW
        else:
            imu_power = cfg.imu_power_sleep_mW  # mW
        # --- Idle power (between rangings, radio standby) ---
        tag_idle_power = cfg.power_idle_mW  # mW

        # --- Deep sleep power ---
        tag_sleep_power = cfg.power_sleep_mW  # mW

        # --- Totals ---
        total_power = uwb_active_power + tag_idle_power + tag_sleep_power + imu_power  # mW
        total_current = total_power / cfg.voltage if cfg.voltage > 0 else 0.0  # mA

        # --- Averages ---
        if self.total_simulation_time_s > 0:
            average_power_mW = (self.cumulative_energy_uJ / 1000.0) / self.total_simulation_time_s
        else:
            average_power_mW = total_power
            
        average_current_mA = average_power_mW / cfg.voltage if cfg.voltage > 0 else 0.0

        # --- Battery life (based on AVERAGE current) ---
        if average_current_mA > 0:
            battery_life_h = cfg.battery_capacity_mAh / average_current_mA
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
            tag_sleep_power_mW=tag_sleep_power,
            imu_power_mW=imu_power,
            total_power_mW=total_power,
            total_current_mA=total_current,
            average_power_mW=average_power_mW,
            average_current_mA=average_current_mA,
            total_energy_consumed_J=self.cumulative_energy_uJ * 1e-6,
            battery_life_hours=battery_life_h,
            battery_life_days=battery_life_d,
            ranging_mode=mode.value,
            uwb_frequency_hz=cfg.uwb_frequency_hz,
            num_anchors=cfg.num_anchors,
            device_name=cfg.uwb_profile_name,
            imu_name=cfg.imu_profile_name,
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
        self.config.num_anchors = max(0, n)

    def set_imu_enabled(self, enabled: bool):
        """Enable or disable IMU power contribution."""
        self.config.imu_enabled = enabled

    def set_uwb_enabled(self, enabled: bool):
        """Enable or disable UWB power contribution (for IMU-only mode)."""
        self.config.uwb_disabled = not enabled

    def get_messages_per_ranging(self) -> int:
        """Return total message count for the current protocol."""
        mode = self.config.get_ranging_mode()
        tx, rx = _PROTOCOL_MSG_COUNT.get(mode, (1, 1))
        return tx + rx
