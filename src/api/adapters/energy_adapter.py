"""
Energy Data Adapter

Provides a clean API interface for accessing UWB tag energy consumption
data from the EnergyCalculator. Follows the same adapter pattern as
ChannelDataAdapter and FilterDataAdapter.
"""

from typing import Dict, Any
from src.core.uwb.energy_model import EnergyCalculator, EnergyConfig, EnergyResult


class EnergyDataAdapter:
    """
    API adapter wrapping an EnergyCalculator for external consumption.

    Usage:
        from src.api.adapters.energy_adapter import EnergyDataAdapter
        from src.core.uwb.energy_model import EnergyCalculator

        adapter = EnergyDataAdapter(EnergyCalculator())
        print(adapter.get_energy_summary())
    """

    def __init__(self, calculator: EnergyCalculator | None = None):
        self._calculator = calculator or EnergyCalculator()

    @property
    def calculator(self) -> EnergyCalculator:
        return self._calculator

    def get_energy_summary(self) -> Dict[str, Any]:
        """
        Compute and return all energy metrics as a flat dictionary.

        Returns:
            dict with keys like 'total_power_mW', 'battery_life_hours', etc.
        """
        result = self._calculator.calculate()
        return result.to_dict()

    def get_config(self) -> Dict[str, Any]:
        """
        Return the current configuration parameters as a dictionary.
        """
        cfg = self._calculator.config
        return {
            "voltage": cfg.voltage,
            "tx_current_mA": cfg.tx_current_mA,
            "rx_current_mA": cfg.rx_current_mA,
            "idle_current_mA": cfg.idle_current_mA,
            "sleep_current_mA": cfg.sleep_current_mA,
            "tx_duration_us": cfg.tx_duration_us,
            "rx_duration_us": cfg.rx_duration_us,
            "processing_duration_us": cfg.processing_duration_us,
            "ranging_mode": cfg.ranging_mode,
            "uwb_frequency_hz": cfg.uwb_frequency_hz,
            "num_anchors": cfg.num_anchors,
            "imu_enabled": cfg.imu_enabled,
            "imu_active_current_mA": cfg.imu_active_current_mA,
            "imu_sample_rate_hz": cfg.imu_sample_rate_hz,
            "battery_capacity_mAh": cfg.battery_capacity_mAh,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Combined output: config + computed energy metrics.
        Ready for JSON serialisation.
        """
        return {
            "config": self.get_config(),
            "results": self.get_energy_summary(),
        }

    def update_config(self, **kwargs):
        """
        Update configuration parameters programmatically.

        Args:
            **kwargs: Any EnergyConfig field, e.g. ranging_mode="DS-TWR",
                      uwb_frequency_hz=20.0, imu_enabled=False
        """
        cfg = self._calculator.config
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
