"""
AI Training API Facade

Provides a focused, scoped API for AI training with explicit GET/SET operations.
All modifications are isolated to the training environment and do NOT affect
the main simulation.

Usage:
    from src.api.ai_training_facade import AITrainingAPI

    api = AITrainingAPI(sim_context)

    # GET operations
    n = api.get_num_anchors()
    info = api.get_algorithm_info()
    energy = api.get_energy_info()

    # SET operations (training-scoped only)
    api.set_num_anchors(6)
    api.set_input_mode("uwb")
    api.set_filter("Extended Kalman Filter", alpha=0.3)
    api.set_energy_profile(ranging_mode="DS-TWR")
"""

from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
import numpy as np

from src.api.adapters.energy_adapter import EnergyDataAdapter
from src.api.adapters.filter_adapter import FilterDataAdapter
from src.api.adapters.channel_adapter import ChannelDataAdapter
from src.api.adapters.geometry_adapter import GeometryDataAdapter
from src.core.uwb.energy_model import EnergyCalculator, EnergyConfig


class TrainingConfig:
    """
    Local configuration for the AI training environment.
    Holds training-scoped parameters that do NOT propagate to the main simulation.
    """

    def __init__(self):
        self.num_anchors: int = 0
        self.num_stacks: int = 1
        self.input_mode: str = "uwb"  # "imu", "uwb", or "both"
        self.filter_name: str = "Extended Kalman Filter"
        self.filter_params: Dict[str, Any] = {}
        self.energy_profile: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_anchors": self.num_anchors,
            "num_stacks": self.num_stacks,
            "input_mode": self.input_mode,
            "filter_name": self.filter_name,
            "filter_params": dict(self.filter_params),
            "energy_profile": dict(self.energy_profile),
        }


class AITrainingAPI:
    """
    Focused API facade for AI training interactions.

    Provides explicit GET (read) and SET (configure) operations that are
    scoped exclusively to the AI training module. Modifications made through
    this API never affect the main simulation.

    Args:
        sim_context: Reference to the main application (LocalizationApp) for
                     read-only access to simulation state.
    """

    def __init__(self, sim_context):
        # Read-only reference to the main simulation
        self._sim = sim_context

        # Training-local configuration (isolated from main sim)
        self._config = TrainingConfig()

        # Adapters for data extraction
        self._energy_adapter = EnergyDataAdapter(EnergyCalculator())
        self._filter_adapter = FilterDataAdapter()
        self._channel_adapter = ChannelDataAdapter()
        self._geometry_adapter = GeometryDataAdapter()

        # Snapshot of latest training step results
        self._latest_measurements: List[float] = []
        self._latest_error: float = 0.0
        self._latest_los_conditions: List[bool] = []

        # Sync initial config from the simulation
        self._sync_from_sim()

    # ================================================================
    #                      INTERNAL HELPERS
    # ================================================================

    def _sync_from_sim(self) -> None:
        """Sync training config with current simulation state (one-time)."""
        if hasattr(self._sim, "anchors"):
            self._config.num_anchors = len(self._sim.anchors)
        if hasattr(self._sim, "algorithm"):
            self._config.filter_name = self._sim.algorithm

    def update_step_data(
        self,
        measurements: List[float],
        error: float,
        los_conditions: List[bool],
    ) -> None:
        """
        Called by the training loop to update latest step data.

        Args:
            measurements: Distance measurements from current step
            error: Localization error from current step
            los_conditions: LOS/NLOS conditions per anchor
        """
        self._latest_measurements = list(measurements)
        self._latest_error = error
        self._latest_los_conditions = list(los_conditions)

    # ================================================================
    #                      GET OPERATIONS (Read Data)
    # ================================================================

    def get_num_anchors(self) -> int:
        """
        Get the number of anchors in the training environment.

        Returns:
            Number of anchors currently configured
        """
        return self._config.num_anchors

    def get_nlos_solutions_count(self) -> int:
        """
        Get the count of anchors currently in NLOS condition.

        Returns:
            Number of anchors experiencing Non-Line-of-Sight
        """
        if not self._latest_los_conditions:
            # Compute from live simulation state
            if hasattr(self._sim, "anchors") and hasattr(self._sim, "channel_conditions"):
                nlos_count = 0
                for anchor in self._sim.anchors:
                    if hasattr(self._sim, "tag"):
                        is_los = self._sim.channel_conditions.check_los_to_anchor(
                            anchor.position, self._sim.tag.position
                        )
                        if not is_los:
                            nlos_count += 1
                return nlos_count
            return 0
        return sum(1 for los in self._latest_los_conditions if not los)

    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get algorithm-related information.

        Returns:
            Dictionary containing:
                - algorithm_type: Name of the current filter/algorithm
                - measurement_noise: Measurement noise standard deviations
                - measurements: Latest distance measurements
                - error: Latest localization error (ground truth vs predicted)
                - filter_params: Current filter configuration parameters
        """
        info: Dict[str, Any] = {
            "algorithm_type": self._config.filter_name,
            "measurement_noise": [],
            "measurements": list(self._latest_measurements),
            "error": self._latest_error,
            "filter_params": dict(self._config.filter_params),
        }

        # Extract measurement noise from channel adapter if available
        if (
            hasattr(self._sim, "anchors")
            and hasattr(self._sim, "channel_conditions")
            and hasattr(self._sim, "tag")
        ):
            noise_stds = []
            for anchor in self._sim.anchors:
                try:
                    link_data = self._channel_adapter.extract_link_data(
                        self._sim.channel_conditions, anchor, self._sim.tag
                    )
                    noise_stds.append(link_data.noise_std)
                except Exception:
                    noise_stds.append(0.0)
            info["measurement_noise"] = noise_stds

        return info

    def get_energy_info(self) -> Dict[str, Any]:
        """
        Get energy-related data.

        Returns:
            Dictionary containing:
                - energy_level: Current battery level estimate
                - energy_consumption: Power consumption breakdown
                - config: Current energy configuration
        """
        summary = self._energy_adapter.get_energy_summary()
        config = self._energy_adapter.get_config()

        # Also pull accumulated energy from the main sim's calculator if available
        accumulated = {}
        if hasattr(self._sim, "energy_calculator"):
            calc = self._sim.energy_calculator
            result = calc.calculate()
            accumulated = {
                "total_power_mW": result.total_power_mW,
                "battery_life_hours": result.battery_life_hours,
                "battery_life_days": result.battery_life_days,
                "total_energy_consumed_J": result.total_energy_consumed_J,
            }

        return {
            "energy_level": accumulated.get("battery_life_hours", summary.get("battery_life_hours", 0)),
            "energy_consumption": summary,
            "config": config,
            "accumulated": accumulated,
        }

    def get_measurements(self) -> List[float]:
        """
        Get the latest distance measurements from all anchors.

        Returns:
            List of measured distances (one per anchor)
        """
        return list(self._latest_measurements)

    def get_error(self) -> float:
        """
        Get the latest localization error.

        Returns:
            Euclidean distance between ground truth and estimated position (meters)
        """
        return self._latest_error

    def get_los_conditions(self) -> List[bool]:
        """
        Get the latest LOS/NLOS conditions for all anchors.

        Returns:
            List of booleans (True = LOS, False = NLOS) per anchor
        """
        return list(self._latest_los_conditions)

    def get_registered_filters(self) -> List[str]:
        """
        Get list of all registered localization filters.

        Returns:
            List of filter names available for selection
        """
        return self._filter_adapter.get_registered_filters()

    def get_training_config(self) -> Dict[str, Any]:
        """
        Get the complete training configuration.

        Returns:
            Dictionary with all training-local parameters
        """
        return self._config.to_dict()

    # ================================================================
    #                    SET OPERATIONS (Modify Config)
    # ================================================================

    def set_num_anchors(self, n: int) -> None:
        """
        Set the number of anchors for the training environment.

        This modifies the training-local configuration only.
        The main simulation's anchor count is NOT affected.

        Args:
            n: Number of anchors (must be >= 3 for localization)
        """
        if n < 1:
            raise ValueError("Number of anchors must be at least 1")
        self._config.num_anchors = n
        # Also update the training energy adapter
        self._energy_adapter.update_config(num_anchors=n)

    def set_num_stacks(self, n: int) -> None:
        """
        Set the number of ranging stacks used for localization.

        More stacks = better ranging accuracy but higher latency and energy use.

        Args:
            n: Number of stacks (>= 1)
        """
        if n < 1:
            raise ValueError("Number of stacks must be at least 1")
        self._config.num_stacks = n

    def set_input_mode(self, mode: str) -> None:
        """
        Set the localization input mode.

        Args:
            mode: One of:
                  - "imu"  : IMU-only navigation
                  - "uwb"  : UWB ranging only
                  - "both" : Fused IMU + UWB

        Raises:
            ValueError: If mode is not one of the allowed values
        """
        allowed = {"imu", "uwb", "both"}
        if mode.lower() not in allowed:
            raise ValueError(f"Input mode must be one of {allowed}, got '{mode}'")
        self._config.input_mode = mode.lower()

        # Update energy adapter to reflect IMU usage
        uses_imu = mode.lower() in ("imu", "both")
        uwb_disabled = mode.lower() == "imu"
        self._energy_adapter.update_config(
            imu_enabled=uses_imu,
        )

    def set_filter(self, name: str, **params) -> None:
        """
        Select and configure the localization filter for training.

        This does NOT change the filter used in the main simulation.

        Args:
            name: Filter name (e.g., "Extended Kalman Filter",
                  "Unscented Kalman Filter", "NLOS-Aware AEKF", etc.)
            **params: Filter-specific parameters (e.g., alpha=0.3, beta=2.0)

        Available filters can be listed with `get_registered_filters()`.
        """
        self._config.filter_name = name
        self._config.filter_params = dict(params)

    def set_energy_profile(self, **kwargs) -> None:
        """
        Configure energy profile settings for the training environment.

        Supported parameters:
            ranging_mode (str): "SS-TWR" or "DS-TWR"
            uwb_frequency_hz (float): UWB ranging frequency in Hz
            imu_enabled (bool): Whether IMU sensor is active
            imu_sample_rate_hz (float): IMU sampling rate
            battery_capacity_mAh (float): Battery capacity
            voltage (float): Operating voltage

        Args:
            **kwargs: Any EnergyConfig field
        """
        self._config.energy_profile.update(kwargs)
        self._energy_adapter.update_config(**kwargs)

    # ================================================================
    #                        SUMMARY / REPR
    # ================================================================

    def get_full_state(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of the training API state.
        Useful for debugging and serialization.

        Returns:
            Dictionary with training config, latest data, and energy info
        """
        return {
            "config": self.get_training_config(),
            "latest_data": {
                "measurements": self.get_measurements(),
                "error": self.get_error(),
                "los_conditions": self.get_los_conditions(),
                "nlos_count": self.get_nlos_solutions_count(),
            },
            "algorithm": self.get_algorithm_info(),
            "energy": self.get_energy_info(),
            "available_filters": self.get_registered_filters(),
        }

    def __repr__(self) -> str:
        return (
            f"AITrainingAPI("
            f"anchors={self._config.num_anchors}, "
            f"mode='{self._config.input_mode}', "
            f"filter='{self._config.filter_name}')"
        )
