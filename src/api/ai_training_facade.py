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

    # Build the complete observation for an RL agent
    obs = api.build_step_observation(
        agent_id=0, step=10, true_pos=[1.0, 2.0],
        est_pos=[1.1, 1.9], measurements={...}, ...
    )
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

        # Per-step energy calculator (accumulates over the training episode)
        self._energy_calculator = EnergyCalculator()

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

    def get_available_algorithms(self) -> List[str]:
        """
        Get the list of all available localization algorithms.

        Returns:
            Sorted list of algorithm display names
        """
        try:
            from src.core.localization.Alghortimes_doc import Alghortimes_doc
            doc = Alghortimes_doc()
            return sorted(doc.get_algorithm_methods().keys())
        except Exception:
            return list(self._filter_adapter.get_registered_filters())

    def get_measurement_source(self) -> str:
        """
        Get the current measurement source configuration.

        Returns:
            One of "uwb", "imu", or "both"
        """
        return self._config.input_mode

    def get_imu_data(self, tag) -> Dict[str, Any]:
        """
        Extract the latest IMU data from a tag object.

        Args:
            tag: Tag object that has imu_data attribute

        Returns:
            Dictionary with acceleration and angular_velocity arrays,
            plus an enabled flag.
        """
        imu_info: Dict[str, Any] = {
            "acceleration": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "enabled": False,
        }

        if tag is None:
            return imu_info

        imu_on = getattr(tag, "imu_data_on", False)
        imu_info["enabled"] = imu_on

        imu_data = getattr(tag, "imu_data", None)
        if imu_data is not None and len(imu_data) > 0:
            imu_info["acceleration"] = [
                float(imu_data.acc_x[-1]),
                float(imu_data.acc_y[-1]),
                float(imu_data.acc_z[-1]),
            ]
            imu_info["angular_velocity"] = [
                float(imu_data.gyro_x[-1]),
                float(imu_data.gyro_y[-1]),
                float(imu_data.gyro_z[-1]),
            ]

        return imu_info

    def get_gdop(self, tag, anchors) -> float:
        """
        Compute the Geometric Dilution of Precision (GDOP).

        Args:
            tag: Tag object
            anchors: List of anchor objects

        Returns:
            GDOP value (lower = better geometry)
        """
        return self._geometry_adapter.calculate_gdop(tag, anchors)

    def get_per_step_energy(self, dt: float) -> Dict[str, Any]:
        """
        Calculate and accumulate energy consumed for one simulation step.

        Args:
            dt: Time step duration (seconds)

        Returns:
            Dictionary with instantaneous and cumulative energy data
        """
        result = self._energy_calculator.calculate_step(dt)
        return {
            "total_power_mW": round(result.total_power_mW, 4),
            "step_energy_uJ": round(result.total_power_mW * dt * 1000.0, 4),
            "cumulative_energy_J": round(result.total_energy_consumed_J, 6),
            "battery_life_hours": round(result.battery_life_hours, 2),
            "duty_cycle_percent": round(result.duty_cycle_percent, 4),
            "ranging_mode": result.ranging_mode,
            "uwb_active_power_mW": round(result.uwb_active_power_mW, 4),
            "imu_power_mW": round(result.imu_power_mW, 4),
        }

    def reset_energy_accumulator(self) -> None:
        """Reset the per-episode energy accumulator."""
        self._energy_calculator.reset_accumulator()

    # ================================================================
    #          OBSERVATION BUILDER (for RL state dict)
    # ================================================================

    def build_step_observation(
        self,
        agent_id: int,
        step: int,
        dt: float,
        true_pos: List[float],
        est_pos: Optional[List[float]],
        tag,
        anchors: List,
        measurements: Dict[str, float],
        los_conditions: List[bool],
        curr_error: float,
        prev_error: float,
        algorithm_name: str,
        movement_speed: float,
        movement_pattern: str,
        channel_model=None,
        measurement_source: Optional[str] = None,
        active_anchor_count: Optional[int] = None,
        cumulative_energy_J: float = 0.0,
        algo_extra_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the complete enriched observation dictionary for one agent at
        one simulation step. This is the canonical state sent to the RL client.

        Args:
            agent_id: Index of this agent
            step: Current simulation step number
            dt: Simulation time step (seconds)
            true_pos: Ground truth position [x, y]
            est_pos: Estimated position [x, y] or None
            tag: Tag object (for IMU data access)
            anchors: List of Anchor objects
            measurements: Dict mapping anchor_id -> measured distance
            los_conditions: List of booleans per anchor
            curr_error: Current localization error
            prev_error: Previous localization error
            algorithm_name: Name of the active algorithm
            movement_speed: Tag movement speed
            movement_pattern: Movement pattern name
            channel_model: ChannelConditions object (optional, for noise data)

        Returns:
            Comprehensive state dictionary ready for JSON serialization
        """
        timestamp = step * dt

        # Anchor metadata
        anchor_positions = [[float(a.position.x), float(a.position.y)] for a in anchors]
        anchor_ids = [a.id for a in anchors]

        # UWB range measurements and true distances
        uwb_ranges = []
        true_distances = []
        for a in anchors:
            uwb_ranges.append(float(measurements.get(a.id, 0.0)))
            true_dist = np.linalg.norm([
                a.position.x - true_pos[0],
                a.position.y - true_pos[1]
            ])
            true_distances.append(float(true_dist))

        # IMU data from the tag
        imu_info = self.get_imu_data(tag)

        # NLOS information
        nlos_count = sum(1 for los in los_conditions if not los)
        nlos_anchor_ids = [
            anchor_ids[i] for i, los in enumerate(los_conditions)
            if i < len(anchor_ids) and not los
        ]

        # Measurement noise standard deviations
        noise_stds = []
        if channel_model is not None:
            for i, a in enumerate(anchors):
                try:
                    is_los = los_conditions[i] if i < len(los_conditions) else True
                    distance = true_distances[i] if i < len(true_distances) else 1.0
                    _, snr_linear = channel_model.get_received_signal_quality(distance)
                    # Use CRLB-based estimate
                    c = channel_model.c
                    bw = channel_model.uwb_params.bandwidth
                    if snr_linear > 0:
                        crlb = (c ** 2) / (8 * np.pi ** 2 * bw ** 2 * snr_linear)
                        noise_std = float(np.sqrt(crlb))
                    else:
                        noise_std = 1.0
                    noise_stds.append(noise_std)
                except Exception:
                    noise_stds.append(0.0)

        # GDOP computation
        try:
            gdop = self.get_gdop(tag, anchors)
        except Exception:
            gdop = float("inf")

        # Available algorithms
        available_algorithms = self.get_available_algorithms()

        # Reconfigure energy calculator for this specific step's source/anchors
        effective_source = measurement_source or self._config.input_mode
        uses_imu = effective_source.lower() in ("imu", "both")
        uses_uwb = effective_source.lower() in ("uwb", "both")
        self._energy_calculator.set_imu_enabled(uses_imu)
        self._energy_calculator.set_uwb_enabled(uses_uwb)
        
        # Use active_anchor_count if provided, otherwise default to len(anchors)
        num_anchors = active_anchor_count if active_anchor_count is not None else len(anchors)
        self._energy_calculator.set_num_anchors(num_anchors)

        # Energy data for this step — use calculate() for instantaneous snapshot
        # instead of calculate_step() which accumulates into a shared counter
        # and cross-contaminates multi-agent energy readings.
        _result = self._energy_calculator.calculate()
        energy_data = {
            "total_power_mW": round(_result.total_power_mW, 4),
            "step_energy_uJ": round(_result.total_power_mW * dt * 1000.0, 4),
            "cumulative_energy_J": round(cumulative_energy_J, 6),
            "battery_life_hours": round(_result.battery_life_hours, 2),
            "duty_cycle_percent": round(_result.duty_cycle_percent, 4),
            "ranging_mode": _result.ranging_mode,
            "uwb_active_power_mW": round(_result.uwb_active_power_mW, 4),
            "imu_power_mW": round(_result.imu_power_mW, 4),
        }

        # Duty-cycle metadata from the EKF algorithm's extra_data
        _extra = algo_extra_data or {}
        duty_cycle_info = {
            "cycle_length": _extra.get("cycle_length", 0.0),
            "active_window": _extra.get("active_window", 0.0),
            "t_imu": _extra.get("t_imu", 0.0),
            "t_uwb": _extra.get("t_uwb", 0.0),
            "uwb_window_open": _extra.get("uwb_window_open", False),
            "cycle_time": _extra.get("cycle_time", 0.0),
            "shadow_imu_position": _extra.get("shadow_imu_position", None),
            "shadow_imu_error": _extra.get("shadow_imu_error", None),
        }

        # Build the complete observation
        state_dict: Dict[str, Any] = {
            "agent_id": agent_id,
            "step": step,
            "timestamp": float(timestamp),

            # Positions
            "tag_position_gt": [float(true_pos[0]), float(true_pos[1]), 0.0],
            "tag_position_estimated": (
                [float(est_pos[0]), float(est_pos[1])] if est_pos is not None
                else None
            ),
            
            # Root-level legacy compatibility fields
            "localization_error": float(curr_error),
            "prev_localization_error": float(prev_error),

            # Anchors
            "anchor_positions": anchor_positions,
            "anchor_ids": anchor_ids,

            # Measurements
            "measurements": {
                "source": self._config.input_mode,
                "uwb_ranges": uwb_ranges,
                "true_distances": true_distances,
            },

            # IMU
            "imu_data": imu_info,

            # NLOS
            "nlos_info": {
                "is_los": los_conditions,
                "nlos_count": nlos_count,
                "nlos_anchor_ids": nlos_anchor_ids,
            },

            # Algorithm
            "algorithm": {
                "name": algorithm_name,
                "available_algorithms": available_algorithms,
                "filter_params": dict(self._config.filter_params),
            },

            # Precision
            "precision": {
                "localization_error": float(curr_error),
                "prev_localization_error": float(prev_error),
                "gdop": float(gdop) if np.isfinite(gdop) else None,
                "measurement_noise_stds": noise_stds,
            },

            # Energy
            "energy": energy_data,

            # Duty-cycle state (from EKF algorithm)
            "duty_cycle": duty_cycle_info,

            # Environment config
            "environment": {
                "dt": float(dt),
                "movement_speed": float(movement_speed),
                "movement_pattern": movement_pattern,
                "measurement_source": effective_source,
                "imu_period": float(dt),
                "uwb_period": float(1.0 / self._energy_calculator.config.uwb_frequency_hz) if self._energy_calculator.config.uwb_frequency_hz > 0 else 0.1,
            },
        }

        return state_dict

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
        self._energy_calculator.set_num_anchors(n)

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
        self._energy_adapter.update_config(
            imu_enabled=uses_imu,
        )
        self._energy_calculator.set_imu_enabled(uses_imu)

    def set_measurement_source(self, source: str) -> None:
        """
        Alias for set_input_mode — sets the measurement source.

        Args:
            source: One of "uwb", "imu", or "both"
        """
        self.set_input_mode(source)

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
        # Also update the per-step calculator
        cfg = self._energy_calculator.config
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

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
            "available_algorithms": self.get_available_algorithms(),
        }

    def __repr__(self) -> str:
        return (
            f"AITrainingAPI("
            f"anchors={self._config.num_anchors}, "
            f"mode='{self._config.input_mode}', "
            f"filter='{self._config.filter_name}')"
        )
