"""
Typed dataclass for parsing the enriched JSON state from the PULSE simulator.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class IMUData:
    """IMU sensor readings from the tag."""
    acceleration: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    enabled: bool = False


@dataclass
class NLOSInfo:
    """Non-Line-of-Sight diagnostics."""
    is_los: List[bool] = field(default_factory=list)
    nlos_count: int = 0
    nlos_anchor_ids: List[str] = field(default_factory=list)


@dataclass
class AlgorithmInfo:
    """Localization algorithm metadata."""
    name: str = "Unknown"
    available_algorithms: List[str] = field(default_factory=list)
    filter_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrecisionInfo:
    """Localization accuracy metrics."""
    localization_error: float = 0.0
    prev_localization_error: float = 0.0
    gdop: Optional[float] = None
    measurement_noise_stds: List[float] = field(default_factory=list)


@dataclass
class EnergyInfo:
    """Energy consumption data."""
    total_power_mW: float = 0.0
    step_energy_uJ: float = 0.0
    cumulative_energy_J: float = 0.0
    battery_life_hours: float = 0.0
    duty_cycle_percent: float = 0.0
    ranging_mode: str = "SS-TWR"
    uwb_active_power_mW: float = 0.0
    imu_power_mW: float = 0.0


@dataclass
class MeasurementData:
    """Distance measurements from the current step."""
    source: str = "uwb"
    uwb_ranges: List[float] = field(default_factory=list)
    true_distances: List[float] = field(default_factory=list)


@dataclass
class EnvironmentConfig:
    """Simulation environment parameters."""
    dt: float = 0.005
    movement_speed: float = 1.0
    movement_pattern: str = "Circular"
    measurement_source: str = "uwb"


@dataclass
class PulseState:
    """
    Complete observation state from the PULSE simulator.

    This is the typed equivalent of the JSON state dict sent over TCP.
    Use ``PulseState.from_dict(json_data)`` to parse.
    """
    agent_id: int = 0
    step: int = 0
    timestamp: float = 0.0
    protocol_version: int = 1

    # Positions
    tag_position_gt: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    tag_position_estimated: Optional[List[float]] = None

    # Anchors
    anchor_positions: List[List[float]] = field(default_factory=list)
    anchor_ids: List[str] = field(default_factory=list)

    # Sub-structures
    measurements: MeasurementData = field(default_factory=MeasurementData)
    imu_data: IMUData = field(default_factory=IMUData)
    nlos_info: NLOSInfo = field(default_factory=NLOSInfo)
    algorithm: AlgorithmInfo = field(default_factory=AlgorithmInfo)
    precision: PrecisionInfo = field(default_factory=PrecisionInfo)
    energy: EnergyInfo = field(default_factory=EnergyInfo)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "PulseState":
        """Parse a JSON state dictionary into a PulseState instance."""
        return cls(
            agent_id=data.get("agent_id", 0),
            step=data.get("step", 0),
            timestamp=data.get("timestamp", 0.0),
            protocol_version=data.get("protocol_version", 1),
            tag_position_gt=data.get("tag_position_gt", [0.0, 0.0, 0.0]),
            tag_position_estimated=data.get("tag_position_estimated"),
            anchor_positions=data.get("anchor_positions", []),
            anchor_ids=data.get("anchor_ids", []),
            measurements=MeasurementData(**data["measurements"]) if "measurements" in data else MeasurementData(),
            imu_data=IMUData(**data["imu_data"]) if "imu_data" in data else IMUData(),
            nlos_info=NLOSInfo(**data["nlos_info"]) if "nlos_info" in data else NLOSInfo(),
            algorithm=AlgorithmInfo(**data["algorithm"]) if "algorithm" in data else AlgorithmInfo(),
            precision=PrecisionInfo(**data["precision"]) if "precision" in data else PrecisionInfo(),
            energy=EnergyInfo(**data["energy"]) if "energy" in data else EnergyInfo(),
            environment=EnvironmentConfig(**data["environment"]) if "environment" in data else EnvironmentConfig(),
        )

    @property
    def num_anchors(self) -> int:
        return len(self.anchor_positions)

    @property
    def error(self) -> float:
        return self.precision.localization_error
