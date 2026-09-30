"""Qt-free training simulation that mirrors the main PULSE simulation loop.

The main window's ``SimulationManager`` is the behavioural reference. This module
duplicates its *sequencing* — clock, motion, LOS, per-link channel preparation,
CPU/GPU backend selection, IMU gating and single-pass energy accounting — while
owning its own per-agent state, so N agents train independently and the whole
engine can run in a process with no Qt event loop.

Deliberate differences from the old ``AITrainingWindow`` loop:

* the clock is advanced *before* motion, as ``SimulationManager`` does, instead of
  sampling the first position at ``t=0``;
* ``exact_mode`` is forwarded to ``MotionController``;
* ranging goes through the same per-anchor channel preparation and the same
  CPU/GPU batch selection as the main simulation, instead of a bare scalar call
  that skipped both;
* filter state is adopted from each algorithm's own output rather than guessed
  from a substring of its name;
* IMU is supplied according to the algorithm's ``uses_imu`` / ``required_sensors``
  declaration AND the tag's ``imu_data_on``, matching ``estimate_position``;
* energy is computed exactly once per step, from the anchors actually selected,
  and carried on the step record instead of being re-derived downstream;
* an unregistered algorithm name raises instead of silently selecting a default.
"""

from __future__ import annotations

import inspect
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.core.localization.Alghortimes_doc import Alghortimes_doc
from src.core.localization.base_algorithm import (
    AlgorithmInput,
    BaseLocalizationAlgorithm,
)
from src.core.motion import MotionController
from src.core.parallel.gpu_backend import gpu_manager
from src.core.uwb.energy_model import EnergyCalculator
from src.core.uwb.uwb_devices import Position, Tag

__all__ = [
    "AITrainingEngine",
    "MeasurementError",
    "MOVEMENT_PATTERNS",
    "TrainingAgentSim",
    "TrainingConfig",
    "UnknownAlgorithmError",
    "validate_movement_pattern",
]

#: Every pattern ``MotionController.update_tag_position`` implements, mirrored
#: from the main window's pattern combo box. Anything else silently leaves the tag
#: at its current position, which looks like a frozen simulation rather than an
#: error, so the engine rejects it up front instead.
MOVEMENT_PATTERNS = (
    "Circular",
    "Figure 8",
    "Square",
    "Random Walk",
    "Fixed Point",
    "Foot Mounted",
)


class UnknownAlgorithmError(LookupError):
    """Raised when a requested algorithm name is not in the registry.

    The previous behaviour was a silent fallback to a default algorithm, so the
    RL agent could believe it was training a duty-cycled filter while a different
    one actually ran.
    """


class UnknownMovementPatternError(LookupError):
    """Raised when a movement pattern ``MotionController`` does not implement.

    ``update_tag_position`` has no ``else`` branch: an unrecognised pattern leaves
    the tag exactly where it was. Training then runs for hours producing a flat
    trajectory that looks like a physics bug rather than a typo.
    """


def validate_movement_pattern(pattern: str) -> str:
    """Return ``pattern`` if ``MotionController`` can execute it, else raise.

    ``Custom:<trajectory>`` is forwarded by name, so only the prefix is checked.
    """
    name = str(pattern)
    if name.startswith("Custom:"):
        if not name[len("Custom:"):].strip():
            raise UnknownMovementPatternError("Custom movement pattern needs a trajectory name")
        return name
    if name not in MOVEMENT_PATTERNS:
        raise UnknownMovementPatternError(
            f"unknown movement pattern {name!r}; "
            f"expected one of {', '.join(MOVEMENT_PATTERNS)} or 'Custom:<trajectory>'"
        )
    return name


def _method_uses_imu(method: Any) -> bool:
    """Mirror ``SimulationManager._method_uses_imu``."""
    if method is None:
        return False
    if inspect.isclass(method) and issubclass(method, BaseLocalizationAlgorithm):
        if bool(getattr(method, "uses_imu", False)):
            return True
        sensors = getattr(method, "required_sensors", ()) or ()
        if isinstance(sensors, str):
            sensors = (sensors,)
        return any(str(sensor).strip().lower() == "imu" for sensor in sensors)
    return False


@dataclass
class TrainingConfig:
    """Immutable snapshot of everything the engine needs, JSON-serialisable."""

    algorithm: str = "Duty-Cycled UWB-IMU NA-AEKF"
    dt: float = 0.005
    movement_pattern: str = "Random Walk"
    movement_speed: float = 1.0
    measurement_source: str = "both"
    num_agents: int = 1
    start_position: Tuple[float, float] = (0.0, 0.0)
    imu_data_on: bool = True
    exact_trajectory_mode: bool = False
    metrics_window: int = 2000
    allow_gpu: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_movement_pattern(self.movement_pattern)
        if float(self.dt) <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt!r}")
        if int(self.num_agents) < 1:
            raise ValueError(f"num_agents must be >= 1, got {self.num_agents!r}")
        if int(self.metrics_window) < 1:
            raise ValueError(f"metrics_window must be >= 1, got {self.metrics_window!r}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "dt": self.dt,
            "movement_pattern": self.movement_pattern,
            "movement_speed": self.movement_speed,
            "measurement_source": self.measurement_source,
            "num_agents": self.num_agents,
            "start_position": list(self.start_position),
            "imu_data_on": self.imu_data_on,
            "exact_trajectory_mode": self.exact_trajectory_mode,
            "metrics_window": self.metrics_window,
            "allow_gpu": self.allow_gpu,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TrainingConfig":
        data = dict(data or {})
        start = list(data.get("start_position", (0.0, 0.0)))
        return cls(
            algorithm=str(data.get("algorithm", "Duty-Cycled UWB-IMU NA-AEKF")),
            dt=float(data.get("dt", 0.005)),
            movement_pattern=str(data.get("movement_pattern", "Random Walk")),
            movement_speed=float(data.get("movement_speed", 1.0)),
            measurement_source=str(data.get("measurement_source", "both")),
            num_agents=int(data.get("num_agents", 1)),
            start_position=(float(start[0]), float(start[1])),
            imu_data_on=bool(data.get("imu_data_on", True)),
            exact_trajectory_mode=bool(data.get("exact_trajectory_mode", False)),
            metrics_window=int(data.get("metrics_window", 2000)),
            allow_gpu=bool(data.get("allow_gpu", True)),
            extra=dict(data.get("extra", {}) or {}),
        )


class MeasurementError(RuntimeError):
    """Raised when ranging keeps failing, mirroring SimulationManager."""


class TrainingAgentSim:
    """One agent's independent simulation state."""

    def __init__(self, agent_id: int, config: TrainingConfig, anchors: Sequence[Any],
                 channel: Any = None):
        self.agent_id = int(agent_id)
        self.config = config
        self.anchors = list(anchors)
        self.exact_mode = bool(config.exact_trajectory_mode)

        if channel is None:
            from src.core.uwb.channel_model import ChannelConditions
            channel = ChannelConditions()
        # Each agent owns its channel: a shared one would let one agent's
        # per-link channel preparation leak into the next agent's ranging.
        self.channel = channel

        self.tag = Tag(Position(float(config.start_position[0]), float(config.start_position[1])))
        self.tag.imu_data_on = bool(config.imu_data_on)
        self.sim_time = 0.0

        # Filter state starts as None on purpose: every algorithm under
        # src/user_algorithms self-initialises when `initialized` is False or
        # state/covariance is None. That removes the previous name-based
        # dimension guessing, which mis-sized some 8-state filters.
        self.state: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None
        self.previous_state: Optional[np.ndarray] = None
        self.previous_covariance: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.initialized = False

        self.energy = EnergyCalculator()
        self.extra_data: Dict[str, Any] = {}
        self.curr_error = 0.0
        self.prev_error = 0.0
        self.last_source = config.measurement_source
        self.last_action: Dict[str, Any] = {}
        self.algorithm_instance: Optional[Any] = None
        self.algorithm_name: Optional[str] = None
        self.consecutive_failures = 0

    def reset(self) -> None:
        from src.core.uwb.channel_model import ChannelConditions
        self.channel = ChannelConditions()
        self.tag = Tag(Position(*self.config.start_position))
        self.tag.imu_data_on = bool(self.config.imu_data_on)
        self.sim_time = 0.0
        self.state = None
        self.covariance = None
        self.previous_state = None
        self.previous_covariance = None
        self.Q = None
        self.R = None
        self.initialized = False
        self.energy = EnergyCalculator()
        self.extra_data = {}
        self.curr_error = 0.0
        self.prev_error = 0.0
        self.last_source = self.config.measurement_source
        self.last_action = {}
        self.algorithm_instance = None
        self.algorithm_name = None
        self.consecutive_failures = 0

    def advance_motion(self) -> None:
        """Main-loop order: bump the clock, then move; IMU is refreshed inside."""
        self.sim_time += self.config.dt
        MotionController.update_tag_position(
            tag=self.tag,
            movement_pattern=self.config.movement_pattern,
            movement_speed=self.config.movement_speed,
            t=self.sim_time,
            frequence=1.0 / self.config.dt if self.config.dt > 0 else 200.0,
            point=(0.0, 0.0),
            exact_mode=self.exact_mode,
            dt=self.config.dt,
        )


class AITrainingEngine:
    """Multi-agent training simulation mirroring ``SimulationManager``."""

    def __init__(self, config: TrainingConfig, anchors: Sequence[Any], channel_conditions: Any = None):
        self.config = config
        self.anchors = list(anchors)
        if not self.anchors:
            raise ValueError("training requires at least one anchor")

        self.algorithm_methods = Alghortimes_doc().get_algorithm_methods()
        self._lookup = {str(name).strip().lower(): name for name in self.algorithm_methods}

        # Each agent gets its own channel model so the N agents stay independent;
        # `channel_conditions` is only the factory seed for agent 0, kept for
        # callers that want to share one explicitly.
        seed = channel_conditions
        self.agents = [
            TrainingAgentSim(i, config, self.anchors,
                             channel=(seed if (i == 0 and seed is not None) else None))
            for i in range(max(1, int(config.num_agents)))
        ]
        self._MAX_CONSECUTIVE_FAILURES = 10
        self.step_index = 0
        self._anchor_positions_array = np.array(
            [[a.position.x, a.position.y] for a in self.anchors], dtype=float
        )
        self.recent_metrics: deque = deque(maxlen=max(1, int(config.metrics_window)))
        self._observation_hook = None
        self.last_backend = "cpu"

    @property
    def channel_conditions(self) -> Any:
        """Agent 0's channel model, for callers that inspect shared state."""
        return self.agents[0].channel

    @property
    def dt(self) -> float:
        return self.config.dt

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    # ── Algorithm registry ──────────────────────────────────────────────────

    def registered_algorithms(self) -> List[str]:
        return sorted(self.algorithm_methods)

    def resolve_algorithm(self, name: Optional[str]) -> str:
        """Exact registry resolution; never silently substitutes a default."""
        if name is None or not str(name).strip():
            name = self.config.algorithm
        key = str(name).strip().lower()
        if key not in self._lookup:
            raise UnknownAlgorithmError(
                f"unknown algorithm {name!r}; registered: {', '.join(self.registered_algorithms())}"
            )
        return self._lookup[key]

    def set_observation_hook(self, hook) -> None:
        """``hook(agent_id, observation_dict) -> dict``, for façade enrichment."""
        self._observation_hook = hook

    def reset(self) -> None:
        for agent in self.agents:
            agent.reset()
        self.step_index = 0
        self.recent_metrics.clear()

    # ── LOS + ranging, mirroring SimulationManager ──────────────────────────

    def _frame_los(self, agent: "TrainingAgentSim") -> List[bool]:
        """Reproduce ``SimulationManager._compute_frame_los``.

        That method refreshes moving NLOS zones against the wall clock, then uses
        ``batch_los_check_gpu`` on the GPU path and ``check_los_condition`` per
        anchor otherwise. Calling ``channel.batch_update_los_conditions`` instead
        would be a different geometry kernel, and skipping the zone refresh would
        let a zone drift out of sync with where it is actually drawn.
        """
        channel = agent.channel
        if not self.anchors:
            return []

        for zone in getattr(channel, "moving_nlos_zones", ()) or ():
            try:
                zone.update_position(time.time())
            except Exception:
                pass

        tag_pos = agent.tag.position
        if self.config.allow_gpu and gpu_manager.available and len(self.anchors) >= 2:
            try:
                from src.core.parallel.geometry_kernels import batch_los_check_gpu

                array = batch_los_check_gpu(
                    anchor_positions=self._anchor_positions_array,
                    tag_position=(tag_pos.x, tag_pos.y),
                    nlos_zones=channel.nlos_zones,
                    moving_nlos_zones=channel.moving_nlos_zones,
                )
                los = [bool(v) for v in np.asarray(array).tolist()]
                if len(los) == len(self.anchors):
                    return los
            except Exception:
                pass

        return [bool(channel.check_los_condition(a.position, tag_pos)) for a in self.anchors]

    def _prepare_channel_for_anchor(self, channel, anchor, tag_position, is_los: bool) -> None:
        """Reproduce ``SimulationManager._measure_single_anchor`` channel setup."""
        if is_los:
            channel.is_los = True
            channel.current_path_loss_params = channel.los_path_loss_params
            channel.current_sv_params = channel.los_sv_params
            channel.current_noise_factor = 1.0
            channel.current_error_bias = 0.0
        else:
            channel.update_los_condition(anchor.position, tag_position)
        channel._current_anchor_pos = anchor.position

    def _measure_sequential(self, agent: "TrainingAgentSim", los: List[bool]) -> List[Optional[Dict[str, Any]]]:
        channel = agent.channel
        records: List[Optional[Dict[str, Any]]] = []
        for i, anchor in enumerate(self.anchors):
            is_los = bool(los[i])
            self._prepare_channel_for_anchor(channel, anchor, agent.tag.position, is_los)
            try:
                distance, _messages, _ranging = agent.tag.measure_distance_with_logs(
                    anchor, channel, agent.sim_time, "SS-TWR"
                )
                distance = float(distance)
            except Exception:
                records.append(None)
                continue
            # Main returns None (rather than raising) for an unusable distance and
            # lets get_measurements() apply its failure accounting.
            if not np.isfinite(distance) or distance < 0:
                records.append(None)
                continue
            records.append({
                "anchor_id": anchor.id,
                "distance": distance,
                "is_los": is_los,
                "true_distance": float(anchor.position.distance_to(agent.tag.position)),
            })
        self.last_backend = "cpu"
        return records

    def _measure_batch(self, agent: "TrainingAgentSim", los: List[bool]) -> List[Optional[Dict[str, Any]]]:
        channel = agent.channel
        anchors = self.anchors
        is_los_array = np.array(los, dtype=bool)
        channel.update_los_condition(anchors[0].position, agent.tag.position)
        tag_xy = np.array([agent.tag.position.x, agent.tag.position.y])
        true_distances = np.sqrt(np.sum((self._anchor_positions_array - tag_xy) ** 2, axis=1))
        ranging_results = channel.measure_distance_batch(
            true_distances, is_los_array, anchor_positions=[a.position for a in anchors]
        )
        records: List[Optional[Dict[str, Any]]] = []
        for i, (anchor, rr) in enumerate(zip(anchors, ranging_results)):
            distance = float(rr.measured_distance)
            if not np.isfinite(distance) or distance < 0:
                records.append(None)
                continue
            records.append({
                "anchor_id": anchor.id,
                "distance": distance,
                "is_los": bool(is_los_array[i]),
                "true_distance": float(true_distances[i]),
            })
        self.last_backend = "gpu"
        return records

    def measure_all(self, agent: "TrainingAgentSim") -> List[Optional[Dict[str, Any]]]:
        """Both main backends, with the same fallback rule."""
        los = self._frame_los(agent)
        if self.config.allow_gpu and gpu_manager.available and len(self.anchors) >= 2:
            try:
                return self._measure_batch(agent, los)
            except Exception:
                self.last_backend = "cpu-fallback"
        return self._measure_sequential(agent, los)

    def _account_measurement(self, agent: "TrainingAgentSim", records: List[Optional[Dict[str, Any]]]
                             ) -> bool:
        """Mirror ``get_measurements`` failure accounting; False means unusable."""
        if any(record is None for record in records):
            agent.consecutive_failures += 1
            if agent.consecutive_failures >= self._MAX_CONSECUTIVE_FAILURES:
                raise MeasurementError(
                    f"Measurements for agent {agent.agent_id} have failed "
                    f"{agent.consecutive_failures} times in a row."
                )
            return False
        agent.consecutive_failures = 0
        return True

    # ── Localization ────────────────────────────────────────────────────────

    def _instance_for(self, agent: "TrainingAgentSim", algo_name: str):
        method = self.algorithm_methods.get(algo_name)
        if method is None:
            raise UnknownAlgorithmError(f"unknown algorithm {algo_name!r}")
        if inspect.isclass(method) and issubclass(method, BaseLocalizationAlgorithm):
            if agent.algorithm_instance is None or agent.algorithm_name != algo_name:
                instance = method()
                instance.initialize()
                agent.algorithm_instance = instance
                agent.algorithm_name = algo_name
            return method, agent.algorithm_instance
        return method, None

    def _localize(self, agent: "TrainingAgentSim", algo_name: str, action: Dict[str, Any],
                  records: List[Dict[str, Any]]) -> Tuple[Tuple[float, float], Dict[str, Any]]:
        method, instance = self._instance_for(agent, algo_name)
        index = {a.id: a for a in self.anchors}
        if action["_anchor_ids"]:
            chosen = [r for r in records if r["anchor_id"] in action["_anchor_ids"]]
        else:
            chosen = list(records)
        anchors = [index[r["anchor_id"]] for r in chosen]
        measurements = [r["distance"] for r in chosen]
        is_los_bits = [0 if r["is_los"] else 1 for r in chosen]

        # IMU gating identical to estimate_position: declared by the algorithm
        # AND enabled on the tag, or forced by an explicit "IMU only" request.
        source = action["measurement_source"].lower()
        provide_imu = _method_uses_imu(method) and bool(agent.tag.imu_data_on)
        if source == "imu":
            provide_imu = True
        elif source == "uwb":
            provide_imu = False

        accel = gyro = None
        control = np.zeros(2)
        if provide_imu and hasattr(agent.tag, "imu_data") and len(agent.tag.imu_data) > 0:
            imu = agent.tag.imu_data
            accel = np.array([float(imu.acc_x[-1]), float(imu.acc_y[-1]), float(imu.acc_z[-1])])
            gyro = np.array([float(imu.gyro_x[-1]), float(imu.gyro_y[-1]), float(imu.gyro_z[-1])])
            control = np.array([accel[0], accel[1]])

        summary = {
            "algorithm": algo_name,
            "num_measurements": len(measurements),
            "provide_imu": bool(provide_imu),
            "is_los": is_los_bits,
        }

        if instance is not None:
            duty = action.get("duty_cycle")
            if duty is not None and hasattr(instance, "set_duty_cycle"):
                instance.set_duty_cycle(
                    cycle_length=duty["cycle_length"], active_window=duty["active_window"]
                )
            output = instance.update(AlgorithmInput(
                measurements=measurements,
                anchors=anchors,
                tag=agent.tag,
                dt=self.config.dt,
                state=agent.state,
                previous_state=agent.previous_state,
                covariance=agent.covariance,
                previous_covariance=agent.previous_covariance,
                Q=agent.Q,
                R=agent.R,
                initialized=agent.initialized,
                imu_data_on=bool(provide_imu),
                accel=accel,
                gyro=gyro,
                control_input=control,
                is_los=is_los_bits,
                params={
                    "movement_speed": self.config.movement_speed,
                    "movement_pattern": self.config.movement_pattern,
                    "dt": self.config.dt,
                },
            ))
            # Adopt the algorithm's real shapes; never guess them.
            agent.previous_state = agent.state
            agent.previous_covariance = agent.covariance
            agent.state = np.asarray(output.state, dtype=float).ravel()
            agent.covariance = np.asarray(output.covariance, dtype=float)
            agent.initialized = bool(output.initialized)
            agent.Q = output.Q
            agent.R = output.R
            if getattr(output, "extra_data", None):
                agent.extra_data = dict(output.extra_data)
            return (float(output.position[0]), float(output.position[1])), summary

        # Legacy static-function algorithms.
        if len(measurements) < 3:
            return (float(agent.tag.position.x), float(agent.tag.position.y)), summary
        result = method(measurements, anchors)
        return (float(result[0]), float(result[1])), summary

    # ── Energy, recorded exactly once per step ──────────────────────────────

    def _account_energy(self, agent: "TrainingAgentSim", action: Dict[str, Any], step_id: int
                        ) -> Dict[str, Any]:
        source = action["measurement_source"].lower()
        calc = agent.energy
        calc.set_num_anchors(len(action["_anchor_ids"]))
        calc.set_imu_enabled(source in ("imu", "both"))
        if "uwb_window_open" in agent.extra_data:
            calc.set_uwb_enabled(bool(agent.extra_data["uwb_window_open"]))
        else:
            calc.set_uwb_enabled(source in ("uwb", "both"))
        result = calc.calculate_step(self.config.dt)
        step_uJ = float(calc.cumulative_energy_uJ)
        return {
            "source_step_id": step_id,
            "num_anchors": len(action["_anchor_ids"]),
            "total_power_mW": float(result.total_power_mW),
            "step_energy_uJ": float(result.total_power_mW * self.config.dt * 1000.0),
            "cumulative_energy_uJ": step_uJ,
            "cumulative_energy_J": step_uJ * 1e-6,
            "uwb_window_open": bool(agent.extra_data.get("uwb_window_open", source in ("uwb", "both"))),
        }

    # ── Action parsing ──────────────────────────────────────────────────────

    def _normalise_action(self, action: Any) -> Dict[str, Any]:
        n = len(self.anchors)
        if action is None:
            action = {}
        if isinstance(action, (list, tuple)):
            action = {"anchors": list(action)}
        if not isinstance(action, dict):
            action = {"anchors": [action]}

        algo_name = self.resolve_algorithm(action.get("filter"))
        source = str(action.get("measurement_source") or self.config.measurement_source)

        raw = action.get("anchors", [])
        if raw is None:
            raw = []
        elif not isinstance(raw, (list, tuple)):
            raw = [raw]
        indices: List[int] = []
        for value in raw:
            try:
                idx = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < n:
                indices.append(idx)

        duty = None
        if action.get("t_imu") is not None or action.get("t_uwb") is not None:
            t_imu = float(action.get("t_imu") or 0.0)
            t_uwb = float(action.get("t_uwb") or 0.0)
            if t_imu < 0 or t_uwb < 0:
                raise ValueError(f"invalid duty cycle t_imu={t_imu} t_uwb={t_uwb}")
            duty = {
                "t_imu": t_imu, "t_uwb": t_uwb,
                "cycle_length": t_imu + t_uwb, "active_window": t_uwb,
            }

        return {
            "filter": algo_name,
            "measurement_source": source,
            "anchors": indices,
            "_anchor_ids": [self.anchors[i].id for i in indices],
            "duty_cycle": duty,
        }

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self, actions: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
        """Advance every agent by one step and return a single-step report."""
        step_id = self.step_index
        normalised = [
            self._normalise_action(actions[i] if actions and i < len(actions) else None)
            for i in range(len(self.agents))
        ]

        per_agent: List[Dict[str, Any]] = []
        for i, agent in enumerate(self.agents):
            action = normalised[i]

            # 1. clock, 2. motion (+ IMU), 3. LOS, 4. ranging
            agent.advance_motion()
            raw_records = self.measure_all(agent)

            if not self._account_measurement(agent, raw_records):
                # Same as the main loop: an unusable frame advances the clock and
                # motion but skips estimation, energy and error entirely, so no
                # half-formed observation is ever published.
                per_agent.append({
                    "agent_id": agent.agent_id,
                    "step_id": step_id,
                    "valid": False,
                    "observation": None,
                    "measurements": [r for r in raw_records if r is not None],
                    "consecutive_failures": agent.consecutive_failures,
                    "error": agent.curr_error,
                    "estimate": None,
                    "true_position": (float(agent.tag.position.x), float(agent.tag.position.y)),
                    "energy": None,
                    "metrics": None,
                    "algorithm_input_summary": None,
                    "duty_cycle": action["duty_cycle"],
                })
                continue

            records = [r for r in raw_records if r is not None]

            # 5. localization with this step's own measurements
            estimate, summary = self._localize(agent, action["filter"], action, records)

            true_pos = (float(agent.tag.position.x), float(agent.tag.position.y))
            error = float(np.sqrt((estimate[0] - true_pos[0]) ** 2 + (estimate[1] - true_pos[1]) ** 2))
            if not np.isfinite(error):
                error = 0.0
            agent.prev_error = agent.curr_error
            agent.curr_error = error
            agent.last_source = action["measurement_source"]
            agent.last_action = {k: v for k, v in action.items() if not k.startswith("_")}

            # 6. energy, recorded exactly once and carried on the step record
            energy = self._account_energy(agent, action, step_id)

            observation = {
                "agent_id": agent.agent_id,
                "step_id": step_id,
                "timestamp": agent.sim_time,
                "tag_position_gt": [true_pos[0], true_pos[1]],
                "tag_position_estimated": [estimate[0], estimate[1]],
                "anchor_positions": [[a.position.x, a.position.y, a.position.z] for a in self.anchors],
                "anchor_ids": [a.id for a in self.anchors],
                "measurements": {r["anchor_id"]: r["distance"] for r in records},
                "true_distances": {r["anchor_id"]: r["true_distance"] for r in records},
                "los": {r["anchor_id"]: bool(r["is_los"]) for r in records},
                "selected_anchors": list(action["anchors"]),
                "algorithm": action["filter"],
                "measurement_source": action["measurement_source"],
                "precision": {
                    "localization_error": error,
                    "prev_localization_error": agent.prev_error,
                },
                "energy": {
                    "total_power_mW": energy["total_power_mW"],
                    "step_energy_uJ": energy["step_energy_uJ"],
                    "cumulative_energy_uJ": energy["cumulative_energy_uJ"],
                    "step_id": step_id,
                },
                "duty_cycle": action["duty_cycle"],
                "environment": {
                    "dt": self.config.dt,
                    "movement_speed": self.config.movement_speed,
                    "movement_pattern": self.config.movement_pattern,
                },
            }
            if self._observation_hook is not None:
                observation = self._observation_hook(agent.agent_id, observation) or observation

            metrics = {
                "step": step_id,
                "time": float(agent.sim_time),
                "agent_id": agent.agent_id,
                "error": error,
                "step_energy_uJ": energy["step_energy_uJ"],
                "cumulative_energy_uJ": energy["cumulative_energy_uJ"],
                "total_power_mW": energy["total_power_mW"],
                "provide_imu": summary["provide_imu"],
                "backend": self.last_backend,
            }
            self.recent_metrics.append(metrics)

            per_agent.append({
                "agent_id": agent.agent_id,
                "step_id": step_id,
                "valid": True,
                "observation": observation,
                "measurements": records,
                "estimate": estimate,
                "true_position": true_pos,
                "error": error,
                "energy": energy,
                "metrics": metrics,
                "algorithm_input_summary": summary,
                "duty_cycle": action["duty_cycle"],
            })

        self.step_index = step_id + 1
        return {
            "step_id": step_id,
            "agents": per_agent,
            "observations": [a["observation"] for a in per_agent if a.get("observation") is not None],
            "reward": {"step_id": step_id, "num_agents": len(per_agent)},
        }
