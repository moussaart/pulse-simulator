"""Headless training worker: owns the simulation, publishes metrics, serves states.

The GUI used to drive training through a ``QTimer``, so closing or freezing the
window stopped the run and a blocked event loop stalled the simulation. This
module inverts that: it owns the engine and writes metrics, and the GUI is only a
viewer. It therefore imports no Qt and runs in a bare interpreter, which is what
"Charts only" and "Headless" modes need.

Run it directly::

    python -m src.api.ai_training_worker --config worker.json --http-port 8765
    PULSE.exe --ai-training-worker --config worker.json

Configuration is validated at construction, not on the first step: losing a
24-hour run to a mistyped algorithm name at step 900000 is not acceptable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from src.api.ai_training_engine import AITrainingEngine, TrainingConfig
from src.api.ai_gym_server import (
    AIGymServer,
    REASON_ACTION,
    REASON_DISCONNECTED,
    REASON_RESET,
    REASON_STOPPED,
)
from src.api.frame_publisher import FramePublisher
from src.api.training_metrics import MetricsHttpServer, MetricsRecorder
from src.core.uwb.uwb_devices import Anchor, Position

__all__ = [
    "TrainingWorker",
    "WORKER_FLAG",
    "default_anchor_layout",
    "is_worker_invocation",
    "load_worker_config",
    "main",
    "strip_worker_flag",
]

PROTOCOL_VERSION = 3

#: ``PULSE.exe --ai-training-worker ...`` runs this instead of the Qt GUI. The
#: same flag works for ``python -m src.api.ai_training_worker`` and for the frozen
#: build, so a supervised worker can be launched identically in development and in
#: production.
WORKER_FLAG = "--ai-training-worker"


def is_worker_invocation(argv: Optional[Sequence[str]] = None) -> bool:
    """True when this process was asked to be the headless worker."""
    args = list(sys.argv[1:] if argv is None else argv)
    return WORKER_FLAG in args


def strip_worker_flag(argv: Optional[Sequence[str]] = None) -> List[str]:
    """Drop :data:`WORKER_FLAG` so argparse never sees it.

    An exact match only: ``--ai-training-worker=1`` is a different, unknown flag
    and must not be silently treated as worker mode.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    return [a for a in args if a != WORKER_FLAG]


def _jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays into plain JSON types.

    The engine returns numpy floats and arrays; they reach the socket, the
    JSONL sink and the HTTP dashboard, none of which can serialise them.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist())
        except Exception:
            pass
    return str(value)


def default_anchor_layout(count: int) -> List[Anchor]:
    """A rectangle of anchors, used when no GUI layout is supplied.

    Headless runs have nothing to drag anchors with, so the default has to be a
    geometry that actually localises: four corners of a square.
    """
    count = int(count)
    if count < 1:
        raise ValueError(f"need at least one anchor, got {count}")
    if count == 1:
        return [Anchor(Position(0.0, 0.0, 0.0))]
    if count == 2:
        coords = [(0.0, 0.0), (5.0, 0.0)]
    elif count == 3:
        coords = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)]
    elif count == 4:
        coords = [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)]
    else:
        side = int(math.ceil(math.sqrt(count)))
        coords = [
            (5.0 * (i % side), 5.0 * (i // side))
            for i in range(count)
        ]
    return [Anchor(Position(float(x), float(y), 0.0)) for x, y in coords]


def load_worker_config(path: Optional[str]) -> TrainingConfig:
    """Load a JSON config, falling back to defaults when it is absent.

    A missing file is not an error: Headless mode is expected to run with no
    config at all. A *malformed* one is, because it usually means a typo in
    something the operator meant to change.
    """
    if not path:
        return TrainingConfig()
    if not os.path.exists(path):
        print(f"[worker] config {path!r} not found; using defaults")
        return TrainingConfig()
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path!r} must contain a JSON object, got {type(raw).__name__}")
    known = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
    unknown = sorted(set(raw) - known)
    if unknown:
        # Silently ignoring typos here is how "why is my dt not changing?" happens.
        print(f"[worker] ignoring unknown config keys: {', '.join(unknown)}")
    return TrainingConfig.from_dict({k: v for k, v in raw.items() if k in known})


class TrainingWorker:
    """Owns one :class:`AITrainingEngine` and publishes its progress."""

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        config_path: Optional[str] = None,
        anchors: Optional[Sequence[Any]] = None,
        num_anchors: int = 4,
        run_dir: Optional[str] = None,
        recorder: Optional[MetricsRecorder] = None,
        serve_http: bool = False,
        http_port: int = 8765,
        serve_frames: bool = False,
        frames_port: int = 0,
        serve_gym: bool = False,
        gym_host: str = "0.0.0.0",
        gym_port: int = 5555,
    ) -> None:
        self.config = config if config is not None else load_worker_config(config_path)
        # Building the engine validates the algorithm, movement pattern and dt,
        # so a bad config fails here rather than hours into a run.
        self.engine = AITrainingEngine(
            self.config,
            anchors=list(anchors) if anchors else default_anchor_layout(num_anchors),
        )

        self.run_dir = run_dir or self._default_run_dir()
        self.recorder = recorder or MetricsRecorder(self.run_dir)
        # port=0 means "bind an ephemeral port", which is different from "off".
        # The control channel is how a supervising GUI drives this worker, so it
        # is part of the same server rather than a second socket to keep in sync.
        self.http = (
            MetricsHttpServer(
                self.recorder, port=http_port, control_handler=self.handle_control
            )
            if serve_http
            else None
        )
        self.frames = FramePublisher(port=frames_port) if serve_frames else None
        self.gym = (
            AIGymServer(
                host=gym_host,
                port=gym_port,
                num_agents=self.config.num_agents,
                algorithms=self.engine.registered_algorithms(),
                on_reset=self.reset,
            )
            if serve_gym
            else None
        )
        self._last_frame: Optional[Dict[str, Any]] = None

        self._phase = "running"
        self._step = 0
        self._pending_reward: Optional[Dict[str, float]] = None
        self._pending_actions: List[Dict[str, Any]] = []
        self._started_wall = time.time()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self.recorder.set_phase("running")

    @staticmethod
    def _default_run_dir() -> str:
        root = os.environ.get("PULSE_TRAINING_RUN_DIR") or os.path.join("runs", "ai-training")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return os.path.join(root, stamp)

    # ── lifecycle ──────────────────────────────────────────────────────────

    @property
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "phase": self._phase,
                "step": self._step,
                "num_agents": self.config.num_agents,
                "algorithm": self.config.algorithm,
                "movement_pattern": self.config.movement_pattern,
                "dt": self.config.dt,
                "num_anchors": len(self.engine.anchors),
                "run_dir": self.run_dir,
                "metrics_path": self.recorder.metrics_path,
                "gym_port": self.gym.port if self.gym is not None else None,
                "uptime_s": time.time() - self._started_wall,
            }

    def pause(self) -> Dict[str, Any]:
        with self._lock:
            if self._phase != "stopped":
                self._phase = "paused"
                self.recorder.set_phase("paused")
                self._wake_event.set()
        return self.status

    def resume(self) -> Dict[str, Any]:
        with self._lock:
            if self._phase == "stopped":
                return self.status
            self._phase = "running"
            self.recorder.set_phase("running")
            self._wake_event.set()
        return self.status

    def reset(self) -> Dict[str, Any]:
        """Restart the run's simulation state without discarding the log."""
        with self._lock:
            self.engine.reset()
            self._step = 0
            self._pending_reward = None
            self._pending_actions.clear()
            self._wake_event.set()
            if self._phase == "stopped":
                self._phase = "running"
            self.recorder.mark_reset()
            self.recorder.set_phase(self._phase)
        return self.status

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        self._wake_event.set()
        # A blocked RL-action wait must be released before taking the worker
        # lifecycle lock; otherwise an HTTP stop can deadlock behind tick().
        if self.gym is not None:
            self.gym.stop()
        with self._lock:
            self._phase = "stopped"
            self.recorder.set_phase("stopped")
        return self.status

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    # ── supervision ────────────────────────────────────────────────────────

    #: Maps the control channel's allowlist onto the lifecycle methods.
    _CONTROL_COMMANDS = {
        "pause": "pause",
        "resume": "resume",
        "reset": "reset",
        "stop": "stop",
    }

    def start_http(self) -> Optional[int]:
        """Bind the dashboard/control server and return its port, or None.

        Idempotent, and called by :func:`main`. A supervising GUI constructs the
        worker itself, so the worker must own starting its own server rather than
        leaving that to the CLI.
        """
        if self.http is None:
            return None
        if getattr(self, "_http_started", False):
            return self.http.port
        port = self.http.start()
        self._http_started = True
        return port

    def start_frames(self) -> Optional[int]:
        """Start the passive frame stream, if enabled, and return its port."""
        if self.frames is None:
            return None
        if getattr(self, "_frames_started", False):
            return self.frames.port
        port = self.frames.start()
        self._frames_started = True
        return port

    def start_services(self) -> Dict[str, Any]:
        """Start the worker's control/metrics and viewer-frame endpoints."""
        services = {
            "http_port": self.start_http(),
            "frames_port": self.start_frames(),
        }
        if self.gym is not None:
            if not self.gym.start():
                raise RuntimeError("could not start RL gym server: %r" % self.gym.last_error)
            services["gym_port"] = self.gym.port
        else:
            services["gym_port"] = None
        return services

    def handle_control(self, command: str) -> Dict[str, Any]:
        """Run one supervision command; returns the resulting status.

        The HTTP layer already rejects names outside the allowlist, so an
        unknown command here is a programming error rather than bad input.
        """
        attr = self._CONTROL_COMMANDS.get(command)
        if attr is None:
            raise ValueError("unknown control command: %r" % (command,))
        return getattr(self, attr)()

    # ── client hand-off ────────────────────────────────────────────────────

    def set_reward(self, step_reward: Optional[float] = None,
                   cumulative_reward: Optional[float] = None) -> None:
        """Record the reward the RL agent computed for the state it last saw."""
        if step_reward is None and cumulative_reward is None:
            return
        self._pending_reward = {
            "reward_step": None if step_reward is None else float(step_reward),
            "reward_cumulative": None if cumulative_reward is None else float(cumulative_reward),
        }

    def submit_action(self, agent_id: int, action: Dict[str, Any]) -> None:
        with self._lock:
            self._pending_actions.append({"agent_id": int(agent_id), "action": dict(action)})

    def _take_actions(self, num_agents: int) -> List[Dict[str, Any]]:
        """Consume queued actions, falling back to a full-anchor default.

        Training must not stall when the agent is slow or absent, so a missing
        action degrades to the previous behaviour rather than blocking the step.
        """
        with self._lock:
            queued = list(self._pending_actions)
            self._pending_actions.clear()

        by_id = {item["agent_id"]: item["action"] for item in queued}
        anchor_ids = list(range(len(self.engine.anchors)))
        actions = []
        for agent in self.engine.agents[:num_agents]:
            action = by_id.get(agent.agent_id)
            if action is None:
                # No invented duty cycle: the engine builds that from t_imu/t_uwb,
                # and a scalar duty_cycle is silently ignored there.
                action = {
                    "filter": self.config.algorithm,
                    "anchors": anchor_ids,
                    "measurement_source": self.config.measurement_source,
                }
            actions.append(action)
        return actions

    # ── stepping ───────────────────────────────────────────────────────────

    def tick(self) -> Dict[str, Any]:
        """Advance the simulation by one step and publish the resulting state."""
        with self._lock:
            if self._phase in ("stopped", "paused"):
                # Report the current frame without advancing anything.
                frame = self._state_frame([], self._step, [])
                return frame

            actions = self._take_actions(self.config.num_agents)
            report = self.engine.step(actions)
            step_id = report["step_id"]
            self._step = step_id + 1

            reward = self._pending_reward or {}
            self._pending_reward = None

            for index, per_agent in enumerate(report["agents"]):
                metrics = per_agent.get("metrics") or {}
                energy = per_agent.get("energy") or {}
                sim_time = metrics.get("time")
                if sim_time is None and index < len(self.engine.agents):
                    sim_time = getattr(self.engine.agents[index], "sim_time", 0.0)
                reward_step = reward.get("reward_step")
                reward_cumulative = reward.get("reward_cumulative")
                if isinstance(reward_step, (list, tuple)):
                    reward_step = reward_step[index] if index < len(reward_step) else None
                if isinstance(reward_cumulative, (list, tuple)):
                    reward_cumulative = (
                        reward_cumulative[index]
                        if index < len(reward_cumulative)
                        else None
                    )
                self.recorder.record({
                    "step": step_id,
                    "time": float(sim_time or 0.0),
                    "agent_id": per_agent.get("agent_id"),
                    "valid": bool(per_agent.get("valid", False)),
                    "error": per_agent.get("error"),
                    "step_energy_uJ": energy.get("step_energy_uJ"),
                    "cumulative_energy_uJ": energy.get("cumulative_energy_uJ"),
                    "total_power_mW": energy.get("total_power_mW"),
                    "backend": metrics.get("backend"),
                    "provide_imu": metrics.get("provide_imu"),
                    "reward_step": reward_step,
                    "reward_cumulative": reward_cumulative,
                })

            frame = self._state_frame(report["agents"], step_id, actions)
            self._last_frame = frame
            if self.frames is not None:
                self.frames.publish(frame)
            return frame

    def _gym_state_list(self, frame: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert engine observations to the external client's typed schema."""
        states = []
        for index, agent_record in enumerate(frame.get("agents", [])):
            obs = agent_record.get("observation") or {}
            if not obs:
                true_position = list(agent_record.get("true_position", [0.0, 0.0]))
                while len(true_position) < 3:
                    true_position.append(0.0)
                states.append({
                    "agent_id": agent_record.get("agent_id", index),
                    "step": frame.get("step_id", 0),
                    "step_id": frame.get("step_id", 0),
                    "protocol_version": PROTOCOL_VERSION,
                    "valid": False,
                    "tag_position_gt": true_position,
                    "tag_position_estimated": None,
                })
                continue

            agent = self.engine.agents[index]
            imu = getattr(agent.tag, "imu_data", None)
            def _last(values):
                try:
                    return float(values[-1])
                except (TypeError, ValueError, IndexError, KeyError):
                    return 0.0

            anchor_ids = list(obs.get("anchor_ids", []))
            raw_ranges = obs.get("measurements", {})
            true_ranges = obs.get("true_distances", {})
            los_by_id = obs.get("los", {})
            nlos_ids = [str(anchor_id) for anchor_id in anchor_ids
                        if not bool(los_by_id.get(anchor_id, True))]
            energy = obs.get("energy", {})
            tag_position_gt = list(obs.get("tag_position_gt", [0.0, 0.0]))
            while len(tag_position_gt) < 3:
                tag_position_gt.append(0.0)
            state = dict(obs)
            state.update({
                "step": int(frame.get("step_id", 0)),
                "step_id": int(frame.get("step_id", 0)),
                "protocol_version": PROTOCOL_VERSION,
                "valid": bool(agent_record.get("valid", False)),
                "tag_position_gt": tag_position_gt,
                "measurements": {
                    "source": obs.get("measurement_source", "uwb"),
                    "uwb_ranges": [float(raw_ranges[aid]) for aid in anchor_ids if aid in raw_ranges],
                    "true_distances": [float(true_ranges[aid]) for aid in anchor_ids if aid in true_ranges],
                },
                "imu_data": {
                    "acceleration": [_last(getattr(imu, key, [])) for key in ("acc_x", "acc_y", "acc_z")],
                    "angular_velocity": [_last(getattr(imu, key, [])) for key in ("gyro_x", "gyro_y", "gyro_z")],
                    "enabled": bool(getattr(agent.tag, "imu_data_on", False)),
                },
                "nlos_info": {
                    "is_los": [bool(los_by_id.get(aid, True)) for aid in anchor_ids],
                    "nlos_count": len(nlos_ids),
                    "nlos_anchor_ids": nlos_ids,
                },
                "algorithm": {"name": obs.get("algorithm", self.config.algorithm)},
                "energy": {
                    "total_power_mW": energy.get("total_power_mW", 0.0),
                    "step_energy_uJ": energy.get("step_energy_uJ", 0.0),
                    "cumulative_energy_J": float(energy.get("cumulative_energy_uJ", 0.0)) / 1e6,
                },
                "timestamp": obs.get("timestamp", frame.get("sim_time", 0.0)),
            })
            states.append(_jsonable(state))
        return states

    def _actions_from_gym(self, raw_action: Any) -> List[Dict[str, Any]]:
        """Adapt external anchor selections to one engine action per agent."""
        n_agents = self.config.num_agents
        if isinstance(raw_action, dict):
            values = [raw_action]
        elif isinstance(raw_action, (list, tuple)):
            values = list(raw_action)
            # A flat list of indices is one agent's anchor selection.
            if values and all(isinstance(item, (int, float)) for item in values):
                values = [values]
        else:
            values = []

        result = []
        default_anchors = list(range(len(self.engine.anchors)))
        for index in range(n_agents):
            value = values[index] if index < len(values) else None
            if isinstance(value, dict):
                action = dict(value)
                action.setdefault("filter", self.config.algorithm)
                action.setdefault("measurement_source", self.config.measurement_source)
                action.setdefault("anchors", default_anchors)
            elif isinstance(value, (list, tuple)):
                action = {
                    "filter": self.config.algorithm,
                    "measurement_source": self.config.measurement_source,
                    "anchors": list(value),
                }
            else:
                action = {
                    "filter": self.config.algorithm,
                    "measurement_source": self.config.measurement_source,
                    "anchors": default_anchors,
                }
            result.append(action)
        return result

    def _exchange_with_agent(self, frame: Dict[str, Any]) -> None:
        """Send a state and wait cancellably for the next external action."""
        if self.gym is None or not self.gym.connected:
            return
        if not self.gym.send_state(self._gym_state_list(frame)):
            return
        while not self.stopped and self.status["phase"] == "running" and self.gym.connected:
            response = self.gym.wait_for_action(timeout=0.25)
            if response is None:
                continue
            if response.reason == REASON_ACTION:
                actions = self._actions_from_gym(response.action)
                with self._lock:
                    for i, action in enumerate(actions):
                        self._pending_actions.append({"agent_id": i, "action": action})
                    metrics = response.metrics or {}
                    steps = metrics.get("step_rewards", [])
                    cumulative = metrics.get("cumulative_rewards", [])
                    if not isinstance(steps, (list, tuple)):
                        steps = [steps]
                    if not isinstance(cumulative, (list, tuple)):
                        cumulative = [cumulative]
                    self._pending_reward = {
                        "reward_step": list(steps),
                        "reward_cumulative": list(cumulative),
                    }
                return
            if response.reason in (REASON_DISCONNECTED, REASON_RESET, REASON_STOPPED):
                return

    def _state_frame(self, agents: List[Dict[str, Any]], step_id: int,
                     actions: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        return {
            "type": "state",
            "protocol_version": PROTOCOL_VERSION,
            "step_id": step_id,
            "phase": self._phase,
            "sim_time": float(getattr(self.engine.agents[0], "sim_time", 0.0)) if self.engine.agents else 0.0,
            "agents": _jsonable(agents),
            # Echo what drove this step, so a stalled or diverging run can be read
            # back from the log without guessing which action was applied.
            "actions": _jsonable(actions or []),
            "num_anchors": len(self.engine.anchors),
        }

    def run_forever(self, max_steps: Optional[int] = None,
                    on_frame=None) -> int:
        """Step until stopped. Returns the number of steps actually taken."""
        taken = 0
        while not self.stopped:
            if max_steps is not None and taken >= max_steps:
                break
            if self.status["phase"] == "paused":
                self._wake_event.wait(0.25)
                self._wake_event.clear()
                continue
            frame = self.tick()
            if frame.get("step_id") == self._step - 1:
                taken += 1
            if on_frame is not None:
                on_frame(frame)
            self._exchange_with_agent(frame)
        return taken

    def run_in_thread(self, max_steps: Optional[int] = None) -> threading.Thread:
        thread = threading.Thread(
            target=self.run_forever, args=(max_steps,), name="pulse-training-worker", daemon=True
        )
        thread.start()
        return thread

    def close(self) -> None:
        self.stop()
        if self.frames is not None:
            self.frames.stop()
        if self.gym is not None:
            self.gym.stop()
        if self.http is not None:
            self.http.stop()
        self.recorder.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PULSE --ai-training-worker",
        description="PULSE headless AI training worker",
    )
    parser.add_argument("--config", default=None, help="path to a JSON config")
    parser.add_argument("--port", type=int, default=0, help="dashboard HTTP port (0 disables)")
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--num-anchors", type=int, default=4)
    parser.add_argument("--algorithm", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--http", action="store_true", help="serve the browser dashboard")
    parser.add_argument("--frames", action="store_true", help="publish state frames to viewer subscribers")
    parser.add_argument("--frames-port", type=int, default=0, help="frame publisher port (0 chooses an ephemeral port)")
    parser.add_argument("--ready-file", default=None, help="write bound endpoint details to this JSON file")
    parser.add_argument("--gym", action="store_true", help="serve the external RL client TCP protocol")
    parser.add_argument("--gym-port", type=int, default=5555, help="RL client TCP port")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    config = load_worker_config(args.config)
    if args.algorithm:
        config.algorithm = args.algorithm
    if args.num_agents:
        config.num_agents = int(args.num_agents)

    http_port = args.port
    worker = TrainingWorker(
        config=config,
        num_anchors=args.num_anchors,
        run_dir=args.run_dir,
        serve_http=bool(args.http),
        http_port=args.port,
        serve_frames=bool(args.frames),
        frames_port=args.frames_port,
        serve_gym=bool(args.gym),
        gym_port=args.gym_port,
    )

    services = worker.start_services()
    if worker.http is not None:
        print(f"[worker] dashboard/control on http://127.0.0.1:{services['http_port']}")
    if worker.frames is not None:
        print(f"[worker] frames on tcp://127.0.0.1:{services['frames_port']}")
    if args.ready_file:
        ready_path = os.path.abspath(args.ready_file)
        os.makedirs(os.path.dirname(ready_path), exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "http_port": services["http_port"],
            "frames_port": services["frames_port"],
            "gym_port": services["gym_port"],
            "run_dir": worker.run_dir,
            "metrics_path": worker.recorder.metrics_path,
        }
        tmp_path = ready_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as ready_file:
            json.dump(payload, ready_file)
        os.replace(tmp_path, ready_path)
        print("[worker] ready " + json.dumps(payload), flush=True)
    print(f"[worker] metrics -> {worker.recorder.metrics_path}")

    def _handle_signal(signum, _frame):
        print(f"[worker] signal {signum}; stopping")
        worker.stop()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _handle_signal)
            except (ValueError, OSError):
                pass

    try:
        steps = worker.run_forever(max_steps=args.max_steps)
        print(f"[worker] completed {steps} steps")
    except KeyboardInterrupt:
        worker.stop()
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
