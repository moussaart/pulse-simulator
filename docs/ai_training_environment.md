# AI Training Environment

## Architecture Overview

The PULSE simulator provides a dedicated AI training environment with a clear
separation between the **main simulation** and **AI training logic**. The
architecture uses a TCP/IP socket for communication between the simulator and
external RL agents, plus a programmatic API for configuration and data retrieval.

```
┌─────────────────────────────────────────────────────────────────┐
│               PULSE Simulator (PyQt5)                           │
│                                                                 │
│  ┌──────────────────┐   ┌──────────────────────────────────┐    │
│  │  Main Simulation  │   │        AI Training Module         │    │
│  │  (Localization    │   │                                    │    │
│  │   Engine)         │   │  ┌──────────┐  ┌──────────────┐   │    │
│  │                   │──▶│  │AITraining│  │ AIGymServer   │   │    │
│  │  • Tag motion     │   │  │   API    │  │ (TCP:5555)    │   │    │
│  │  • Channel model  │   │  │ GET/SET  │  │               │   │    │
│  │  • Localization   │   │  └──────────┘  └───────┬───────┘   │    │
│  │  • Energy model   │   │                        │           │    │
│  └──────────────────┘   └────────────────────────┼───────────┘    │
│                                                   │               │
└───────────────────────────────────────────────────┼───────────────┘
                                                    │ TCP/IP
                                            ┌───────▼───────┐
                                            │  External RL   │
                                            │  Agent (PyTorch │
                                            │  / Stable       │
                                            │  Baselines)     │
                                            └────────────────┘
```

## API Reference

### `AITrainingAPI` — Configuration & Data Access

The primary facade for AI training. Imported from `src.api`:

```python
from src.api import AITrainingAPI

api = AITrainingAPI(sim_context=app)
```

> **Scope Isolation:** All SET operations modify a training-local configuration
> copy. The main simulation state is **never** affected.

---

#### GET Operations (Read Data)

| Method | Returns | Description |
|--------|---------|-------------|
| `get_num_anchors()` | `int` | Number of anchors in the environment |
| `get_nlos_solutions_count()` | `int` | Count of anchors in NLOS condition |
| `get_algorithm_info()` | `dict` | Algorithm type, noise, measurements, error |
| `get_energy_info()` | `dict` | Energy level, consumption, battery life |
| `get_measurements()` | `list[float]` | Latest distance measurements per anchor |
| `get_error()` | `float` | Latest localization error (meters) |
| `get_los_conditions()` | `list[bool]` | LOS/NLOS per anchor (True=LOS) |
| `get_registered_filters()` | `list[str]` | Available localization filters |
| `get_training_config()` | `dict` | Complete training configuration |
| `get_full_state()` | `dict` | Full API state snapshot |

**Example — Algorithm Info:**

```python
info = api.get_algorithm_info()
# {
#     "algorithm_type": "Extended Kalman Filter",
#     "measurement_noise": [0.05, 0.03, 0.08, 0.04],
#     "measurements": [3.21, 5.67, 2.89, 4.12],
#     "error": 0.042,
#     "filter_params": {}
# }
```

**Example — Energy Info:**

```python
energy = api.get_energy_info()
# {
#     "energy_level": 1200.5,         # hours
#     "energy_consumption": {...},     # breakdown
#     "config": {...},                 # energy config
#     "accumulated": {
#         "total_power_mW": 12.5,
#         "battery_life_days": 50.2,
#         "total_energy_consumed_J": 0.0034
#     }
# }
```

---

#### SET Operations (Modify Training Configuration)

| Method | Parameters | Description |
|--------|------------|-------------|
| `set_num_anchors(n)` | `n: int` | Set anchor count (≥ 1) |
| `set_num_stacks(n)` | `n: int` | Set ranging stack count (≥ 1) |
| `set_input_mode(mode)` | `mode: str` | `"imu"`, `"uwb"`, or `"both"` |
| `set_filter(name, **params)` | `name: str` | Select localization filter + params |
| `set_energy_profile(**kwargs)` | kwargs | Configure energy profile |

**Example — Configure Training:**

```python
api.set_num_anchors(6)
api.set_input_mode("both")        # Fused IMU + UWB
api.set_filter("NLOS-Aware AEKF", alpha=0.3, beta=2.0)
api.set_energy_profile(ranging_mode="DS-TWR", battery_capacity_mAh=500)
```

---

### `AIGymServer` — RL Communication

TCP server for OpenAI Gym-style interaction with external agents:

```python
from src.api.ai_gym_server import AIGymServer

server = AIGymServer(host="localhost", port=5555)
server.start()
```

**Protocol:**
- `send_state(state_dict)` — Send observation to agent
- `receive_action()` — Receive action from agent
- `send_reward(reward, done)` — Send reward and episode status

---

### `TrainingDataAPI` — Offline Data Collection

For batch data collection and export (dataset creation):

```python
from src.api import TrainingDataAPI

api = TrainingDataAPI(buffer_size=10000)
api.select_data(channel=True, filter_outputs=True, ground_truth=True)
api.enable_collection()
# ... run simulation ...
api.export_to_file("training_data.npz")
```

**Supported export formats:** JSON, CSV, NPZ, PyTorch Dataset

---

## Module Structure

```
src/api/
├── __init__.py                 # Exports: AITrainingAPI, TrainingDataAPI
├── ai_training_facade.py       # [NEW] Scoped GET/SET facade
├── ai_gym_server.py            # TCP server for RL agents
├── training_api.py             # Offline data collection facade
├── adapters/
│   ├── channel_adapter.py      # UWB channel data extraction
│   ├── filter_adapter.py       # Localization filter data extraction
│   ├── geometry_adapter.py     # Geometry/position data extraction
│   └── energy_adapter.py       # Energy consumption data extraction
├── collectors/
│   └── data_collector.py       # DataSample, DataBuffer, DataCollector
└── export/
    └── data_exporter.py        # Multi-format export (JSON/CSV/NPZ/PyTorch)
```

### Removed APIs
- ~~`ai_streamer.py`~~ — Unused TCP streamer (superseded by `AIGymServer`)
- ~~`DataExporter.to_tensorflow_tfrecord()`~~ — TensorFlow not in dependencies
- ~~`TrainingDataAPI.export_for_tensorflow()`~~ — Wrapper for removed exporter
