# PULSE AI Training Client (`pulse-ai-client`)

A pip-installable Python SDK for connecting Reinforcement Learning agents to the **PULSE UWB Simulator**.

## Quick Start

### 1. Install the package

```bash
cd external_ai_client
pip install -e .
```

### 2. Start the PULSE Simulator

1. Open the PULSE Simulator GUI.
2. Click **🤖 Start AI** to open the AI Training Window.
3. Set the **Port** (default: `5555`) and **Measurement Source** (UWB / IMU / Both).
4. Click **▶️ Start Training** — the simulator will begin stepping and waiting for your agent.

### 3. Run the training agent

```bash
python train_agent.py
```

## Enriched State (Protocol v2)

Each step, your agent receives a comprehensive observation containing:

| Field | Type | Description |
|-------|------|-------------|
| `tag_position_gt` | `[x, y, z]` | Ground truth tag position |
| `tag_position_estimated` | `[x, y]` | Estimated position from the localization algorithm |
| `measurements.uwb_ranges` | `float[]` | UWB distance measurements per anchor |
| `measurements.true_distances` | `float[]` | True distances per anchor |
| `measurements.source` | `str` | Active measurement source: `"uwb"`, `"imu"`, or `"both"` |
| `imu_data.acceleration` | `[ax, ay, az]` | Tag accelerometer readings (m/s²) |
| `imu_data.angular_velocity` | `[gx, gy, gz]` | Tag gyroscope readings (rad/s) |
| `nlos_info.is_los` | `bool[]` | LOS/NLOS condition per anchor |
| `nlos_info.nlos_count` | `int` | Number of NLOS anchors |
| `algorithm.name` | `str` | Active localization algorithm name |
| `algorithm.available_algorithms` | `str[]` | All available algorithms |
| `precision.localization_error` | `float` | Current error (meters) |
| `precision.gdop` | `float` | Geometric Dilution of Precision |
| `energy.step_energy_uJ` | `float` | Energy consumed this step (µJ) |
| `energy.cumulative_energy_J` | `float` | Total energy consumed (J) |
| `energy.battery_life_hours` | `float` | Estimated battery life |
| `energy.ranging_mode` | `str` | `"SS-TWR"` or `"DS-TWR"` |
| `environment.dt` | `float` | Simulation time step (seconds) |

## Usage Examples

### Gymnasium Environment

```python
from pulse_ai_client import PulseRLEnv

env = PulseRLEnv(port=5555, num_anchors=8)
obs, info = env.reset()

for step in range(1000):
    action = env.action_space.sample()  # Replace with your RL model
    obs, reward, terminated, truncated, info = env.step(action)

    # Access enriched state
    state = info["full_state"]
    print(f"Error: {state.error:.3f}m, NLOS: {state.nlos_info.nlos_count}")
    print(f"IMU: {state.imu_data.acceleration}")
    print(f"Energy: {state.energy.step_energy_uJ:.1f} µJ")

env.close()
```

### Low-Level Client

```python
from pulse_ai_client import PulseClient

with PulseClient(port=5555) as client:
    states = client.receive_state()
    for state in states:
        print(f"Agent {state.agent_id}: {state.tag_position_gt}")
    client.send_action([[0, 1, 3]])
```

## Package Structure

```
external_ai_client/
├── pyproject.toml              # pip install configuration
├── train_agent.py              # Example training script
├── README.md                   # This file
└── pulse_ai_client/
    ├── __init__.py             # Package entry (exports PulseRLEnv, PulseClient, PulseState)
    ├── client.py               # TCP client for server communication
    ├── env.py                  # Gymnasium environment wrapper
    └── state.py                # Typed dataclass for the enriched state
```

## Configuring the Port

The port can be changed in two places:

1. **GUI**: Use the **Port** spinner in the AI Training Window toolbar.
2. **Code**: Pass `port=XXXX` to `PulseRLEnv(port=5555)` or `PulseClient(port=5555)`.

Both must match for the connection to work.
