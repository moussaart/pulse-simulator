"""
PULSE AI Client — SDK for connecting RL agents to the PULSE UWB Simulator.

Usage:
    from pulse_ai_client import PulseRLEnv, PulseClient, PulseState

    # Quick start with Gymnasium
    env = PulseRLEnv(port=5555)
    obs, info = env.reset()

    # Or use the low-level client directly
    client = PulseClient(port=5555)
    client.connect()
    state = client.receive_state()
    client.send_action([0, 1, 3])
"""

__version__ = "1.0.0"

from .client import PulseClient
from .env import PulseRLEnv
from .state import PulseState

__all__ = ["PulseClient", "PulseRLEnv", "PulseState", "__version__"]
