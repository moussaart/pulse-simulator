"""
Low-level TCP client for communicating with the PULSE AI Gym Server.
"""

import socket
import json
import time
from typing import Optional, List, Dict, Any

from .state import PulseState


class PulseClient:
    """
    TCP client that connects to the PULSE simulator's AIGymServer.

    Handles connection management, state reception, and action sending.

    Args:
        host: Server hostname (default: localhost)
        port: Server port (default: 5555)
        timeout: Socket timeout in seconds (default: 30)

    Example:
        >>> client = PulseClient(port=5555)
        >>> client.connect()
        >>> states = client.receive_state()
        >>> for state in states:
        ...     print(f"Agent {state.agent_id}: error={state.error:.3f}m")
        >>> client.send_action([[0, 1, 3]])
        >>> client.close()
    """

    def __init__(self, host: str = "localhost", port: int = 5555, timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._reader = None

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self, retries: int = 10, delay: float = 1.0) -> None:
        """
        Connect to the PULSE simulator server.

        Args:
            retries: Number of connection attempts
            delay: Delay between retries (seconds)
        """
        for attempt in range(retries):
            try:
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.timeout)
                self._socket.connect((self.host, self.port))
                self._reader = self._socket.makefile('r', encoding='utf-8')
                print(f"[PulseClient] Connected to {self.host}:{self.port}")
                return
            except ConnectionRefusedError:
                print(f"[PulseClient] Connection attempt {attempt + 1}/{retries} failed, retrying in {delay}s...")
                time.sleep(delay)
            except Exception as e:
                print(f"[PulseClient] Connection error: {e}")
                time.sleep(delay)

        raise ConnectionError(
            f"Failed to connect to PULSE server at {self.host}:{self.port} after {retries} attempts"
        )

    def receive_state(self) -> List[PulseState]:
        """
        Receive and parse state(s) from the server.

        Returns:
            List of PulseState objects (one per agent)

        Raises:
            ConnectionError: If not connected or connection lost
        """
        if not self._reader:
            raise ConnectionError("Not connected to server")

        try:
            line = self._reader.readline()
            if not line:
                raise ConnectionError("Server disconnected")

            data = json.loads(line)

            # Server sends a list of state dicts (one per agent)
            if isinstance(data, list):
                return [PulseState.from_dict(s) for s in data]
            else:
                return [PulseState.from_dict(data)]

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from server: {e}")

    def send_action(
        self,
        action: List[Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Send action(s) back to the server.

        Args:
            action: List of actions (one per agent).
                    Each action can be a list of anchor indices, or a generic dictionary
                    specifying 'anchors', 'filter', 'measurement_source', etc.
            metrics: Optional training metrics (reward, loss, entropy, etc.)
        """
        if not self._socket:
            raise ConnectionError("Not connected to server")

        payload: Dict[str, Any] = {"action": action}
        if metrics:
            payload["metrics"] = metrics

        try:
            msg = (json.dumps(payload) + "\n").encode('utf-8')
            self._socket.sendall(msg)
        except Exception as e:
            raise ConnectionError(f"Failed to send action: {e}")

    def close(self) -> None:
        """Close the connection."""
        if self._reader:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None

        if self._socket:
            try:
                # Gracefully shut down the socket connection first
                self._socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
            print("[PulseClient] Disconnected.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
