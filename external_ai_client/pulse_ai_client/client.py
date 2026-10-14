"""
Low-level TCP client for communicating with the PULSE AI Gym Server.
"""

import socket
import json
import time
from typing import Optional, List, Dict, Any

from .state import PulseState

PROTOCOL_VERSION = 3


class ProtocolError(ConnectionError):
    """The server replied with a malformed or incompatible protocol frame."""


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
        self.handshake: Optional[Dict[str, Any]] = None
        self.algorithms: List[str] = []
        self.num_agents = 1

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
            candidate = None
            reader = None
            try:
                candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                candidate.settimeout(self.timeout)
                candidate.connect((self.host, self.port))
                reader = candidate.makefile('r', encoding='utf-8')
                line = reader.readline()
                if not line:
                    raise ConnectionError("Server disconnected before protocol handshake")
                try:
                    hello = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProtocolError("Invalid JSON in protocol handshake") from exc
                if not isinstance(hello, dict) or hello.get("type") != "hello":
                    raise ProtocolError("Expected protocol hello handshake")
                if hello.get("protocol_version") != PROTOCOL_VERSION:
                    raise ProtocolError(
                        "protocol version mismatch: expected %d, got %r"
                        % (PROTOCOL_VERSION, hello.get("protocol_version"))
                    )
                self._socket = candidate
                self._reader = reader
                self.handshake = hello
                self.algorithms = list(hello.get("algorithms", []) or [])
                self.num_agents = int(hello.get("num_agents", 1))
                print(f"[PulseClient] Connected to {self.host}:{self.port}")
                return
            except ProtocolError:
                if reader is not None:
                    reader.close()
                if candidate is not None:
                    candidate.close()
                raise
            except ConnectionRefusedError:
                print(f"[PulseClient] Connection attempt {attempt + 1}/{retries} failed, retrying in {delay}s...")
                time.sleep(delay)
            except Exception as e:
                print(f"[PulseClient] Connection error: {e}")
                time.sleep(delay)
            finally:
                if candidate is not None and candidate is not self._socket:
                    if reader is not None:
                        try:
                            reader.close()
                        except Exception:
                            pass
                    try:
                        candidate.close()
                    except Exception:
                        pass

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
            data = self._receive_json()
            if isinstance(data, dict) and data.get("type") == "error":
                raise ProtocolError(
                    "server error %s: %s"
                    % (data.get("code", "unknown"), data.get("message", ""))
                )

            # Server sends a list of state dicts (one per agent)
            if isinstance(data, list):
                return [PulseState.from_dict(s) for s in data]
            else:
                return [PulseState.from_dict(data)]

        except json.JSONDecodeError as e:
            raise ProtocolError(f"Invalid JSON from server: {e}")

    def _receive_json(self) -> Any:
        if not self._reader:
            raise ConnectionError("Not connected to server")
        line = self._reader.readline()
        if not line:
            raise ConnectionError("Server disconnected")
        try:
            result = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProtocolError("Invalid JSON from server: %s" % exc) from exc
        if not isinstance(result, (dict, list)):
            raise ProtocolError("Server frame must be a JSON object or list")
        return result

    def reset(self) -> Dict[str, Any]:
        """Request an acknowledged environment reset (the next state is separate)."""
        if not self._socket:
            raise ConnectionError("Not connected to server")
        try:
            self._socket.sendall((json.dumps({"command": "reset"}) + "\n").encode("utf-8"))
        except OSError as exc:
            raise ConnectionError("Failed to send reset command: %s" % exc) from exc
        response = self._receive_json()
        if not isinstance(response, dict):
            raise ProtocolError("Expected reset acknowledgement object")
        if response.get("type") == "error":
            raise ProtocolError(
                "server error %s: %s"
                % (response.get("code", "unknown"), response.get("message", ""))
            )
        if response.get("type") != "ack" or response.get("command") != "reset":
            raise ProtocolError("Expected reset acknowledgement")
        return response

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
            self.handshake = None
            self.algorithms = []
            self.num_agents = 1
            print("[PulseClient] Disconnected.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
