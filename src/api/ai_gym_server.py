import socket
import threading
import json
import time
from typing import Optional

# Protocol version for client compatibility detection
PROTOCOL_VERSION = 2


class AIGymServer:
    """
    Bidirectional TCP Server that acts as a Gymnasium environment backend.
    Sends `state` to an external RL client and waits for `action`.

    Protocol v2 features:
        - Enriched state with IMU, energy, NLOS, algorithm, and precision data.
        - Configurable port from the GUI.
        - Protocol version field in every state message for client detection.
    """

    def __init__(self, port: int = 5555):
        self.port = port
        self.host = '0.0.0.0'
        
        self.server_socket: Optional[socket.socket] = None
        self.client_socket: Optional[socket.socket] = None
        self.client_address = None
        
        self.is_running = False
        self._server_thread: Optional[threading.Thread] = None
        
        # Thread sync
        self.current_action = None
        self.current_metrics = None
        self.action_event = threading.Event()

        # Connection status tracking
        self._connected = False
        self._lock = threading.Lock()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        """Whether a client is currently connected."""
        with self._lock:
            return self._connected

    @property
    def protocol_version(self) -> int:
        """Return the current protocol version."""
        return PROTOCOL_VERSION

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        """Starts the TCP server in a background thread."""
        if self.is_running:
            return
            
        self.is_running = True
        self.action_event.clear()
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        print(f"[AIGymServer] Listening on {self.host}:{self.port}")

    def stop(self):
        """Stops the TCP server and closes connections."""
        self.is_running = False
        self.action_event.set()  # Unblock waiting threads
        
        if self.client_socket:
            try:
                # Gracefully shut down the client socket first
                self.client_socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.client_socket.close()
            except Exception:
                pass
            self.client_socket = None
            
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

        with self._lock:
            self._connected = False
            
        print("[AIGymServer] Stopped.")

    # ── Internal ──────────────────────────────────────────────────────────

    def _run_server(self):
        """Main loop for accepting client connections."""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)
            
            while self.is_running:
                try:
                    client_sock, addr = self.server_socket.accept()
                    print(f"[AIGymServer] Client connected from {addr}")
                    
                    if self.client_socket:
                        self.client_socket.close()
                        
                    self.client_socket = client_sock
                    self.client_address = addr
                    with self._lock:
                        self._connected = True
                    
                    # Handle incoming actions from this client
                    self._handle_client()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.is_running:
                        print(f"[AIGymServer] Accept error: {e}")
                        
        except Exception as e:
            if self.is_running:
                print(f"[AIGymServer] Server error: {e}")
        finally:
            self.stop()

    def _handle_client(self):
        """Reads actions back from the RL client."""
        try:
            reader = self.client_socket.makefile('r', encoding='utf-8')
            for line in reader:
                if not self.is_running:
                    break
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if "metrics" in data:
                        self.current_metrics = data["metrics"]
                    if "action" in data:
                        self.current_action = data["action"]
                        self.action_event.set()
                    elif "anchor_mask" in data:
                        # Fallback for legacy v1 client
                        mask = data["anchor_mask"]
                        indices = [i for i, val in enumerate(mask) if val]
                        self.current_action = indices
                        self.action_event.set()
                except Exception as e:
                    print(f"[AIGymServer] Error parsing client action: {e}")
        except Exception as e:
            print(f"[AIGymServer] Client disconnected: {e}")
        finally:
            if self.client_socket:
                try: self.client_socket.close()
                except: pass
            self.client_socket = None
            with self._lock:
                self._connected = False
            self.action_event.set()  # Unblock if disconnected

    # ── Public API ────────────────────────────────────────────────────────

    def send_state(self, state_data) -> bool:
        """
        Sends the state (dictionary or list of dictionaries) to the client.

        Automatically injects ``protocol_version`` into each state dict so
        clients can detect the schema.

        Returns True if successful.
        """
        if not self.is_running or not self.client_socket:
            return False
            
        try:
            # Inject protocol version into each state dict
            if isinstance(state_data, list):
                for s in state_data:
                    if isinstance(s, dict):
                        s["protocol_version"] = PROTOCOL_VERSION
            elif isinstance(state_data, dict):
                state_data["protocol_version"] = PROTOCOL_VERSION

            json_str = json.dumps(state_data)
            message = (json_str + "\n").encode('utf-8')
            self.client_socket.sendall(message)
            self.action_event.clear()  # Reset event to wait for next action
            return True
        except Exception as e:
            print(f"[AIGymServer] Failed to send state: {e}")
            self.action_event.set()
            return False
            
    def wait_for_action(self, timeout: float = None) -> Optional[tuple]:
        """
        Blocks until the client responds with an action. Give timeout for non-blocking.
        Returns a tuple of (action, metrics).
        """
        if not self.client_socket:
            return None
            
        success = self.action_event.wait(timeout)
        if success and self.current_action is not None:
            act = self.current_action
            met = self.current_metrics
            self.current_action = None
            self.current_metrics = None
            self.action_event.clear()
            return (act, met)
        return None
