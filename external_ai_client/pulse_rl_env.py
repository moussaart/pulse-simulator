import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json

class PulseRLEnv(gym.Env):
    """
    OpenAI Gymnasium Environment that connects to the PULSE Simulator GUI 
    via TCP socket. The GUI acts as the server, and this environment acts
    as the client.
    
    The RL agent learns to select a subset of anchors to minimize localization error.
    """
    
    def __init__(self, host='127.0.0.1', port=5555, max_anchors=4):
        super(PulseRLEnv, self).__init__()
        
        self.host = host
        self.port = port
        self.max_anchors = max_anchors
        self.client_socket = None
        
        # Action space: Binary array [1, 0, 1, 1] means use A1, A3, A4. Drop A2.
        self.action_space = spaces.MultiBinary(self.max_anchors)
        
        # Observation space: Anchor positions (max_anchors * 2) + localization_error (1)
        # Note: We remove measurements as per request
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(self.max_anchors * 2 + 1,), dtype=np.float32)
        
        self.connect()

    def connect(self):
        if self.client_socket:
            self.client_socket.close()
            
        print(f"Connecting to PULSE Simulator at {self.host}:{self.port}...")
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Keep trying to connect (wait for GUI to select the AI algorithm)
        while True:
            try:
                self.client_socket.connect((self.host, self.port))
                print("Connected successfully!")
                break
            except ConnectionRefusedError:
                pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # To completely reset, you can either send a reset command to GUI
        # Or you can just wait for the next state from the ongoing simulation
        
        # Wait for the first state from the server
        state_dict = self._receive_state()
        
        obs = self._extract_obs(state_dict)
        return obs, {}

    def step(self, action):
        # 1. Send action (anchor mask) to simulator
        action_dict = {"anchor_mask": action.tolist()}
        self._send_action(action_dict)
        
        # 2. Wait for next state from simulator
        state_dict = self._receive_state()
        
        # 3. Extract observation
        obs = self._extract_obs(state_dict)
        
        # 4. Calculate reward (Negative error between estimated and true position)
        # Note: In a real scenario, the GUI would compute the position using the previous action,
        # then send the new state. For a simple reward, we can check how many anchors were used
        # and if the state provides the error. Here we just use a placeholder reward.
        reward = self._compute_reward(state_dict, action)
        
        # For continuous simulation, we never terminate unless GUI stops sending
        terminated = False
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _extract_obs(self, state_dict):
        # State now consists of anchor positions and current localization error
        anchor_poses = state_dict.get("anchor_positions", [])
        curr_error = state_dict.get("localization_error", 0.0)
        
        # Flatten anchor positions: [x1, y1, x2, y2, ...]
        obs_parts = []
        for i in range(self.max_anchors):
            if i < len(anchor_poses):
                obs_parts.extend([float(anchor_poses[i][0]), float(anchor_poses[i][1])])
            else:
                obs_parts.extend([0.0, 0.0]) # Padding
        
        obs_parts.append(float(curr_error))
        return np.array(obs_parts, dtype=np.float32)

    def _compute_reward(self, state_dict, action):
        # Reward = +1 if error decreases, -1 otherwise
        curr_error = state_dict.get("localization_error", 0.0)
        prev_error = state_dict.get("prev_localization_error", 0.0)
        
        if curr_error < prev_error:
            return 1.0
        else:
            return -1.0

    def _receive_state(self):
        buffer = ""
        while True:
            try:
                data = self.client_socket.recv(4096).decode('utf-8')
                if not data:
                    raise ConnectionError("Simulator disconnected")
                
                buffer += data
                if "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    return json.loads(line)
            except Exception as e:
                print(f"Connection lost: {e}")
                self.connect() # Reconnect on failure

    def _send_action(self, action_dict):
        try:
            data_str = json.dumps(action_dict) + "\n"
            self.client_socket.sendall(data_str.encode('utf-8'))
        except Exception as e:
            print(f"Failed to send action: {e}")
            self.connect()

    def close(self):
        if self.client_socket:
            self.client_socket.close()
