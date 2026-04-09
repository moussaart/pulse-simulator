import socket
import json
import torch    
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import time

# ============================================================================
# PULSE UWB Simulator - PyTorch RL Agent Example
# ============================================================================
# Connects to the AITrainingWindow (port 5555), receives the environment 
# state, and uses a PyTorch Neural Network to predict the 3 best anchors
# with the goal to minimize localization error (Proxy: select LOS + close).
# ============================================================================

HOST = '127.0.0.1'
PORT = 5555

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"==========================================")
print(f" PyTorch Network Device: {device}")
print(f"==========================================")

class AnchorSelectNetwork(nn.Module):
    def __init__(self, num_anchors, hidden_size=64):
        super(AnchorSelectNetwork, self).__init__()
        # Input features: anchor positions (num_anchors * 2) + localization error (1)
        input_size = num_anchors * 2 + 1
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_anchors) # Output: Logits for each anchor
        )
    
    def forward(self, state):
        return self.net(state)

def connect_to_simulator(host=HOST, port=PORT, retries=5):
    for attempt in range(retries):
        try:
            print(f"Attempting to connect to AI Window at {host}:{port}...")
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_sock.connect((host, port))
            print("✅ Successfully connected to the PULSE AI Window!")
            return client_sock
        except ConnectionRefusedError:
            print(f"Connection refused. Ensure the AI Window is open and Play is pressed. (Attempt {attempt+1}/{retries})")
            time.sleep(2)
    return None

def run_rl_loop():
    client_sock = connect_to_simulator()
    if not client_sock:
        print("❌ Failed to connect. Exiting.")
        return

    stream_reader = client_sock.makefile('r', encoding='utf-8')
    print("\nListening for environment state...\n")
    print("-" * 60)
    
    # RL Setup
    num_anchors = None
    policy_net = None
    optimizer = None
    
    # Store environment parameters once received
    env_params_printed = False
    
    try:
        for line in stream_reader:
            if not line.strip():
                continue
                
            try:
                # 1. READ STATE
                raw_data = json.loads(line)
                if isinstance(raw_data, list):
                    states = raw_data
                else:
                    states = [raw_data]
                    
                all_actions = []
                sum_reward = 0.0
                sum_loss = 0.0
                sum_entropy = 0.0
                
                for state_dict in states:
                    step = state_dict.get('step', 0)
                    agent_id = state_dict.get('agent_id', 0)
                    gt = state_dict.get('tag_position_gt', [0, 0, 0])
                    anchor_positions = state_dict.get('anchor_positions', [])
                    curr_error = state_dict.get('localization_error', 0.0)
                    prev_error = state_dict.get('prev_localization_error', 0.0)
                    los_conditions = state_dict.get('is_los', state_dict.get('los_conditions', []))
                    
                    if not env_params_printed:
                        print("\n" + "="*40)
                        print(f" Environment Parameters:")
                        print(f" - Algorithm: {state_dict.get('algorithm', 'Unknown')}")
                        print(f" - Time Step (dt): {state_dict.get('dt', 'Unknown')}s")
                        print(f" - Speed: {state_dict.get('movement_speed', 'Unknown')}m/s")
                        print(f" - Pattern: {state_dict.get('movement_pattern', 'Unknown')}")
                        print("="*40 + "\n")
                        env_params_printed = True

                    if agent_id == 0:
                        print(f"[Step {step:04d}] Processing {len(states)} agents. Agent 0 GT: ({gt[0]:.2f}, {gt[1]:.2f})")
                    
                    # Dynamic Initialization of PyTorch network based on environment config
                    if policy_net is None:
                        num_anchors = len(anchor_positions)
                        if num_anchors > 0:
                            policy_net = AnchorSelectNetwork(num_anchors=num_anchors).to(device)
                            optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
                            print(f"✅ Initialized PyTorch Network for {num_anchors} anchors on {device}.")

                    if num_anchors is None or num_anchors < 3:
                        # Fallback for environments with too few anchors
                        action_indices = list(range(num_anchors if num_anchors else 0))
                        reward_val, policy_loss_val, entropy_val = 0.0, 0.0, 0.0
                    else:
                        # 2. PREPROCESS STATE
                        # State: flattened anchor positions + localization error
                        anchor_arr = np.array(anchor_positions, dtype=np.float32).flatten() / 10.0 # Normalize roughly
                        error_val = np.array([curr_error], dtype=np.float32)
                        
                        state_features = np.concatenate((anchor_arr, error_val))
                        state_tensor = torch.tensor(state_features, dtype=torch.float32, device=device)
                        
                        # 3. FORWARD PASS
                        logits = policy_net(state_tensor)
                        dist = Categorical(logits=logits) # Numerically stable
                        probs = dist.probs
                        
                        action_probs, action_indices_tensor = torch.topk(probs, k=3)
                        action_indices = action_indices_tensor.tolist()
                        
                        # 4. COMPUTE REWARD
                        # +1 if error decreases, else -1
                        if curr_error < prev_error:
                            reward_val = 1.0
                        else:
                            reward_val = -1.0
                        
                        # 5. POLICY GRADIENT UPDATE
                        joint_log_prob = dist.log_prob(action_indices_tensor).sum()
                        loss = -joint_log_prob * reward_val
                        
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0) # Prevent exploding gradients
                        optimizer.step()
                        
                        policy_loss_val = loss.item()
                        entropy_val = dist.entropy().item()
                        
                    all_actions.append(action_indices)
                    sum_reward += reward_val
                    sum_loss += policy_loss_val
                    sum_entropy += entropy_val
                
                num_states = max(1, len(states))
                avg_reward = sum_reward / num_states
                avg_loss = sum_loss / num_states
                avg_entropy = sum_entropy / num_states

                print(f"  └─ Avg Metrics: Reward={avg_reward:.2f} | Loss={avg_loss:.4f} | Entropy={avg_entropy:.4f}")
                
                # 6. SEND ACTION BACK TO SIMULATOR
                response = {
                    "action": all_actions,
                    "metrics": {
                        "reward": float(avg_reward),
                        "policy_loss": float(avg_loss),
                        "entropy": float(avg_entropy)
                    }
                }
                json_response = json.dumps(response) + "\n"
                client_sock.sendall(json_response.encode('utf-8'))
                
            except json.JSONDecodeError:
                print("Failed to decode JSON state.")
            except Exception as e:
                print(f"Error processing step: {e}")
                
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    except Exception as e:
        print(f"\n❌ Client Error: {e}")
    finally:
        client_sock.close()
        print("Connection closed.")

if __name__ == "__main__":
    run_rl_loop()
