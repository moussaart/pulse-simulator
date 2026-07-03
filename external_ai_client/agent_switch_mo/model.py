import torch
import torch.nn as nn
import random
import numpy as np

class QNetwork(nn.Module):
    """Deep Q-Network for mapping augmented states to action Q-values."""
    def __init__(self, state_dim: int, action_dim: int):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience Replay Buffer for uniform off-policy sampling."""
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int, device: torch.device = torch.device("cpu")):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).to(device)
        )
        
    def __len__(self):
        return len(self.buffer)


def train_dqn_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma=0.99, device: torch.device = torch.device("cpu")):
    """Performs a single Deep Q-Network optimization step."""
    if len(replay_buffer) < batch_size:
        return 0.0
        
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size, device)
    
    # Q(s, a)
    q_values = policy_net(states)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # max Q(s', a') using the target network
    with torch.no_grad():
        next_q_values = target_net(next_states)
        next_state_values = next_q_values.max(1)[0]
        expected_state_action_values = rewards + (1 - dones) * gamma * next_state_values
        
    loss = nn.MSELoss()(state_action_values, expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stabilization
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    return float(loss.item())
