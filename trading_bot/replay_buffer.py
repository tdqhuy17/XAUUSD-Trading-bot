"""
Replay Buffer for SAC Agent
- Prioritized Experience Replay for efficient learning
- GPU-compatible tensor storage
"""

import torch
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Standard Replay Buffer
    Stores transitions (s, a, r, s', done) for off-policy learning
    """
    def __init__(self, capacity: int, device: str = 'cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions
        Returns tensors on the specified device
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
    def is_ready(self, batch_size: int):
        """Check if buffer has enough samples"""
        return len(self.buffer) >= batch_size
    
    def save(self, path: str):
        """Save buffer to file"""
        import pickle
        data = {
            'buffer': list(self.buffer),
            'capacity': self.capacity,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[ReplayBuffer] Saved {len(self.buffer)} transitions to {path}")
    
    def load(self, path: str):
        """Load buffer from file"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.buffer = deque(data['buffer'], maxlen=self.capacity)
        print(f"[ReplayBuffer] Loaded {len(self.buffer)} transitions from {path}")


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER)
    Samples important transitions more frequently
    """
    def __init__(self, capacity: int, device: str = 'cpu', alpha: float = 0.6, beta: float = 0.4):
        super().__init__(capacity, device)
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = 0.001
        self.max_priority = 1.0
        self.priorities = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Add transition with max priority"""
        super().push(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
        
    def sample(self, batch_size: int):
        """
        Sample with priority-based probabilities
        Returns importance sampling weights for gradient correction
        """
        if len(self.buffer) < batch_size:
            return super().sample(batch_size)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Increment beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get batch data
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors"""
        priorities = np.abs(priorities) + 1e-6  # Small epsilon for numerical stability
        for idx, priority in zip(indices, priorities):
            priority_clipped = min(priority, self.max_priority)
            self.priorities[idx] = priority_clipped
            self.max_priority = max(self.max_priority, priority_clipped)
    
    def save(self, path: str):
        """Save buffer with priorities to file"""
        import pickle
        data = {
            'buffer': list(self.buffer),
            'capacity': self.capacity,
            'priorities': list(self.priorities),
            'alpha': self.alpha,
            'beta': self.beta,
            'max_priority': self.max_priority,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[PrioritizedReplayBuffer] Saved {len(self.buffer)} transitions to {path}")
    
    def load(self, path: str):
        """Load buffer with priorities from file"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.buffer = deque(data['buffer'], maxlen=self.capacity)
        self.priorities = deque(data['priorities'], maxlen=self.capacity)
        self.alpha = data.get('alpha', 0.6)
        self.beta = data.get('beta', 0.4)
        self.max_priority = data.get('max_priority', 1.0)
        print(f"[PrioritizedReplayBuffer] Loaded {len(self.buffer)} transitions from {path}")


class GPUReplayBuffer:
    """
    GPU-native replay buffer using pre-allocated tensors.
    
    Stores transitions directly on the GPU device, eliminating
    the CPU->GPU transfer that happens on every batch sample with
    the standard ReplayBuffer. This is the main bottleneck for
    pure GPU training on CUDA devices.
    
    Requires state_dim and action_dim at construction time.
    """
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.capacity = capacity
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0      # Write pointer (circular)
        self._size = 0    # Current fill level

        # Pre-allocate tensors directly on the target device
        self.states      = torch.zeros(capacity, state_dim,  dtype=torch.float32, device=device)
        self.actions     = torch.zeros(capacity, action_dim, dtype=torch.float32, device=device)
        self.rewards     = torch.zeros(capacity, 1,          dtype=torch.float32, device=device)
        self.next_states = torch.zeros(capacity, state_dim,  dtype=torch.float32, device=device)
        self.dones       = torch.zeros(capacity, 1,          dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done):
        """Write one transition into the circular buffer."""
        idx = self.ptr
        # Convert numpy -> CPU tensor -> GPU in one step
        self.states[idx]      = torch.as_tensor(state,      dtype=torch.float32).to(self.device)
        self.actions[idx]     = torch.as_tensor(action,     dtype=torch.float32).to(self.device)
        self.rewards[idx]     = float(reward)
        self.next_states[idx] = torch.as_tensor(next_state, dtype=torch.float32).to(self.device)
        self.dones[idx]       = float(done)

        self.ptr   = (self.ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Sample a batch entirely on the GPU — no CPU/GPU transfer.
        Uses numpy randint (CPU) for index generation, then indexes
        pre-allocated GPU tensors (fast gather operation).
        """
        idxs = np.random.randint(0, self._size, size=batch_size)
        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs],
        )

    def is_ready(self, batch_size: int) -> bool:
        return self._size >= batch_size

    def __len__(self) -> int:
        return self._size

    def save(self, path: str):
        """Save buffer content to disk (CPU conversion for portability)."""
        import pickle
        data = {
            'states':      self.states[:self._size].cpu().numpy(),
            'actions':     self.actions[:self._size].cpu().numpy(),
            'rewards':     self.rewards[:self._size].cpu().numpy(),
            'next_states': self.next_states[:self._size].cpu().numpy(),
            'dones':       self.dones[:self._size].cpu().numpy(),
            'ptr':         self.ptr,
            'size':        self._size,
            'capacity':    self.capacity,
            'state_dim':   self.state_dim,
            'action_dim':  self.action_dim,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[GPUReplayBuffer] Saved {self._size} transitions to {path}")

    def load(self, path: str):
        """Load buffer from disk back onto the GPU device."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        n = data['size']
        self.states[:n]      = torch.as_tensor(data['states'],      dtype=torch.float32).to(self.device)
        self.actions[:n]     = torch.as_tensor(data['actions'],     dtype=torch.float32).to(self.device)
        self.rewards[:n]     = torch.as_tensor(data['rewards'],     dtype=torch.float32).to(self.device)
        self.next_states[:n] = torch.as_tensor(data['next_states'], dtype=torch.float32).to(self.device)
        self.dones[:n]       = torch.as_tensor(data['dones'],       dtype=torch.float32).to(self.device)
        self.ptr   = data['ptr']
        self._size = n
        print(f"[GPUReplayBuffer] Loaded {n} transitions from {path}")


class EpisodeBuffer:
    """
    Buffer for storing complete episodes
    Useful for recurrent policies or analyzing performance
    """
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.episodes = deque(maxlen=capacity)
        self.current_episode = []
        
    def push_step(self, state, action, reward, next_state, done, info=None):
        """Add step to current episode"""
        self.current_episode.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info or {}
        })
        
        if done:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            
    def get_latest_episode(self):
        """Get the most recent complete episode"""
        if self.episodes:
            return self.episodes[-1]
        return None
    
    def get_episode_stats(self):
        """Calculate statistics across episodes"""
        if not self.episodes:
            return {}
        
        rewards = [sum(step['reward'] for step in ep) for ep in self.episodes]
        lengths = [len(ep) for ep in self.episodes]
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_length': np.mean(lengths),
            'num_episodes': len(self.episodes)
        }


class RolloutBuffer:
    """
    Buffer for on-policy rollouts
    Used when collecting trajectories with the current policy
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def push(self, state, action, reward, value, log_prob, done):
        """Add step to rollout"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        
    def clear(self):
        """Clear all stored data"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def compute_returns(self, gamma: float = 0.99, lam: float = 0.95):
        """
        Compute GAE (Generalized Advantage Estimation)
        Returns advantages and returns
        """
        n = len(self.rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)
        
        last_gae = 0
        last_return = self.values[-1] if self.values else 0
        
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = 0
                next_non_terminal = 1 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1 - self.dones[t]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
            
            returns[t] = advantages[t] + self.values[t]
            
        return advantages, returns
    
    def get_batch(self, device: str = 'cpu'):
        """Get all data as tensors"""
        return (
            torch.FloatTensor(np.array(self.states)).to(device),
            torch.FloatTensor(np.array(self.actions)).to(device),
            torch.FloatTensor(np.array(self.rewards)).to(device),
            torch.FloatTensor(np.array(self.values)).to(device),
            torch.FloatTensor(np.array(self.log_probs)).to(device),
            torch.FloatTensor(np.array(self.dones)).to(device),
        )
    
    def __len__(self):
        return len(self.states)


# ============== UTILITY FUNCTIONS ==============

def create_buffer(config: dict, device: str = 'cpu',
                  state_dim: int = None, action_dim: int = None):
    """
    Factory function to create the appropriate replay buffer.

    Buffer types:
    - 'gpu'         : GPUReplayBuffer  – pre-allocated tensors on the GPU device
                      (requires state_dim and action_dim, best for CUDA)
    - 'prioritized' : PrioritizedReplayBuffer – CPU-side PER
    - 'standard'    : plain ReplayBuffer (default / fallback)
    """
    buffer_type = config.get('buffer_type', 'standard')
    capacity    = config.get('buffer_size', 1_000_000)

    if buffer_type == 'gpu':
        if state_dim is None or action_dim is None:
            raise ValueError("GPUReplayBuffer requires state_dim and action_dim")
        print(f"[Buffer] Using GPUReplayBuffer on {device} "
              f"(capacity={capacity:,}, s={state_dim}, a={action_dim})")
        return GPUReplayBuffer(capacity, state_dim, action_dim, device)

    elif buffer_type == 'prioritized':
        alpha = config.get('per_alpha', 0.6)
        beta  = config.get('per_beta',  0.4)
        print(f"[Buffer] Using PrioritizedReplayBuffer (capacity={capacity:,})")
        return PrioritizedReplayBuffer(capacity, device, alpha, beta)

    else:
        print(f"[Buffer] Using standard ReplayBuffer (capacity={capacity:,})")
        return ReplayBuffer(capacity, device)
