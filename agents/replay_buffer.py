"""
Experience Replay Buffer for DQN training.

Stores transitions as numpy arrays instead of PyTorch tensors
to reduce memory usage by ~4x (float32 numpy vs. CUDA/CPU tensors
with autograd metadata).
"""
import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Fixed-size circular buffer that stores (s, a, s', r, done) tuples.

    Usage:
        buffer = ReplayBuffer(capacity=50000)
        buffer.push(state, action, next_state, reward, done)
        states, actions, next_states, rewards, dones = buffer.sample(32)
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        """
        Store a single transition.

        Args:
            state:      numpy array — current observation
            action:     int         — action taken
            next_state: numpy array — resulting observation
            reward:     float       — reward received
            done:       bool        — whether episode ended
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            np.array(next_state, dtype=np.float32),
            float(reward),
            float(done),
        ))

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions.

        Returns:
            Tuple of numpy arrays:
                states      — (batch, *obs_shape)
                actions     — (batch,)
                next_states — (batch, *obs_shape)
                rewards     — (batch,)
                dones       — (batch,)   (1.0 if done, 0.0 otherwise)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(next_states, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
