"""
DQN Agent with Double DQN, Dueling architecture support,
soft (Polyak) target updates, and gradient clipping.
"""
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    FRAME_SIZE, FEATURE_SIZE, N_ACTIONS,
    BATCH_SIZE, GAMMA, LEARNING_RATE, MEMORY_SIZE, WARMUP_STEPS,
    EPS_START, EPS_END, EPS_DECAY, TARGET_UPDATE,
)
from models.dqn import DQN, DuelingDQN, MLPDQN
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Deep Q-Learning Agent.

    Capabilities:
        • Standard DQN **or** Dueling DQN architecture
        • Double DQN target computation (eliminates Q-value overestimation)
        • Soft (Polyak) **or** hard target-network updates
        • Warmup phase: random play fills the buffer before any gradient step
        • Gradient clipping for training stability
        • Full checkpoint save/load (model + optimizer + training state)
    """

    def __init__(self, obs_mode='pixels', architecture='dueling',
                 double_dqn=True, soft_update=True, tau=0.005):
        """
        Args:
            obs_mode:     'pixels' (CNN) or 'features' (MLP)
            architecture: 'dqn' or 'dueling'
            double_dqn:   Use Double DQN target computation
            soft_update:  Use Polyak averaging instead of periodic hard copy
            tau:          Polyak averaging coefficient (0 < τ ≪ 1)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.obs_mode = obs_mode
        self.double_dqn = double_dqn
        self.soft_update = soft_update
        self.tau = tau
        self.steps_done = 0

        # --- Build networks ---
        if obs_mode == 'pixels':
            NetClass = DuelingDQN if architecture == 'dueling' else DQN
            self.policy_net = NetClass(FRAME_SIZE, FRAME_SIZE, N_ACTIONS).to(self.device)
            self.target_net = NetClass(FRAME_SIZE, FRAME_SIZE, N_ACTIONS).to(self.device)
        else:
            self.policy_net = MLPDQN(FEATURE_SIZE, N_ACTIONS).to(self.device)
            self.target_net = MLPDQN(FEATURE_SIZE, N_ACTIONS).to(self.device)

        # Sync target ← policy
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer & replay memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(MEMORY_SIZE)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, evaluate=False):
        """
        Epsilon-greedy action selection.

        Args:
            state:    numpy array observation
            evaluate: If True, always exploit (ε = 0)

        Returns:
            (action: int, q_values: np.ndarray or None)
        """
        if evaluate:
            eps = 0.0
        else:
            eps = EPS_END + (EPS_START - EPS_END) * \
                  np.exp(-1.0 * self.steps_done / EPS_DECAY)
            self.steps_done += 1

        if random.random() > eps:
            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax(dim=1).item(), q_values.cpu().numpy()[0]
        else:
            return random.randrange(N_ACTIONS), None

    def get_epsilon(self):
        """Current exploration rate."""
        return EPS_END + (EPS_START - EPS_END) * \
               np.exp(-1.0 * self.steps_done / EPS_DECAY)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def store_transition(self, state, action, next_state, reward, done):
        """Push a transition into the replay buffer."""
        self.memory.push(state, action, next_state, reward, done)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self):
        """
        One gradient step on a random mini-batch from replay memory.

        Returns:
            float: loss value, or None if the buffer hasn't met warmup threshold.
        """
        if len(self.memory) < max(BATCH_SIZE, WARMUP_STEPS):
            return None

        # Sample
        states, actions, next_states, rewards, dones = self.memory.sample(BATCH_SIZE)

        # Numpy → Tensors
        states_t      = torch.from_numpy(states).to(self.device)
        actions_t     = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        rewards_t     = torch.from_numpy(rewards).to(self.device)
        dones_t       = torch.from_numpy(dones).to(self.device)

        # --- Current Q-values: Q(s, a) ---
        current_q = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        # --- Target Q-values ---
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: policy_net SELECTS action, target_net EVALUATES it
                best_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_net(next_states_t).max(1)[0]

            # Bellman equation: r + γ · Q(s', a') · (1 − done)
            target_q = rewards_t + GAMMA * next_q * (1.0 - dones_t)

        # --- Loss & Backprop ---
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # --- Update target network ---
        if self.soft_update:
            # Polyak averaging: θ_target = τ·θ_policy + (1−τ)·θ_target
            for tp, pp in zip(self.target_net.parameters(),
                              self.policy_net.parameters()):
                tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)
        else:
            if self.steps_done % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, filepath, episode, best_score):
        """Save full training state (model + optimizer + counters)."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        torch.save({
            'episode': episode,
            'steps_done': self.steps_done,
            'best_score': best_score,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath):
        """
        Restore full training state for resuming.

        Returns:
            (start_episode: int, best_score: float)
        """
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
        self.target_net.load_state_dict(ckpt['target_net_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.steps_done = ckpt['steps_done']
        return ckpt.get('episode', 0), ckpt.get('best_score', 0)

    def load_model_only(self, filepath):
        """Load just the model weights (for evaluation / play mode)."""
        data = torch.load(filepath, map_location=self.device)
        if 'policy_net_state_dict' in data:
            self.policy_net.load_state_dict(data['policy_net_state_dict'])
        else:
            # Legacy format: plain state_dict from the original codebase
            self.policy_net.load_state_dict(data)
        self.policy_net.eval()
