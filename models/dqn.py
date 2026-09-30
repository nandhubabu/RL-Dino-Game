"""
Neural network architectures for Deep Q-Learning.

Architectures:
    DQN         - Standard CNN-based Deep Q-Network (Nature DQN)
    DuelingDQN  - CNN with separate Value and Advantage streams
    MLPDQN      - Lightweight MLP for feature-vector observations
"""
import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Standard Deep Q-Network with CNN backbone.

    Architecture (Nature DQN):
        Conv2d(4→32, 8×8, stride=4) → ReLU
        Conv2d(32→64, 4×4, stride=2) → ReLU
        Conv2d(64→64, 3×3, stride=1) → ReLU
        Flatten → Linear(→512) → ReLU → Linear(→n_actions)

    Input:  (batch, 4, 84, 84) — stack of 4 grayscale frames
    Output: (batch, n_actions)  — Q-value per action
    """

    def __init__(self, h, w, n_actions):
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        # Decision layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        return self.head(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN: decomposes Q(s,a) into Value V(s) and Advantage A(s,a).

        Q(s,a) = V(s) + [ A(s,a) − mean_a'(A(s,a')) ]

    Why it helps:
        The network can learn which *states* are valuable independently
        of which *action* is best, leading to faster convergence.

    Same CNN backbone as standard DQN, but the fully-connected head
    splits into two separate streams.
    """

    def __init__(self, h, w, n_actions):
        super().__init__()
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        # Value stream: "How good is this state overall?"
        self.value_fc = nn.Linear(linear_input_size, 512)
        self.value = nn.Linear(512, 1)

        # Advantage stream: "How much better is each action than average?"
        self.advantage_fc = nn.Linear(linear_input_size, 512)
        self.advantage = nn.Linear(512, n_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # Value stream
        v = torch.relu(self.value_fc(x))
        v = self.value(v)                           # (batch, 1)

        # Advantage stream
        a = torch.relu(self.advantage_fc(x))
        a = self.advantage(a)                       # (batch, n_actions)

        # Combine: Q = V + (A - mean(A))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class MLPDQN(nn.Module):
    """
    Simple MLP-based DQN for feature-vector observations.

    Architecture:
        Linear(input→128) → ReLU → Linear(128→128) → ReLU → Linear(128→n_actions)

    Much faster to train than CNN — ideal for rapid prototyping
    and demonstrating RL concepts without GPU requirements.
    """

    def __init__(self, input_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)
