import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
from dino_game import DinoGame  # Imports your game file

# --- 1. HYPERPARAMETERS ---
BATCH_SIZE = 32
GAMMA = 0.99           # Discount factor (cares about future)
EPS_START = 1.0        # 100% random at start
EPS_END = 0.01         # 1% random at end
EPS_DECAY = 10000      # How fast to stop being random
TARGET_UPDATE = 1000   # Update "Teacher" network every 1k steps
LR = 0.00025           # Learning Rate
MEMORY_SIZE = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. THE BRAIN (Neural Network) ---
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        # 1. Visual Layers (CNN)
        # Input: 4 stacked frames (grayscale)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the output from conv layers to feed into linear
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        # 2. Decision Layers (Fully Connected)
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.head = nn.Linear(512, outputs) # Outputs: [Jump_Score, Do_Nothing_Score]

    def forward(self, x):
        x = x.to(device)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        x = torch.relu(self.fc1(x))
        return self.head(x)

# --- 3. HELPER: PROCESS IMAGE ---
def process_screen(screen):
    # Convert to grayscale
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    # Resize to 84x84
    screen = cv2.resize(screen, (84, 84))
    # Normalize (0-1) and add dimension for PyTorch
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255.0
    screen = torch.from_numpy(screen)
    return screen.unsqueeze(0) # Shape: (1, 84, 84)

# --- 4. MAIN TRAINING LOOP ---
game = DinoGame()
n_actions = 2 # Jump (1) or Nothing (0)

# Initialize Networks
policy_net = DQN(84, 84, n_actions).to(device)
target_net = DQN(84, 84, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict()) # Clone student to teacher
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

steps_done = 0

print(f"Training on {device}... Press Ctrl+C to stop.")

for i_episode in range(1000): # Play 1000 games
    # Reset Environment
    raw_screen = game.reset()
    current_screen = process_screen(raw_screen)
    # Create initial stack of 4 frames (all same at start)
    state = torch.cat([current_screen] * 4, dim=0).unsqueeze(0) # Shape (1, 4, 84, 84)
    
    total_reward = 0
    
    while True:
        # A. SELECT ACTION (Epsilon Greedy)
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        np.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                # Ask the brain (Exploit)
                action = policy_net(state).max(1)[1].view(1, 1)
        else:
            # Random move (Explore)
            action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

        # B. EXECUTE ACTION
        raw_screen, reward, done = game.step(action.item())
        total_reward += reward
        
        # Process new state
        screen_tensor = process_screen(raw_screen)
        # Shift stack: Remove oldest frame, add newest
        next_state = torch.cat((state[0, 1:], screen_tensor), dim=0).unsqueeze(0)

        # C. STORE MEMORY
        reward_tensor = torch.tensor([reward], device=device)
        memory.append((state, action, next_state, reward_tensor, done))
        state = next_state

        # D. OPTIMIZE (Train the Brain)
        if len(memory) > BATCH_SIZE:
            transitions = random.sample(memory, BATCH_SIZE)
            # Unzip the batch data
            batch_state, batch_action, batch_next, batch_reward, batch_done = zip(*transitions)
            
            batch_state = torch.cat(batch_state)
            batch_action = torch.cat(batch_action)
            batch_reward = torch.cat(batch_reward)
            batch_next = torch.cat(batch_next)
            
            # 1. Current Q-Values (Student Guess)
            current_q_values = policy_net(batch_state).gather(1, batch_action)
            
            # 2. Target Q-Values (Teacher Truth)
            # "What is the best future move?"
            next_q_values = target_net(batch_next).max(1)[0].detach()
            
            # Bellman Equation
            # If done, target = reward. If not, target = reward + gamma * future
            # We use a mask (1 - batch_done) to handle game over logic
            mask = torch.tensor([0 if d else 1 for d in batch_done], device=device)
            expected_q_values = batch_reward + (next_q_values * GAMMA * mask)

            # 3. Compute Loss & Backprop
            loss = nn.functional.smooth_l1_loss(current_q_values, expected_q_values.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update Target Network occasionally
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            break
            
    # ... inside the main loop ...
        
    print(f"Episode {i_episode} - Score: {total_reward} - Epsilon: {eps_threshold:.2f}")

    # SAVE THE BRAIN every 50 episodes
    if i_episode % 50 == 0:
        torch.save(policy_net.state_dict(), f"dino_brain_{i_episode}.pth")
        print(f"--- Saved Checkpoint: dino_brain_{i_episode}.pth ---")

