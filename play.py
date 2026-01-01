import torch
import cv2
import numpy as np
import time
from dino_game import DinoGame
from train import DQN, process_screen  # Import the class from your training file

# --- SETUP ---
MODEL_PATH = "dino_brain_150.pth"  # <--- CHANGE THIS to your saved file name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD THE BRAIN ---
n_actions = 2
model = DQN(84, 84, n_actions).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()  # Set to "Evaluation Mode" (No learning, just doing)

print("AI Loaded! Press Ctrl+C to stop.")

# --- PLAY LOOP ---
game = DinoGame()

while True:
    # Reset
    raw_screen = game.reset()
    current_screen = process_screen(raw_screen)
    state = torch.cat([current_screen] * 4, dim=0).unsqueeze(0)
    
    total_reward = 0
    done = False
    
    while not done:
        # Ask the AI
        with torch.no_grad():
            # Get the best action (No randomness/Epsilon here!)
            action = model(state).max(1)[1].view(1, 1)
        
        # Move
        raw_screen, reward, done = game.step(action.item())
        total_reward += reward
        
        # Update "Eyes"
        screen_tensor = process_screen(raw_screen)
        next_state = torch.cat((state[0, 1:], screen_tensor), dim=0).unsqueeze(0)
        state = next_state
        
        # Slow it down slightly so you can watch (optional)
        time.sleep(0.03) 

    print(f"Game Over. Score: {total_reward}")