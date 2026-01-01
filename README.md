# RL-Game Model (Dino Run AI)

A Reinforcement Learning project where an AI Agent learns to play the Chrome Dino game using a Deep Q-Network (DQN).

## 🎮 Project Overview

This project implements a custom version of the famous Dino Run game using `pygame` and trains an agent to play it using `PyTorch`. The agent "sees" the game screen as raw pixels (just like a human) and learns optimal actions (Jump or Run) through trial and error.

### Key Features
- **Custom Environment**: A lightweight Dino game environment built from scratch.
- **Deep Q-Learning (DQN)**: Uses a Convolutional Neural Network (CNN) to process screen frames.
- **Experience Replay**: Stores and learns from past moves to improve stability.
- **Model Saving/Loading**: Train the agent and watch it play later.

## 📂 Project Structure

- `dino_game.py`: The game engine (Environment). Handles physics, rendering, and game logic.
- `train.py`: The training loop. Implements the DQN architecture, experience replay, and optimization.
- `play.py`: Loads a trained model and lets the AI play the game.
- `requirements.txt`: List of dependencies.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation

1. Clone the repository (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd RL-Game-model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🧠 How to Use

### 1. Train the AI (Teach it!)
Run the training script to start the learning process. The agent will start knowing nothing and improve over time.
```bash
python train.py
```
- It will save checkpoints as `dino_brain_<episode>.pth` every 50 episodes.
- Press `Ctrl+C` to stop training.

### 2. Watch the AI Play (Test it!)
Once you have a trained model (e.g., `dino_brain_150.pth`), you can watch it play.

1. Open `play.py`.
2. Update the `MODEL_PATH` variable to match your saved model filename:
   ```python
   MODEL_PATH = "dino_brain_150.pth"
   ```
3. Run the script:
   ```bash
   python play.py
   ```

## 🛠️ Dependencies
- `torch` (PyTorch)
- `pygame`
- `opencv-python`
- `numpy`
