"""
Centralized configuration for the RL Dino Game.
All hyperparameters, game settings, and training configs in one place.
"""

# ============================================================
# GAME SETTINGS
# ============================================================
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 300
GROUND_Y = 280                  # y-coordinate of the ground line

DINO_X = 50
DINO_WIDTH = 40
DINO_HEIGHT = 60
JUMP_VELOCITY = -15
GRAVITY = 1

OBSTACLE_MIN_WIDTH = 20
OBSTACLE_MAX_WIDTH = 40
OBSTACLE_MIN_HEIGHT = 30
OBSTACLE_MAX_HEIGHT = 60

MIN_SPAWN_INTERVAL = 30        # Minimum frames between obstacle spawns
MAX_SPAWN_INTERVAL = 80        # Maximum frames between obstacle spawns

INITIAL_GAME_SPEED = 6
MAX_GAME_SPEED = 15
SPEED_INCREMENT = 0.5
SPEED_INCREMENT_INTERVAL = 500  # Increase speed every N frames

# ============================================================
# REWARD SETTINGS
# ============================================================
REWARD_ALIVE = 0.1              # Per-frame survival reward
REWARD_OBSTACLE_CLEARED = 5.0   # Bonus for clearing an obstacle
REWARD_DEATH = -10.0            # Penalty for dying
REWARD_JUMP_PENALTY = -0.5      # Penalty for jumping (only from ground)

# ============================================================
# DQN HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 32
GAMMA = 0.99                    # Discount factor (how much to value future rewards)
EPS_START = 1.0                 # 100% random at start
EPS_END = 0.01                  # 1% random at end
EPS_DECAY = 10000               # Exploration decay rate
TARGET_UPDATE = 1000            # Hard target update interval (if not using soft)
LEARNING_RATE = 0.00025
MEMORY_SIZE = 50000
WARMUP_STEPS = 1000             # Fill buffer with random play before training

# ============================================================
# TRAINING SETTINGS
# ============================================================
NUM_EPISODES = 1000
CHECKPOINT_INTERVAL = 50
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "runs"

# ============================================================
# OBSERVATION SETTINGS
# ============================================================
FRAME_SIZE = 84                 # Resize frames to 84x84
FRAME_STACK = 4                 # Stack 4 frames for temporal information
N_ACTIONS = 2                   # 0 = Do Nothing, 1 = Jump

# Feature vector size (for MLP mode)
FEATURE_SIZE = 6                # [dino_y, vel_y, obs_dist, obs_w, obs_h, speed]
