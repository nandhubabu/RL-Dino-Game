"""
Gymnasium-compatible Dino Run environment.

Bug fixes over original dino_game.py:
  - Score is now tracked and incremented when clearing obstacles
  - Obstacle spawning uses min/max intervals (no impossible gaps)
  - Game speed increases over time (matches real Chrome Dino)
  - Jump penalty only applies when actually jumping from the ground
  - Graceful quit via pygame.QUIT and ESC key
  - Headless mode for fast training (no window, uncapped FPS)
  - Supports both pixel and feature-vector observations
"""
import pygame
import random
import numpy as np
import cv2
import sys

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, GROUND_Y,
    DINO_X, DINO_WIDTH, DINO_HEIGHT, JUMP_VELOCITY, GRAVITY,
    OBSTACLE_MIN_WIDTH, OBSTACLE_MAX_WIDTH,
    OBSTACLE_MIN_HEIGHT, OBSTACLE_MAX_HEIGHT,
    MIN_SPAWN_INTERVAL, MAX_SPAWN_INTERVAL,
    INITIAL_GAME_SPEED, MAX_GAME_SPEED,
    SPEED_INCREMENT, SPEED_INCREMENT_INTERVAL,
    REWARD_ALIVE, REWARD_OBSTACLE_CLEARED,
    REWARD_DEATH, REWARD_JUMP_PENALTY,
    FRAME_SIZE, FRAME_STACK, N_ACTIONS,
)


class DinoEnv:
    """
    Chrome Dino Run environment for Reinforcement Learning.

    Observation modes:
        'pixels'   : 4-stacked 84x84 grayscale frames → shape (4, 84, 84)
        'features' : Normalized feature vector          → shape (6,)

    Gymnasium-style API:
        reset()       → (observation, info)
        step(action)  → (observation, reward, terminated, truncated, info)
    """

    def __init__(self, headless=False, obs_mode='pixels',
                 render_fps=30, max_steps=10000):
        """
        Args:
            headless:   If True, runs without a visible window (fastest).
            obs_mode:   'pixels' for CNN input, 'features' for MLP input.
            render_fps: Frame cap when rendering. Set to 0 for uncapped.
            max_steps:  Maximum steps per episode before truncation.
        """
        pygame.init()
        self.headless = headless
        self.obs_mode = obs_mode
        self.render_fps = render_fps
        self.max_steps = max_steps
        self.n_actions = N_ACTIONS

        # Screen setup — off-screen surface for headless, window otherwise
        if headless:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Dino Run - RL Agent")

        self.clock = pygame.time.Clock()

        # Font for in-game score display
        if not headless:
            try:
                self.font = pygame.font.SysFont('consolas', 18)
            except Exception:
                self.font = pygame.font.Font(None, 22)

        # Initialize game state
        self._init_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        """Reset environment to initial state and return first observation."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._init_state()

        # Build initial frame stack (all identical at start)
        if self.obs_mode == 'pixels':
            self._render_frame()
            frame = self._capture_frame()
            self.frame_stack = [frame.copy() for _ in range(FRAME_STACK)]
        else:
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        """
        Execute one environment step.

        Args:
            action: 0 = Do Nothing, 1 = Jump

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        self.steps += 1
        reward = REWARD_ALIVE  # Base survival reward

        # --- Handle pygame events (graceful quit) ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                sys.exit()

        # --- 1. Dino Physics ---
        on_ground = self.dino_y >= GROUND_Y - DINO_HEIGHT
        jumped = False

        if action == 1 and on_ground:
            self.dino_vel_y = JUMP_VELOCITY
            jumped = True

        self.dino_vel_y += GRAVITY
        self.dino_y += self.dino_vel_y

        # Floor collision
        if self.dino_y >= GROUND_Y - DINO_HEIGHT:
            self.dino_y = GROUND_Y - DINO_HEIGHT
            self.dino_vel_y = 0

        # --- 2. Obstacle Spawning (with min/max spacing) ---
        self.frames_since_spawn += 1
        if self.frames_since_spawn >= self.next_spawn_at:
            obs_w = random.randint(OBSTACLE_MIN_WIDTH, OBSTACLE_MAX_WIDTH)
            obs_h = random.randint(OBSTACLE_MIN_HEIGHT, OBSTACLE_MAX_HEIGHT)
            self.obstacles.append({
                'x': float(SCREEN_WIDTH),
                'y': float(GROUND_Y - obs_h),
                'w': obs_w,
                'h': obs_h,
                'passed': False,
            })
            self.frames_since_spawn = 0
            # Scale spawn intervals with speed (faster ⇒ tighter, but always fair)
            speed_ratio = self.game_speed / INITIAL_GAME_SPEED
            min_iv = max(int(MIN_SPAWN_INTERVAL / speed_ratio), 15)
            max_iv = max(int(MAX_SPAWN_INTERVAL / speed_ratio), min_iv + 10)
            self.next_spawn_at = random.randint(min_iv, max_iv)

        # --- 3. Move Obstacles & Check Collisions ---
        dino_rect = pygame.Rect(DINO_X, int(self.dino_y), DINO_WIDTH, DINO_HEIGHT)

        for obs in self.obstacles:
            obs['x'] -= self.game_speed
            obs_rect = pygame.Rect(int(obs['x']), int(obs['y']), obs['w'], obs['h'])

            # Collision detection
            if obs_rect.colliderect(dino_rect):
                self.game_over = True

            # Obstacle-cleared bonus (dino's right edge passes obstacle's right edge)
            if not obs['passed'] and obs['x'] + obs['w'] < DINO_X:
                obs['passed'] = True
                reward += REWARD_OBSTACLE_CLEARED
                self.score += 1

        # Remove off-screen obstacles
        self.obstacles = [o for o in self.obstacles if o['x'] > -60]

        # --- 4. Speed Progression ---
        if self.steps % SPEED_INCREMENT_INTERVAL == 0:
            self.game_speed = min(self.game_speed + SPEED_INCREMENT, MAX_GAME_SPEED)

        # --- 5. Reward Adjustments ---
        if jumped:
            reward += REWARD_JUMP_PENALTY  # Only penalize actual jumps from ground

        if self.game_over:
            reward = REWARD_DEATH

        # --- 6. Render & Build Observation ---
        self._render_frame()

        if self.obs_mode == 'pixels':
            frame = self._capture_frame()
            self.frame_stack.pop(0)
            self.frame_stack.append(frame)

        terminated = self.game_over
        truncated = self.steps >= self.max_steps

        # FPS control (only in visible mode)
        if not self.headless and self.render_fps > 0:
            self.clock.tick(self.render_fps)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_state(self):
        """Reset all internal game state variables."""
        self.dino_y = GROUND_Y - DINO_HEIGHT
        self.dino_vel_y = 0

        self.obstacles = []
        self.frames_since_spawn = 0
        self.next_spawn_at = random.randint(MIN_SPAWN_INTERVAL, MAX_SPAWN_INTERVAL)

        self.game_speed = INITIAL_GAME_SPEED
        self.score = 0
        self.steps = 0
        self.game_over = False

        if self.obs_mode == 'pixels':
            self.frame_stack = []

    def _render_frame(self):
        """Draw the current game state onto the screen surface."""
        self.screen.fill((255, 255, 255))  # White background

        # Ground line
        pygame.draw.line(self.screen, (100, 100, 100),
                         (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)

        # Dino (dark green rectangle)
        dino_rect = pygame.Rect(DINO_X, int(self.dino_y), DINO_WIDTH, DINO_HEIGHT)
        pygame.draw.rect(self.screen, (34, 139, 34), dino_rect)

        # Obstacles (dark red rectangles)
        for obs in self.obstacles:
            obs_rect = pygame.Rect(int(obs['x']), int(obs['y']), obs['w'], obs['h'])
            pygame.draw.rect(self.screen, (178, 34, 34), obs_rect)

        # Score overlay (visible mode only)
        if not self.headless and hasattr(self, 'font'):
            score_surf = self.font.render(
                f"Score: {self.score}  Speed: {self.game_speed:.1f}",
                True, (50, 50, 50))
            self.screen.blit(score_surf, (10, 10))

        if not self.headless:
            pygame.display.flip()

    def _capture_frame(self):
        """Capture the screen as an 84×84 grayscale numpy array in [0, 1]."""
        pixels = pygame.surfarray.array3d(self.screen)        # (W, H, 3)
        pixels = pixels.transpose([1, 0, 2])                  # (H, W, 3)
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)       # (H, W)
        resized = cv2.resize(gray, (FRAME_SIZE, FRAME_SIZE))  # (84, 84)
        return np.ascontiguousarray(resized, dtype=np.float32) / 255.0

    def _get_obs(self):
        """Return current observation based on obs_mode."""
        if self.obs_mode == 'pixels':
            return np.array(self.frame_stack, dtype=np.float32)
        return self._get_feature_vector()

    def _get_feature_vector(self):
        """
        Compact feature vector (6 values, all normalized to ~[0, 1]):
            [dino_y_norm, vel_norm, dist_to_obstacle, obs_width, obs_height, speed]
        """
        ground_top = GROUND_Y - DINO_HEIGHT
        dino_y_norm = 1.0 - (self.dino_y / ground_top) if ground_top != 0 else 0.0
        vel_norm = self.dino_vel_y / abs(JUMP_VELOCITY) if JUMP_VELOCITY != 0 else 0.0

        # Find nearest obstacle ahead of the dino
        ahead = [o for o in self.obstacles if o['x'] + o['w'] > DINO_X]
        if ahead:
            nearest = min(ahead, key=lambda o: o['x'])
            dist_norm = (nearest['x'] - DINO_X) / SCREEN_WIDTH
            width_norm = nearest['w'] / OBSTACLE_MAX_WIDTH
            height_norm = nearest['h'] / OBSTACLE_MAX_HEIGHT
        else:
            dist_norm = 1.0
            width_norm = 0.0
            height_norm = 0.0

        speed_norm = self.game_speed / MAX_GAME_SPEED

        return np.array([
            dino_y_norm, vel_norm, dist_norm,
            width_norm, height_norm, speed_norm
        ], dtype=np.float32)

    def _get_info(self):
        """Auxiliary info dictionary."""
        return {
            'score': self.score,
            'speed': self.game_speed,
            'steps': self.steps,
        }
