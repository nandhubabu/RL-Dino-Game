"""
Evaluation & visualization script for the trained Dino RL agent.

Features:
    - Auto-detects the best available checkpoint
    - Live HUD overlay showing Q-values, action, score, and game stats
    - CLI for checkpoint selection, FPS control, and game count

Usage:
    python play.py                                          # Auto-detect best model
    python play.py --checkpoint checkpoints/best_model.pth  # Specific checkpoint
    python play.py --fps 60                                 # Faster playback
    python play.py --games 10                               # Play exactly 10 games
"""
import argparse
import glob
import os
import sys

import numpy as np
import pygame

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, CHECKPOINT_DIR, N_ACTIONS,
)
from env.dino_env import DinoEnv
from agents.dqn_agent import DQNAgent


def parse_args():
    p = argparse.ArgumentParser(description='Watch the trained AI play Dino Run')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to model checkpoint (auto-detects if omitted)')
    p.add_argument('--obs', choices=['pixels', 'features'], default='pixels',
                   help='Observation mode (must match training)')
    p.add_argument('--arch', choices=['dqn', 'dueling'], default='dueling',
                   help='Architecture (must match training)')
    p.add_argument('--fps', type=int, default=30,
                   help='Playback FPS (default: 30)')
    p.add_argument('--games', type=int, default=0,
                   help='Number of games (0 = play forever)')
    return p.parse_args()


# ------------------------------------------------------------------
# Checkpoint auto-detection
# ------------------------------------------------------------------

def auto_detect_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    """Find the best available checkpoint automatically.

    Search order: best_model.pth → latest.pth → highest-numbered
    checkpoint → any .pth in the root directory (legacy files).
    """
    for name in ('best_model.pth', 'latest.pth'):
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path):
            return path

    pth_files = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not pth_files:
        pth_files = glob.glob('*.pth')  # Legacy root-level files

    return sorted(pth_files)[-1] if pth_files else None


# ------------------------------------------------------------------
# HUD Overlay
# ------------------------------------------------------------------

def draw_hud(screen, q_values, action, score, high_score, game_speed, episode):
    """
    Render a translucent heads-up display on the game screen showing:
        • Top bar  — score, high score, speed, game number, current action
        • Bottom-right panel — Q-value bar chart for Run vs. Jump
    """
    font_sm = pygame.font.SysFont('consolas', 14)
    font_lg = pygame.font.SysFont('consolas', 18, bold=True)

    # ---- Top bar ----
    bar = pygame.Surface((SCREEN_WIDTH, 35), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 160))
    screen.blit(bar, (0, 0))

    screen.blit(font_lg.render(f'Score: {score}', True, (0, 255, 128)), (10, 8))
    screen.blit(font_sm.render(f'Best: {high_score}', True, (200, 200, 200)), (160, 10))
    screen.blit(font_sm.render(f'Speed: {game_speed:.1f}', True, (200, 200, 200)), (280, 10))
    screen.blit(font_sm.render(f'Game #{episode}', True, (200, 200, 200)), (420, 10))

    action_name = 'JUMP ^' if action == 1 else 'RUN  >'
    action_color = (255, 200, 50) if action == 1 else (100, 200, 255)
    screen.blit(font_lg.render(action_name, True, action_color),
                (SCREEN_WIDTH - 110, 8))

    # ---- Q-Value bar chart (bottom-right) ----
    if q_values is not None:
        pw, ph = 160, 60
        px = SCREEN_WIDTH - pw - 10
        py = SCREEN_HEIGHT - ph - 10

        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 140))
        screen.blit(panel, (px, py))

        screen.blit(font_sm.render('Q-Values', True, (200, 200, 200)),
                    (px + 5, py + 3))

        labels = ['Run ', 'Jump']
        colors = [(100, 200, 255), (255, 200, 50)]
        q_min, q_max = min(q_values), max(q_values)
        q_range = max(q_max - q_min, 1e-3)

        for i, (lbl, clr) in enumerate(zip(labels, colors)):
            y = py + 22 + i * 18
            screen.blit(font_sm.render(lbl, True, (180, 180, 180)),
                        (px + 5, y))

            bar_w = max(int(((q_values[i] - q_min) / q_range) * 80), 2)
            best = (i == int(np.argmax(q_values)))
            pygame.draw.rect(screen, clr if best else (80, 80, 80),
                             (px + 50, y + 2, bar_w, 12))
            screen.blit(font_sm.render(f'{q_values[i]:.1f}', True, (220, 220, 220)),
                        (px + 50 + bar_w + 4, y))


# ------------------------------------------------------------------
# Play loop
# ------------------------------------------------------------------

def play(args):
    # ---- Find checkpoint ----
    ckpt = args.checkpoint or auto_detect_checkpoint()
    if ckpt is None:
        print('No checkpoint found!')
        print('  Train a model first : python train.py')
        print('  Or specify manually : python play.py --checkpoint <path>')
        sys.exit(1)

    print('=' * 60)
    print('  Dino RL - AI Playback')
    print(f'  Checkpoint  : {ckpt}')
    print(f'  Architecture: {args.arch}')
    print(f'  Obs Mode    : {args.obs}')
    print(f'  FPS         : {args.fps}')
    print('  Press ESC or close the window to quit')
    print('=' * 60)

    # ---- Setup ----
    env = DinoEnv(headless=False, obs_mode=args.obs, render_fps=args.fps)
    agent = DQNAgent(obs_mode=args.obs, architecture=args.arch)
    agent.load_model_only(ckpt)

    high_score = 0
    episode = 0

    try:
        while True:
            episode += 1
            if 0 < args.games < episode:
                break

            obs, info = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action, q_values = agent.select_action(obs, evaluate=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Draw HUD over the game frame
                draw_hud(env.screen, q_values, action,
                         info['score'], high_score, info['speed'], episode)
                pygame.display.flip()

            high_score = max(high_score, info['score'])
            print(f'  Game {episode:3d} | Score: {info["score"]:4d} | '
                  f'Best: {high_score:4d} | Reward: {total_reward:.1f}')

    except (KeyboardInterrupt, SystemExit):
        pass

    finally:
        env.close()
        print(f'\n  High Score   : {high_score}')
        print(f'  Games Played : {episode}')


if __name__ == '__main__':
    play(parse_args())