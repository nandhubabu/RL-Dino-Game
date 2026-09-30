"""
Training script for the Dino RL agent.

Usage:
    python train.py                                        # Default: Dueling DDQN, pixel mode
    python train.py --obs features                         # Fast MLP training (~2 min)
    python train.py --episodes 2000                        # Train for 2000 episodes
    python train.py --resume checkpoints/latest.pth        # Resume interrupted run
    python train.py --no-render                            # Headless (fastest)
    python train.py --arch dqn --no-double                 # Vanilla DQN baseline

After training, view metrics:
    tensorboard --logdir runs
"""
import argparse
import os
import time
import numpy as np
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from config import (
    NUM_EPISODES, CHECKPOINT_INTERVAL, CHECKPOINT_DIR, LOG_DIR,
)
from env.dino_env import DinoEnv
from agents.dqn_agent import DQNAgent


def parse_args():
    p = argparse.ArgumentParser(description='Train a DQN agent to play Dino Run')
    p.add_argument('--episodes', type=int, default=NUM_EPISODES,
                   help=f'Number of training episodes (default: {NUM_EPISODES})')
    p.add_argument('--obs', choices=['pixels', 'features'], default='pixels',
                   help='Observation mode: pixels (CNN) or features (MLP)')
    p.add_argument('--arch', choices=['dqn', 'dueling'], default='dueling',
                   help='Network architecture (default: dueling)')
    p.add_argument('--no-double', action='store_true',
                   help='Disable Double DQN (use standard DQN targets)')
    p.add_argument('--no-render', action='store_true',
                   help='Headless mode — no window, fastest training')
    p.add_argument('--resume', type=str, default=None,
                   help='Path to checkpoint file to resume training')
    p.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR,
                   help=f'Checkpoint directory (default: {CHECKPOINT_DIR})')
    p.add_argument('--log-dir', type=str, default=LOG_DIR,
                   help=f'TensorBoard log directory (default: {LOG_DIR})')
    return p.parse_args()


def train(args):
    # ---- Setup --------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{args.arch}_{args.obs}_{timestamp}'
    writer = SummaryWriter(os.path.join(args.log_dir, run_name))

    env = DinoEnv(
        headless=args.no_render,
        obs_mode=args.obs,
        render_fps=0 if args.no_render else 30,
    )

    agent = DQNAgent(
        obs_mode=args.obs,
        architecture=args.arch,
        double_dqn=not args.no_double,
    )

    algo = f"{'Double ' if not args.no_double else ''}{'Dueling ' if args.arch == 'dueling' else ''}DQN"
    print('=' * 60)
    print('  Dino RL Training')
    print(f'  Algorithm  : {algo}')
    print(f'  Obs Mode   : {args.obs}')
    print(f'  Device     : {agent.device}')
    print(f'  Episodes   : {args.episodes}')
    print(f'  Headless   : {args.no_render}')
    print(f'  TensorBoard: tensorboard --logdir {args.log_dir}')
    print('=' * 60)

    # Resume from checkpoint
    start_episode = 0
    best_reward = float('-inf')
    best_game_score = 0
    if args.resume:
        start_episode, best_reward = agent.load_checkpoint(args.resume)
        print(f'  Resumed from episode {start_episode}, best reward: {best_reward:.1f}')

    # ---- Training Loop ------------------------------------------------
    recent_scores = []
    total_steps = 0

    try:
        for episode in range(start_episode, args.episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            loss_count = 0
            episode_steps = 0
            t_start = time.time()

            while True:
                # Select action
                action, q_values = agent.select_action(obs)

                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Store & learn
                agent.store_transition(obs, action, next_obs, reward, done)
                loss = agent.optimize()
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1

                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                if done:
                    break

            # ---- Logging ----
            epsilon = agent.get_epsilon()
            avg_loss = episode_loss / max(loss_count, 1)
            fps = episode_steps / max(time.time() - t_start, 1e-3)

            recent_scores.append(info['score'])
            if len(recent_scores) > 100:
                recent_scores.pop(0)
            avg_score = np.mean(recent_scores)

            # Console output
            print(f'Ep {episode:4d} | '
                  f'Score: {info["score"]:3d} | '
                  f'Avg100: {avg_score:6.1f} | '
                  f'Reward: {episode_reward:8.1f} | '
                  f'e: {epsilon:.3f} | '
                  f'Loss: {avg_loss:.4f} | '
                  f'Steps: {episode_steps:5d} | '
                  f'FPS: {fps:.0f}')

            # TensorBoard scalars
            writer.add_scalar('Episode/Score', info['score'], episode)
            writer.add_scalar('Episode/Reward', episode_reward, episode)
            writer.add_scalar('Episode/AvgScore_100', avg_score, episode)
            writer.add_scalar('Episode/Epsilon', epsilon, episode)
            writer.add_scalar('Episode/AvgLoss', avg_loss, episode)
            writer.add_scalar('Episode/Steps', episode_steps, episode)
            writer.add_scalar('Episode/GameSpeed', info['speed'], episode)
            writer.add_scalar('Training/FPS', fps, episode)
            writer.add_scalar('Training/TotalSteps', total_steps, episode)
            writer.add_scalar('Training/BufferSize', len(agent.memory), episode)

            # ---- Checkpointing ----
            # Periodic snapshot
            if episode > 0 and episode % CHECKPOINT_INTERVAL == 0:
                path = os.path.join(args.checkpoint_dir,
                                    f'checkpoint_ep{episode}.pth')
                agent.save_checkpoint(path, episode, best_reward)
                print(f'  Saved checkpoint: {path}')

            # Best model (prioritizes obstacles cleared, then reward)
            is_new_best = (info['score'] > best_game_score) or \
                          (info['score'] == best_game_score and episode_reward > best_reward)
            if is_new_best:
                best_game_score = info['score']
                best_reward = episode_reward
                path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                agent.save_checkpoint(path, episode, best_reward)
                print(f'  New best! Score: {best_game_score:2d}, Reward: {best_reward:6.1f} -> {path}')

            # Latest (always, for easy resume)
            agent.save_checkpoint(
                os.path.join(args.checkpoint_dir, 'latest.pth'),
                episode, best_reward)

    except KeyboardInterrupt:
        print(f'\n{"=" * 60}')
        print(f'  Training interrupted at episode {episode}')
        path = os.path.join(args.checkpoint_dir,
                            f'interrupted_ep{episode}.pth')
        agent.save_checkpoint(path, episode, best_reward)
        print(f'  Saved: {path}')
        print('=' * 60)

    finally:
        writer.close()
        env.close()
        print(f'\n  Best score achieved: {best_game_score} (Reward: {best_reward:.1f})')
        print(f'  TensorBoard logs  : {args.log_dir}/{run_name}')
        print(f'  To view: tensorboard --logdir {args.log_dir}')


if __name__ == '__main__':
    train(parse_args())
