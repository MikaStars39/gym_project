# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import torch


def train(args):
    """Training function for humanoid robots on complex terrain"""
    
    # Parse environment name
    env_name = args.task
    print(f"Training {env_name} on complex terrain...")
    
    # Create environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # Configure speed ranges based on task requirements
    if "high_speed" in args.run_name.lower():
        print("Configuring for high-speed locomotion [1.5-2.0 m/s]...")
        env.cfg.commands.ranges.lin_vel_x = [1.5, 2.0]
        env.cfg.commands.ranges.lin_vel_y = [0.0, 0.5]
        env.cfg.commands.ranges.ang_vel_yaw = [0.0, 0.5]
        # Increase training iterations for high-speed
        train_cfg.runner.max_iterations = 3000
    else:
        print("Configuring for low-speed locomotion [0.0-1.0 m/s]...")
        env.cfg.commands.ranges.lin_vel_x = [0.0, 1.0]
        env.cfg.commands.ranges.lin_vel_y = [0.0, 1.0]
        env.cfg.commands.ranges.ang_vel_yaw = [0.0, 0.5]
    
    # Setup terrain curriculum
    if hasattr(env.cfg.terrain, 'curriculum') and env.cfg.terrain.curriculum:
        print("Complex terrain curriculum enabled:")
        print(f"  - Terrain proportions: {env.cfg.terrain.terrain_proportions}")
        print(f"  - Difficulty levels: {env.cfg.terrain.num_rows}")
        print(f"  - Terrain types: {env.cfg.terrain.num_cols}")
    
    # Configure logging
    log_dir = train_cfg.runner.experiment_name
    if args.run_name:
        log_dir += f"_{args.run_name}"
    log_dir += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger = Logger(log_dir)
    
    # Setup training
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, 
                    init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args) 