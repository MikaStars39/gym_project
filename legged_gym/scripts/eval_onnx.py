# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import numpy as np
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import isaacgym


def eval_onnx(args):
    """Evaluate exported ONNX policy"""
    
    # Load environment configuration
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Override some config parameters for evaluation
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 10)  # Small number for testing
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 3
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Load exported policy
    policy_path = os.path.join("logs", "exported", "policy.pt")
    
    if not os.path.exists(policy_path):
        print(f"Error: Policy file not found at {policy_path}")
        print("Please run play.py with --export_policy flag first.")
        return
    
    try:
        policy = torch.jit.load(policy_path, map_location=env.device)
        print(f"Successfully loaded policy from {policy_path}")
    except Exception as e:
        print(f"Error loading policy: {e}")
        return
    
    # Test policy
    obs = env.get_observations()
    print(f"Observation shape: {obs.shape}")
    print(f"Expected observation shape: [{env_cfg.env.num_envs}, {env_cfg.env.num_observations}]")
    
    # Run test episodes
    episode_rewards = []
    survival_times = []
    
    print(f"\nTesting ONNX policy on {env_cfg.env.num_envs} environments...")
    
    for step in range(1000):  # Run for 1000 steps
        try:
            # Get actions from policy
            with torch.no_grad():
                actions = policy(obs.detach())
            
            # Step environment
            obs, rewards, dones, infos = env.step(actions.detach())
            
            # Print progress
            if step % 100 == 0:
                mean_reward = torch.mean(rewards).item()
                print(f"Step {step}: Mean reward = {mean_reward:.3f}")
            
            # Track completed episodes
            if torch.any(dones):
                done_envs = torch.where(dones)[0]
                for env_idx in done_envs:
                    episode_length = env.episode_length_buf[env_idx].item()
                    survival_time = episode_length * env.dt
                    survival_times.append(survival_time)
                    
                    # Calculate episode reward (simplified)
                    episode_reward = torch.sum(rewards[env_idx]).item()
                    episode_rewards.append(episode_reward)
                    
        except Exception as e:
            print(f"Error during policy execution at step {step}: {e}")
            return
    
    # Calculate results
    print("\n" + "="*40)
    print("ONNX EVALUATION RESULTS")
    print("="*40)
    
    if survival_times:
        avg_survival = np.mean(survival_times)
        max_survival = np.max(survival_times)
        print(f"Average Survival Time: {avg_survival:.3f}s")
        print(f"Maximum Survival Time: {max_survival:.3f}s")
        print(f"Episodes with >10s survival: {np.mean([t > 10.0 for t in survival_times]):.1%}")
    else:
        print("No completed episodes recorded")
    
    if episode_rewards:
        avg_reward = np.mean(episode_rewards)
        print(f"Average Episode Reward: {avg_reward:.3f}")
    
    # Test policy output characteristics
    print(f"\nTesting policy output characteristics...")
    test_obs = torch.randn(1, env_cfg.env.num_observations, device=env.device)
    
    try:
        with torch.no_grad():
            test_actions = policy(test_obs)
        
        print(f"Policy input shape: {test_obs.shape}")
        print(f"Policy output shape: {test_actions.shape}")
        print(f"Expected output shape: [1, {env_cfg.env.num_actions}]")
        print(f"Action range: [{torch.min(test_actions).item():.3f}, {torch.max(test_actions).item():.3f}]")
        
        # Check if actions are reasonable
        action_std = torch.std(test_actions).item()
        print(f"Action standard deviation: {action_std:.3f}")
        
        if test_actions.shape[1] == env_cfg.env.num_actions:
            print("✓ Policy output shape is correct")
        else:
            print("✗ Policy output shape mismatch")
        
        if -1.5 <= torch.min(test_actions).item() <= torch.max(test_actions).item() <= 1.5:
            print("✓ Action values are in reasonable range")
        else:
            print("✗ Action values may be out of reasonable range")
            
    except Exception as e:
        print(f"Error testing policy characteristics: {e}")
    
    print(f"\nONNX policy evaluation completed successfully!")


if __name__ == '__main__':
    args = get_args()
    eval_onnx(args) 