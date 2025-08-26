# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import statistics
from collections import deque
import torch
import numpy as np

from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import isaacgym


def play(args):
    """Play/evaluate trained policy and optionally export to ONNX"""
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Override some config parameters for evaluation
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)  # Reduce for visualization
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False  # Test on all terrain types
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    terrain_crossing_success = []
    survival_times = []
    velocity_tracking_errors = []
    
    # Joint position errors for style evaluation
    upper_body_errors = []
    lower_body_errors = []
    
    # Define target joint positions for style evaluation
    if args.task == "g1":
        target_shoulder_pitch = 0.0  # Target for shoulder pitch joints
        target_elbow = 0.0           # Target for elbow joints
        target_hip_pitch = -0.3      # Target for hip pitch joints
        target_ankle_pitch = -0.3    # Target for ankle pitch joints
    elif args.task == "h1":
        target_shoulder_pitch = 0.0
        target_elbow = 0.0
        target_hip_pitch = -0.4
        target_ankle_pitch = -0.4
    
    print(f"Evaluating {args.task} policy...")
    print(f"Environment: {env_cfg.env.num_envs} robots")
    print(f"Terrain: {env_cfg.terrain.num_rows}x{env_cfg.terrain.num_cols} grid")
    
    # Run evaluation
    for i in range(1000):  # Evaluation steps
        actions = policy(obs.detach())
        obs, rewards, dones, infos = env.step(actions.detach())
        
        # Collect metrics
        if i % 100 == 0:
            print(f"Step {i}: Mean reward = {torch.mean(rewards).item():.3f}")
            
            # Calculate velocity tracking error
            cmd_vel = torch.norm(env.commands[:, :2], dim=1)  # Command velocity magnitude
            actual_vel = torch.norm(env.base_lin_vel[:, :2], dim=1)  # Actual velocity magnitude
            vel_error = torch.abs(cmd_vel - actual_vel)
            velocity_tracking_errors.extend(vel_error.cpu().numpy())
            
            # Calculate joint position errors for style evaluation
            if hasattr(env, 'dof_pos') and hasattr(env, 'dof_names'):
                # Upper body style evaluation (shoulder and elbow joints)
                shoulder_pitch_errors = []
                elbow_errors = []
                
                for joint_name in env.dof_names:
                    joint_idx = env.dof_names.index(joint_name)
                    if 'shoulder_pitch' in joint_name:
                        error = torch.abs(env.dof_pos[:, joint_idx] - target_shoulder_pitch)
                        shoulder_pitch_errors.extend(error.cpu().numpy())
                    elif 'elbow' in joint_name:
                        error = torch.abs(env.dof_pos[:, joint_idx] - target_elbow)
                        elbow_errors.extend(error.cpu().numpy())
                
                if shoulder_pitch_errors or elbow_errors:
                    upper_body_error = np.mean(shoulder_pitch_errors + elbow_errors)
                    upper_body_errors.append(upper_body_error)
                
                # Lower body style evaluation (hip and ankle pitch joints)
                hip_pitch_errors = []
                ankle_pitch_errors = []
                
                for joint_name in env.dof_names:
                    joint_idx = env.dof_names.index(joint_name)
                    if 'hip_pitch' in joint_name:
                        error = torch.abs(env.dof_pos[:, joint_idx] - target_hip_pitch)
                        hip_pitch_errors.extend(error.cpu().numpy())
                    elif 'ankle_pitch' in joint_name:
                        error = torch.abs(env.dof_pos[:, joint_idx] - target_ankle_pitch)
                        ankle_pitch_errors.extend(error.cpu().numpy())
                
                if hip_pitch_errors or ankle_pitch_errors:
                    lower_body_error = np.mean(hip_pitch_errors + ankle_pitch_errors)
                    lower_body_errors.append(lower_body_error)
        
        # Track episode completions
        if torch.any(dones):
            done_envs = torch.where(dones)[0]
            for env_idx in done_envs:
                episode_length = env.episode_length_buf[env_idx].item()
                episode_reward = torch.sum(env.episode_rewards[env_idx]).item()
                
                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
                
                # Calculate survival time
                survival_time = episode_length * env.dt
                survival_times.append(survival_time)
                
                # Check terrain crossing success
                distance_traveled = torch.norm(env.root_states[env_idx, :2] - env.env_origins[env_idx, :2]).item()
                terrain_block_size = env_cfg.terrain.terrain_length
                crossing_success = distance_traveled > (terrain_block_size / 2)
                terrain_crossing_success.append(crossing_success)
    
    # Calculate final metrics
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    if episode_rewards:
        print(f"Average Episode Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    
    if survival_times:
        avg_survival = np.mean(survival_times)
        print(f"Average Survival Time: {avg_survival:.3f}s")
        print(f"Survival Success (>10s): {np.mean([t > 10.0 for t in survival_times]):.1%}")
    
    if velocity_tracking_errors:
        avg_vel_error = np.mean(velocity_tracking_errors)
        print(f"Average Velocity Tracking Error: {avg_vel_error:.3f} m/s")
        
        # Check success criteria
        low_speed_success = avg_vel_error < 0.25
        high_speed_success = avg_vel_error < 0.5
        print(f"Low-speed criteria (error < 0.25 m/s): {'✓' if low_speed_success else '✗'}")
        print(f"High-speed criteria (error < 0.5 m/s): {'✓' if high_speed_success else '✗'}")
    
    if terrain_crossing_success:
        crossing_rate = np.mean(terrain_crossing_success)
        print(f"Terrain Crossing Success Rate: {crossing_rate:.1%}")
    
    # Style evaluation
    print("\nSTYLE EVALUATION:")
    if upper_body_errors:
        avg_upper_error = np.mean(upper_body_errors)
        print(f"Upper Body Style Error: {avg_upper_error:.3f} rad")
        upper_style_success = avg_upper_error < 0.5
        print(f"Upper Body Style Success (error < 0.5 rad): {'✓' if upper_style_success else '✗'}")
    
    if lower_body_errors:
        avg_lower_error = np.mean(lower_body_errors)
        print(f"Lower Body Style Error: {avg_lower_error:.3f} rad")
        lower_style_success = avg_lower_error < 0.8
        print(f"Lower Body Style Success (error < 0.8 rad): {'✓' if lower_style_success else '✗'}")
    
    # Export policy
    if args.export_policy:
        print(f"\nExporting policy to ONNX format...")
        export_dir = os.path.join("logs", "exported")
        os.makedirs(export_dir, exist_ok=True)
        
        export_path = os.path.join(export_dir, "policy.pt")
        export_policy_as_jit(ppo_runner.alg.actor_critic, export_path)
        print(f"Policy exported to: {export_path}")
        
        # Verify export
        try:
            exported_policy = torch.jit.load(export_path)
            test_obs = torch.randn(1, env_cfg.env.num_observations)
            test_output = exported_policy(test_obs)
            print(f"Export verification successful. Output shape: {test_output.shape}")
        except Exception as e:
            print(f"Export verification failed: {e}")


if __name__ == '__main__':
    args = get_args()
    play(args) 