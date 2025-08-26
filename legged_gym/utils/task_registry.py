# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Tuple
from legged_gym.utils.helpers import class_to_dict


class TaskRegistry:
    """Registry for robot tasks and their configurations"""
    
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, task_name: str, env_cfg, train_cfg):
        """Register a new task with its configurations"""
        self.env_cfgs[task_name] = env_cfg
        self.train_cfgs[task_name] = train_cfg
    
    def get_cfgs(self, name: str) -> Tuple:
        """Get environment and training configurations for a task"""
        return self.env_cfgs[name], self.train_cfgs[name]
    
    def make_env(self, name: str, args=None, env_cfg=None) -> Tuple:
        """Create environment instance"""
        if env_cfg is None:
            env_cfg = self.env_cfgs[name]
        
        # Import the environment class
        if name == "g1":
            from legged_gym.envs.g1.g1 import G1
            env_class = G1
        elif name == "h1":
            from legged_gym.envs.h1.h1 import H1
            env_class = H1
        else:
            raise ValueError(f"Unknown task: {name}")
        
        # Override config parameters from args if provided
        if args is not None:
            if hasattr(args, 'num_envs') and args.num_envs is not None:
                env_cfg.env.num_envs = args.num_envs
            if hasattr(args, 'seed') and args.seed is not None:
                env_cfg.seed = args.seed
        
        # Create simulation parameters (simplified)
        sim_params = self._create_sim_params(env_cfg)
        
        # Create environment
        env = env_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine="physx",  # Using PhysX
            sim_device="cuda:0",     # GPU device
            headless=getattr(args, 'headless', False) if args else False
        )
        
        return env, env_cfg
    
    def make_alg_runner(self, env, name: str, args=None, train_cfg=None):
        """Create algorithm runner (PPO)"""
        if train_cfg is None:
            train_cfg = self.train_cfgs[name]
        
        # Override config from args
        if args is not None:
            if hasattr(args, 'max_iterations') and args.max_iterations is not None:
                train_cfg.runner.max_iterations = args.max_iterations
            if hasattr(args, 'resume') and args.resume:
                train_cfg.runner.resume = True
            if hasattr(args, 'checkpoint') and args.checkpoint != -1:
                train_cfg.runner.checkpoint = args.checkpoint
            if hasattr(args, 'run_name') and args.run_name:
                train_cfg.runner.run_name = args.run_name
        
        # Import RSL-RL components
        try:
            from rsl_rl.runners import OnPolicyRunner
            from rsl_rl.algorithms import PPO
        except ImportError:
            print("Error: RSL-RL not found. Please install rsl_rl package.")
            raise
        
        # Create PPO algorithm
        ppo_runner = OnPolicyRunner(env, train_cfg, device=env.device)
        
        return ppo_runner, train_cfg
    
    def _create_sim_params(self, env_cfg):
        """Create simulation parameters from environment config"""
        # This is a simplified version - in practice you'd use Isaac Gym's parameter structures
        class SimParams:
            def __init__(self):
                self.dt = env_cfg.sim.dt
                self.substeps = env_cfg.sim.substeps
                self.gravity = env_cfg.sim.gravity
                self.up_axis = env_cfg.sim.up_axis
        
        return SimParams()


# Global task registry instance
task_registry = TaskRegistry() 