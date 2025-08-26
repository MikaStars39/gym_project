# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
import argparse
from typing import Union
import os


def class_to_dict(obj) -> dict:
    """Convert class attributes to dictionary"""
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key, val in obj.__dict__.items():
        if key.startswith("_"):
            continue
        element = []
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    """Update class attributes from dictionary"""
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def get_load_path(root, load_run=-1, checkpoint=-1):
    """Get path to load checkpoint"""
    if isinstance(load_run, str):
        load_run = int(load_run)
    if isinstance(checkpoint, str):
        checkpoint = int(checkpoint)
        
    if load_run == -1:
        load_run = sorted([int(dir_name) for dir_name in os.listdir(root) if dir_name.isdigit()])[-1]
    
    run_path = os.path.join(root, str(load_run))
    
    if checkpoint == -1:
        checkpoint_files = [f for f in os.listdir(run_path) if f.startswith('model_') and f.endswith('.pt')]
        if checkpoint_files:
            checkpoint = max([int(f.split('_')[1].split('.')[0]) for f in checkpoint_files])
        else:
            checkpoint = 0
    
    checkpoint_path = os.path.join(run_path, f'model_{checkpoint}.pt')
    return checkpoint_path


def export_policy_as_jit(actor_critic, path):
    """Export policy as JIT traced model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Create dummy input
    dummy_input = torch.randn(1, actor_critic.num_obs).to(actor_critic.actor.parameters().__next__().device)
    
    # Trace the model
    traced_script_module = torch.jit.trace(actor_critic.act_inference, dummy_input)
    traced_script_module.save(path)


def get_args():
    """Parse command line arguments"""
    custom_parameters = [
        {"name": "--task", "type": str, "default": "g1", "help": "Task to run (g1, h1)"},
        {"name": "--run_name", "type": str, "default": "", "help": "Run name for logging"},
        {"name": "--resume", "action": "store_true", "help": "Resume training from checkpoint"},
        {"name": "--checkpoint", "type": int, "default": -1, "help": "Checkpoint to load (-1 for latest)"},
        {"name": "--num_envs", "type": int, "help": "Number of environments"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "help": "Maximum training iterations"},
        {"name": "--export_policy", "action": "store_true", "help": "Export policy as JIT"},
        {"name": "--headless", "action": "store_true", "help": "Run headless (no GUI)"},
        {"name": "--play", "action": "store_true", "help": "Play mode (no training)"},
    ]
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Isaac Gym Humanoid Training')
    for param in custom_parameters:
        parser.add_argument(param["name"], **{k: v for k, v in param.items() if k != "name"})
    
    args = parser.parse_args()
    return args


def torch_rand_sqrt_float(lower, upper, shape, device):
    """Generate random numbers with square root distribution"""
    r = (upper - lower) * torch.sqrt(torch.rand(shape, device=device)) + lower
    return r 