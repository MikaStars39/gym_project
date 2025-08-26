# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os

# Set environment variables for Isaac Gym
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs')

# Make sure Isaac Gym can find our custom environments
if LEGGED_GYM_ENVS_DIR not in os.sys.path:
    os.sys.path.append(LEGGED_GYM_ENVS_DIR)

from legged_gym.envs import * 