# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .g1_config import G1RoughCfg


class G1(LeggedRobot):
    """G1 humanoid robot environment for full-body locomotion on complex terrain"""
    
    cfg: G1RoughCfg
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """Initialize G1 environment"""
        
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        self.headless = headless
        
        # Extract joint indices for upper/lower body symmetry checking
        self.left_joints = []
        self.right_joints = []
        self.upper_body_joints = []
        self.lower_body_joints = []
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize additional buffers for humanoid-specific observations
        self._init_humanoid_buffers()
        
        # Extract joint mappings after calling parent init
        self._setup_joint_mappings()
        
    def _init_humanoid_buffers(self):
        """Initialize additional buffers for humanoid robot observations"""
        
        # Upper body orientation tracking
        self.upper_body_orientation_buf = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Joint symmetry tracking
        self.joint_symmetry_error_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Gravity vector in robot frame for balance
        self.gravity_vec_buf = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Additional terrain info
        self.terrain_levels_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.terrain_types_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        
    def _setup_joint_mappings(self):
        """Setup joint index mappings for symmetry constraints"""
        
        # Define joint name patterns for left/right symmetry
        left_joint_names = [
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint'
        ]
        
        right_joint_names = [
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint'
        ]
        
        # Upper body joint names (excluding legs)
        upper_body_joint_names = [
            'torso_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
            'left_shoulder_yaw_joint', 'left_elbow_joint', 'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
            'head_yaw_joint', 'head_pitch_joint'
        ]
        
        # Lower body joint names (legs only)
        lower_body_joint_names = [
            'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
        ]
        
        # Map joint names to indices
        for i, name in enumerate(self.dof_names):
            if name in left_joint_names:
                self.left_joints.append(i)
            if name in right_joint_names:
                self.right_joints.append(i)
            if name in upper_body_joint_names:
                self.upper_body_joints.append(i)
            if name in lower_body_joint_names:
                self.lower_body_joints.append(i)
                
        # Convert to tensors for efficient indexing
        self.left_joints = torch.tensor(self.left_joints, device=self.device, dtype=torch.long)
        self.right_joints = torch.tensor(self.right_joints, device=self.device, dtype=torch.long)
        self.upper_body_joints = torch.tensor(self.upper_body_joints, device=self.device, dtype=torch.long)
        self.lower_body_joints = torch.tensor(self.lower_body_joints, device=self.device, dtype=torch.long)
        
    def step(self, actions):
        """Environment step with humanoid-specific post-processing"""
        
        # Call parent step function
        obs, rew, done, info = super().step(actions)
        
        # Update humanoid-specific buffers
        self._update_humanoid_buffers()
        
        return obs, rew, done, info
        
    def _update_humanoid_buffers(self):
        """Update humanoid-specific observation buffers"""
        
        # Update gravity vector in robot frame
        self.gravity_vec_buf = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # Update upper body orientation (torso orientation relative to gravity)
        torso_quat = self.base_quat  # Assuming base is torso
        self.upper_body_orientation_buf = quat_to_euler_xyz(torso_quat)
        
        # Calculate joint symmetry error
        if len(self.left_joints) > 0 and len(self.right_joints) > 0:
            left_pos = self.dof_pos[:, self.left_joints]
            right_pos = self.dof_pos[:, self.right_joints]
            
            # For symmetric joints, right should be negative of left (mirrored)
            # Handle special cases for certain joints that don't mirror directly
            symmetry_error = torch.abs(left_pos + right_pos)  # Simplified symmetry check
            self.joint_symmetry_error_buf = torch.mean(symmetry_error, dim=1)
        
    def compute_observations(self):
        """Compute observations including humanoid-specific features"""
        
        # Update terrain info
        self.terrain_levels_buf = self.terrain_levels.clone()
        self.terrain_types_buf = self.terrain_types.clone()
        
        # Base observations from parent class
        base_obs = super().compute_observations()
        
        # Additional humanoid observations
        self.privileged_obs_buf = torch.cat((
            base_obs,
            self.upper_body_orientation_buf,  # 3
            self.gravity_vec_buf,  # 3
            self.joint_symmetry_error_buf.unsqueeze(1),  # 1
            (self.terrain_levels_buf / self.max_terrain_level).unsqueeze(1),  # 1 (normalized)
            self.terrain_types_buf.unsqueeze(1).float(),  # 1
        ), dim=-1)
        
        # Standard observations (subset for policy)
        self.obs_buf = base_obs
        
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            
    def _reward_joint_pos_symmetry(self):
        """Reward for symmetric joint positions between left and right sides"""
        if len(self.left_joints) == 0 or len(self.right_joints) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        left_pos = self.dof_pos[:, self.left_joints]
        right_pos = self.dof_pos[:, self.right_joints]
        
        # Calculate symmetry error (right should mirror left)
        # For most joints, symmetric means right = -left
        symmetry_error = torch.abs(left_pos + right_pos)
        
        # Special handling for joints that should be equal rather than mirrored
        # (e.g., knee joints often bend the same way)
        knee_left_idx = self._get_joint_index('left_knee_joint')
        knee_right_idx = self._get_joint_index('right_knee_joint')
        
        if knee_left_idx is not None and knee_right_idx is not None:
            knee_left_pos = self.dof_pos[:, knee_left_idx]
            knee_right_pos = self.dof_pos[:, knee_right_idx]
            knee_symmetry = torch.abs(knee_left_pos - knee_right_pos)
            
            # Replace knee symmetry error in the overall calculation
            left_knee_in_left = (self.left_joints == knee_left_idx).nonzero(as_tuple=True)[0]
            if len(left_knee_in_left) > 0:
                symmetry_error[:, left_knee_in_left[0]] = knee_symmetry
        
        # Return positive reward (closer to 0 symmetry error is better)
        return torch.exp(-torch.sum(symmetry_error, dim=1) / 0.5)  # Exponential reward
        
    def _reward_joint_vel_symmetry(self):
        """Reward for symmetric joint velocities"""
        if len(self.left_joints) == 0 or len(self.right_joints) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        left_vel = self.dof_vel[:, self.left_joints]
        right_vel = self.dof_vel[:, self.right_joints]
        
        # Calculate velocity symmetry error
        symmetry_error = torch.abs(left_vel + right_vel)
        return torch.exp(-torch.sum(symmetry_error, dim=1) / 1.0)
        
    def _reward_upper_body_orientation(self):
        """Reward for keeping upper body upright"""
        # Penalize tilt in roll and pitch, allow yaw rotation
        roll_pitch_error = torch.abs(self.upper_body_orientation_buf[:, :2])  # Roll and pitch only
        return torch.exp(-torch.sum(roll_pitch_error, dim=1) / 0.3)
        
    def _reward_arm_swing_symmetry(self):
        """Reward for natural arm swing during walking"""
        
        # Get shoulder and elbow positions
        left_shoulder_pitch_idx = self._get_joint_index('left_shoulder_pitch_joint')
        right_shoulder_pitch_idx = self._get_joint_index('right_shoulder_pitch_joint')
        
        if left_shoulder_pitch_idx is None or right_shoulder_pitch_idx is None:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            
        left_shoulder = self.dof_pos[:, left_shoulder_pitch_idx]
        right_shoulder = self.dof_pos[:, right_shoulder_pitch_idx]
        
        # Natural arm swing should be opposite to each other
        arm_symmetry = torch.abs(left_shoulder + right_shoulder)
        
        # Reward when arms swing naturally (opposite directions)
        return torch.exp(-arm_symmetry / 0.5)
        
    def _reward_head_stability(self):
        """Reward for keeping head stable"""
        head_yaw_idx = self._get_joint_index('head_yaw_joint')
        head_pitch_idx = self._get_joint_index('head_pitch_joint')
        
        head_stability = 0.0
        if head_yaw_idx is not None:
            head_stability += torch.abs(self.dof_pos[:, head_yaw_idx])
        if head_pitch_idx is not None:
            head_stability += torch.abs(self.dof_pos[:, head_pitch_idx])
            
        return torch.exp(-head_stability / 0.2)
        
    def _reward_energy_consumption(self):
        """Reward for energy efficiency"""
        # Energy is roughly proportional to torque * velocity
        energy = torch.abs(self.torques * self.dof_vel)
        total_energy = torch.sum(energy, dim=1)
        
        # Normalize by robot weight and number of joints
        energy_per_joint = total_energy / self.num_dof
        return torch.exp(-energy_per_joint / 10.0)
        
    def _reward_terrain_adaptation(self):
        """Bonus reward for successfully traversing different terrain types"""
        # Simple implementation: reward forward progress on difficult terrain
        terrain_difficulty = self.terrain_levels_buf.float() / self.max_terrain_level
        forward_progress = torch.abs(self.base_lin_vel[:, 0])  # Forward velocity
        
        # Higher reward for progress on more difficult terrain
        return terrain_difficulty * forward_progress * 0.1
        
    def _reward_step_height(self):
        """Penalize excessive foot lift"""
        # This would require foot position tracking - simplified version
        # Penalize high foot velocities in z direction
        foot_z_vel = torch.abs(self.base_lin_vel[:, 2])  # Simplified as base z velocity
        return torch.exp(-foot_z_vel / 0.5)
        
    def _reward_foot_clearance(self):
        """Reward proper foot clearance over obstacles"""
        # This would need detailed foot tracking - simplified version
        # Reward moderate vertical movement during locomotion
        z_vel = torch.abs(self.base_lin_vel[:, 2])
        optimal_clearance = 0.1  # Target clearance velocity
        clearance_error = torch.abs(z_vel - optimal_clearance)
        return torch.exp(-clearance_error / 0.1)
        
    def _get_joint_index(self, joint_name):
        """Helper function to get joint index by name"""
        try:
            return self.dof_names.index(joint_name)
        except ValueError:
            return None
            
    def compute_reward(self):
        """Compute reward with humanoid-specific components"""
        
        # Get base rewards from parent class
        super().compute_reward()
        
        # Add humanoid-specific rewards
        self.reward_buf += self.cfg.rewards.scales.joint_pos_symmetry * self._reward_joint_pos_symmetry()
        self.reward_buf += self.cfg.rewards.scales.joint_vel_symmetry * self._reward_joint_vel_symmetry()
        self.reward_buf += self.cfg.rewards.scales.upper_body_orientation * self._reward_upper_body_orientation()
        self.reward_buf += self.cfg.rewards.scales.arm_swing_symmetry * self._reward_arm_swing_symmetry()
        self.reward_buf += self.cfg.rewards.scales.head_stability * self._reward_head_stability()
        self.reward_buf += self.cfg.rewards.scales.energy_consumption * self._reward_energy_consumption()
        self.reward_buf += self.cfg.rewards.scales.terrain_adaptation * self._reward_terrain_adaptation()
        self.reward_buf += self.cfg.rewards.scales.step_height * self._reward_step_height()
        self.reward_buf += self.cfg.rewards.scales.foot_clearance * self._reward_foot_clearance()
        
    def reset_idx(self, env_ids):
        """Reset environments with humanoid-specific initialization"""
        
        # Call parent reset
        super().reset_idx(env_ids)
        
        # Reset humanoid-specific buffers
        self.upper_body_orientation_buf[env_ids] = 0.0
        self.joint_symmetry_error_buf[env_ids] = 0.0
        self.gravity_vec_buf[env_ids] = torch.tensor([0., 0., -1.], device=self.device, dtype=torch.float)
        
        # Initialize with slight random joint positions for exploration
        if len(env_ids) > 0:
            # Add small random offsets to upper body joints for natural poses
            for joint_idx in self.upper_body_joints:
                if joint_idx < self.num_dof:
                    noise = torch_rand_sqrt_float(-0.05, 0.05, (len(env_ids),), device=self.device)
                    self.dof_pos[env_ids, joint_idx] += noise 