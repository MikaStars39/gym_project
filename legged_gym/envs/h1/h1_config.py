# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.base_config import LeggedRobotCfg, LeggedRobotCfgPPO

class H1RoughCfg(LeggedRobotCfg):
    """Configuration for H1 full-size humanoid robot on rough terrain"""
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 287  # Increased for full-size humanoid observations
        num_privileged_obs = 225  # Privileged observations for asymmetric training
        num_actions = 19  # H1 DOF count (may vary based on specific configuration)
        episode_length_s = 20
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        
        # Complex terrain configuration as specified
        terrain_proportions = [0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1]  # [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gaps]
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # difficulty levels
        num_cols = 20  # terrain types
        max_init_terrain_level = 5
        
        # Height measurements for terrain perception
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = True
        
        class ranges(LeggedRobotCfg.commands.ranges):
            # H1 walking: xy [0.0, 1.0] m/s, yaw [0.0, 0.5] rad/s
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.05]  # H1 is taller, start higher
        rot = [0.0, 0.0, 0.0, 1.0]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]
        
        # H1 default joint angles (adjust based on actual H1 URDF)
        default_joint_angles = {
            # Lower body joints
            'left_hip_yaw': 0.0,
            'left_hip_roll': 0.0,
            'left_hip_pitch': -0.4,
            'left_knee': 0.8,
            'left_ankle_pitch': -0.4,
            'left_ankle_roll': 0.0,
            'right_hip_yaw': 0.0,
            'right_hip_roll': 0.0,
            'right_hip_pitch': -0.4,
            'right_knee': 0.8,
            'right_ankle_pitch': -0.4,
            'right_ankle_roll': 0.0,
            
            # Upper body joints (H1 specific)
            'torso': 0.0,
            'left_shoulder_pitch': 0.0,
            'left_shoulder_roll': 0.0,
            'left_elbow': 0.0,
            'right_shoulder_pitch': 0.0,
            'right_shoulder_roll': 0.0,
            'right_elbow': 0.0,
        }
    
    class control(LeggedRobotCfg.control):
        control_type = 'P'  # Position control
        stiffness = {
            # Lower body - higher stiffness for larger robot
            'hip_yaw': 200.0,
            'hip_roll': 200.0,
            'hip_pitch': 250.0,
            'knee': 250.0,
            'ankle_pitch': 30.0,
            'ankle_roll': 30.0,
            
            # Upper body - moderate stiffness for full-size humanoid
            'torso': 150.0,
            'shoulder_pitch': 80.0,
            'shoulder_roll': 80.0,
            'elbow': 50.0,
        }
        damping = {
            # Lower body damping
            'hip_yaw': 8.0,
            'hip_roll': 8.0,
            'hip_pitch': 10.0,
            'knee': 10.0,
            'ankle_pitch': 2.0,
            'ankle_roll': 2.0,
            
            # Upper body damping
            'torso': 5.0,
            'shoulder_pitch': 3.0,
            'shoulder_roll': 3.0,
            'elbow': 2.0,
        }
        action_scale = 0.25
        decimation = 4
    
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "h1"
        foot_name = "ankle"
        penalize_contacts_on = ["torso", "pelvis", "shoulder", "elbow", "forearm"]
        terminate_after_contacts_on = ["torso", "pelvis"]
        self_collisions = 1  # Enable self-collision detection
        flip_visual_attachments = True
        replace_cylinder_with_capsule = True
        fix_base_link = False
        
    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            # Basic locomotion rewards
            termination = -0.0
            tracking_lin_vel = 2.0  # Increased for better tracking reward
            tracking_ang_vel = 1.0  # Increased for better tracking
            lin_vel_z = -1.0  # Reduced penalty
            ang_vel_xy = -0.02  # Reduced penalty
            orientation = -0.0
            torques = -0.000005  # Reduced penalty
            dof_vel = -0.0
            dof_acc = -1.0e-7  # Reduced penalty  
            base_height = -0.0
            feet_air_time = 2.0  # Increased reward
            collision = -0.5  # Reduced penalty
            feet_stumble = -0.0
            action_rate = -0.005  # Reduced penalty
            stand_still = -0.0
            
            # Full-size humanoid specific rewards
            joint_pos_symmetry = 0.3  # Changed to positive reward
            joint_vel_symmetry = 0.15  # Changed to positive reward
            upper_body_orientation = 0.8  # Changed to positive reward
            arm_swing_symmetry = 0.2  # Changed to positive reward
            energy_consumption = -0.001  # Reduced penalty
            
            # H1-specific rewards
            torso_stability = 0.5  # Changed to positive reward
            balance_recovery = 0.3  # Increased bonus
            
            # Terrain-specific rewards
            terrain_adaptation = 0.3  # Increased bonus
            step_height = -0.02  # Reduced penalty
            foot_clearance = 0.15  # Increased reward
            
        only_positive_rewards = False  # Allow negative rewards for better learning signal
        tracking_sigma = 0.25
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 1.05  # H1 standing height (taller than G1)
        max_contact_force = 500.0  # Higher for larger humanoid
        
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-10., 10.]  # kg variation for larger humanoid
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.2  # Slightly higher for larger robot
        
        # Additional randomization for H1
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]  # rad
        
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
            imu = 0.1  # IMU noise
            
    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 100.
        clip_actions = 100.
        action_scale = 0.25
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            gravity_vec = 1.0
            
    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1
        
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class H1RoughCfgPPO(LeggedRobotCfgPPO):
    """PPO configuration for H1 full-size humanoid robot"""
    
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 2500  # Increased for complex full-size humanoid training
        
        save_interval = 50
        experiment_name = 'h1_rough_terrain'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None 