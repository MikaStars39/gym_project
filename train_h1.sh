#!/bin/bash

# H1 Locomotion Training Script
# xy axis linear velocity: [0.0, 1.0] m/s
# yaw axis angular velocity: [0.0, 0.5] rad/s
# Velocity tracking error requirement: < 0.25 m/s
# Survival time requirement: > 10s

echo "Starting H1 Full-Size Humanoid Locomotion Training..."
echo "Target: xy velocity [0.0-1.0] m/s, yaw velocity [0.0-0.5] rad/s"
echo "Success criteria: velocity error < 0.25 m/s, survival > 10s"

python legged_gym/scripts/train.py \
    --task=h1 \
    --run_name=h1_locomotion \
    --num_envs=4096 \
    --headless \
    --max_iterations=2500

echo "Training completed. Check logs for results." 