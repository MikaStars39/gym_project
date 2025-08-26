#!/bin/bash

# High-Speed Locomotion Training Script
# Choose G1 or H1 for high-speed challenge
# xy axis linear velocity: [1.5, 2.0] m/s
# yaw axis angular velocity: [0.0, 0.5] rad/s
# Velocity tracking error requirement: < 0.5 m/s
# Survival time requirement: > 10s

ROBOT=${1:-g1}  # Default to G1, can specify h1 as argument

echo "Starting $ROBOT High-Speed Locomotion Training..."
echo "Target: xy velocity [1.5-2.0] m/s, yaw velocity [0.0-0.5] rad/s"
echo "Success criteria: velocity error < 0.5 m/s, survival > 10s"

python legged_gym/scripts/train.py \
    --task=$ROBOT \
    --run_name=${ROBOT}_high_speed \
    --num_envs=4096 \
    --headless \
    --max_iterations=3000

echo "High-speed training completed. Check logs for results."
echo "Usage: ./train_high_speed.sh [g1|h1]" 