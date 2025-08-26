#!/bin/bash

# Policy Evaluation Script
# Evaluate trained policies and export to ONNX format

TASK=${1:-g1}
RUN_NAME=${2:-""}
CHECKPOINT=${3:--1}

if [ -z "$RUN_NAME" ]; then
    echo "Usage: ./evaluate_policy.sh <task> <run_name> [checkpoint]"
    echo "Example: ./evaluate_policy.sh g1 g1_low_speed 500"
    echo "Available tasks: g1, h1"
    exit 1
fi

echo "Evaluating $TASK policy: $RUN_NAME"
echo "Checkpoint: $CHECKPOINT (-1 = latest)"

# Run evaluation and export policy
python legged_gym/scripts/play.py \
    --task=$TASK \
    --run_name=$RUN_NAME \
    --checkpoint=$CHECKPOINT \
    --export_policy \
    --num_envs=50

echo "Evaluation completed."
echo "Policy exported to logs/exported/policy.pt"

# Verify exported policy
echo "Verifying exported ONNX policy..."
python legged_gym/scripts/eval_onnx.py --task=$TASK

echo "Evaluation and export completed successfully!" 