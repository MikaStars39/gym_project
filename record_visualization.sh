#!/bin/bash

# Isaac Gym Visualization Recording Script
# Records training or policy execution for later viewing

TASK=${1:-g1}
MODE=${2:-play}
RUN_NAME=${3:-""}

echo "Recording Isaac Gym visualization..."
echo "Task: $TASK"
echo "Mode: $MODE"

# Create recordings directory
mkdir -p recordings

if [ "$MODE" = "train" ]; then
    echo "Recording training process..."
    python legged_gym/scripts/record_video.py \
        --task=$TASK \
        --run_name=recording_${TASK}_train
        
elif [ "$MODE" = "play" ]; then
    if [ -z "$RUN_NAME" ]; then
        echo "Error: Please specify run_name for play mode"
        echo "Usage: $0 <task> play <run_name>"
        exit 1
    fi
    
    echo "Recording policy playback: $RUN_NAME"
    python legged_gym/scripts/record_video.py \
        --task=$TASK \
        --run_name=$RUN_NAME
        
else
    echo "Usage: $0 <task> [train|play] [run_name]"
    echo "Examples:"
    echo "  $0 g1 train                    # Record G1 training"
    echo "  $0 h1 play h1_locomotion       # Record H1 policy playback"
    echo "  $0 g1 play g1_low_speed        # Record G1 trained policy"
fi

echo "Recordings will be saved in the 'recordings/' directory" 