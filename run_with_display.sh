#!/bin/bash

# Isaac Gym Visualization Script for Remote Linux Server
# This script sets up X11 forwarding for visualization

echo "Setting up Isaac Gym visualization..."

# Check if DISPLAY is set
if [ -z "$DISPLAY" ]; then
    echo "No DISPLAY variable set. Setting up virtual display..."
    export DISPLAY=:0
fi

# Set up virtual framebuffer if needed
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "Starting virtual framebuffer..."
    Xvfb :0 -screen 0 1920x1080x24 &
    sleep 2
fi

# Set Isaac Gym environment variables for better compatibility
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

echo "Display setup complete. DISPLAY=$DISPLAY"

# Run the command with visualization enabled
if [ "$1" = "train" ]; then
    echo "Starting training with visualization..."
    python legged_gym/scripts/train.py \
        --task=${2:-g1} \
        --run_name=${3:-"visual_test"} \
        --num_envs=16 \
        --max_iterations=100
elif [ "$1" = "play" ]; then
    echo "Starting policy playback with visualization..."
    python legged_gym/scripts/play.py \
        --task=${2:-g1} \
        --run_name=${3:-""} \
        --num_envs=4
else
    echo "Usage: $0 [train|play] [task] [run_name]"
    echo "Example: $0 train g1 my_visual_test"
    echo "Example: $0 play g1 g1_low_speed"
fi 