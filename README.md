# Isaac Gym Humanoid Robot Complex Terrain Locomotion

This project implements full-body humanoid robot locomotion on complex terrain using Isaac Gym and PPO reinforcement learning. It features both G1 (23-DOF) and H1 (full-size) humanoid robots navigating challenging terrains with symmetric movement constraints.

## Features

- **Humanoid Robots**: G1 (23-DOF compact) and H1 (full-size) robots
- **Complex Terrain**: 7 terrain types with 10 difficulty levels each
- **Full-Body Control**: Upper body symmetry constraints and natural movement
- **Speed Variants**: Low-speed (0-1 m/s) and high-speed (1.5-2 m/s) locomotion
- **Comprehensive Evaluation**: Style scoring, terrain crossing, and survival metrics

## Terrain Types

The environment includes 7 terrain types with specified proportions:
- **Smooth Slope** (25%): Gentle inclined surfaces
- **Rough Slope** (25%): Irregular inclined terrain
- **Rough Stairs Up** (10%): Ascending irregular steps
- **Rough Stairs Down** (10%): Descending irregular steps
- **Discrete** (10%): Isolated platforms and gaps
- **Stepping Stones** (10%): Scattered footholds
- **Gaps** (10%): Chasms requiring jumping

## Robot Configurations

### G1 Humanoid (23-DOF)
- Compact humanoid with full upper body control
- 12 lower body DOF + 11 upper body DOF
- Optimized for agile movement and terrain adaptation

### H1 Humanoid (Full-Size)
- Full-scale humanoid robot
- Enhanced stability and power for challenging terrain
- Advanced balance recovery capabilities

## Installation

### Prerequisites
- Ubuntu 18.04/20.04 LTS
- NVIDIA GPU with CUDA support
- Python 3.8+
- Isaac Gym Preview 4

### Setup Instructions

1. **Create Conda Environment**
   ```bash
   conda create -n locomotion python=3.8
   conda activate locomotion
   ```

2. **Install PyTorch**
   ```bash
   pip3 install torch torchvision torchaudio
   ```

3. **Install Isaac Gym**
   ```bash
   cd isaacgym/python && pip install -e .
   # Test installation
   cd examples && python 1080_balls_of_solitude.py
   ```

4. **Install RSL-RL**
   ```bash
   cd rsl_rl && pip install -e .
   ```

5. **Install Legged Gym**
   ```bash
   cd legged_gym && pip install -e .
   ```

## Training

### G1 Low-Speed Locomotion (30%)
Train G1 for low-speed walking with velocity tracking:
```bash
# Using script
./train_g1_low_speed.sh

# Manual command
python legged_gym/scripts/train.py \
    --task=g1 \
    --run_name=g1_low_speed \
    --num_envs=4096 \
    --headless
```

**Requirements:**
- xy velocity: [0.0, 1.0] m/s
- yaw velocity: [0.0, 0.5] rad/s
- Velocity tracking error < 0.25 m/s
- Average survival time > 10s

### H1 Locomotion (20%)
Train H1 full-size humanoid:
```bash
# Using script
./train_h1.sh

# Manual command
python legged_gym/scripts/train.py \
    --task=h1 \
    --run_name=h1_locomotion \
    --num_envs=4096 \
    --headless
```

**Requirements:**
- xy velocity: [0.0, 1.0] m/s
- yaw velocity: [0.0, 0.5] rad/s
- Velocity tracking error < 0.25 m/s
- Average survival time > 10s

### High-Speed Locomotion (10%)
Train either G1 or H1 for high-speed movement:
```bash
# G1 high-speed
./train_high_speed.sh g1

# H1 high-speed
./train_high_speed.sh h1

# Manual command
python legged_gym/scripts/train.py \
    --task=g1 \
    --run_name=g1_high_speed \
    --num_envs=4096 \
    --headless
```

**Requirements:**
- xy velocity: [1.5, 2.0] m/s
- yaw velocity: [0.0, 0.5] rad/s
- Velocity tracking error < 0.5 m/s
- Average survival time > 10s

## Evaluation

### Policy Testing and Export
Evaluate trained policies and export to ONNX format:
```bash
# Evaluate and export policy
./evaluate_policy.sh g1 g1_low_speed 500

# Manual evaluation
python legged_gym/scripts/play.py \
    --task=g1 \
    --run_name=g1_low_speed \
    --checkpoint=500 \
    --export_policy
```

### ONNX Verification
Verify exported policy functionality:
```bash
python legged_gym/scripts/eval_onnx.py --task=g1
```

## Scoring System

### Task Evaluation (60%)

#### G1 Low-Speed Walking (30%)
- Velocity tracking performance
- Terrain crossing success
- Survival time metrics

#### H1 Walking (20%)
- Full-size humanoid locomotion
- Balance and stability
- Terrain adaptation

#### High-Speed Challenge (10%)
- Either G1 or H1
- Enhanced velocity requirements
- Advanced terrain navigation

### Terrain Evaluation (30%)

Terrain crossing success rates:
- **Stairs**: 0.25 points per difficulty level
- **Other terrains**: 0.5 points per difficulty level
- **Total**: 7 terrain types × 10 difficulty levels

### Style Evaluation (10%)

#### Upper Body Style (5%)
- Shoulder and elbow joint position accuracy
- Target error < 0.5 rad for symmetric poses

#### Lower Body Style (5%)
- Hip and ankle joint position accuracy
- Target error < 0.8 rad for natural gait

## Key Features

### Humanoid-Specific Rewards
- **Joint Position Symmetry**: Enforces natural symmetric movement
- **Joint Velocity Symmetry**: Promotes coordinated motion
- **Upper Body Orientation**: Maintains upright posture
- **Arm Swing Symmetry**: Natural arm movement during walking
- **Energy Consumption**: Efficiency optimization
- **Head Stability**: Stable head position
- **Terrain Adaptation**: Bonus for difficult terrain traversal

### Advanced Observations
- **Terrain Height Maps**: Local terrain perception
- **Gravity Vector**: Balance-aware observations
- **Joint Symmetry Error**: Real-time symmetry monitoring
- **Terrain Type Information**: Adaptive behavior cues

### Domain Randomization
- **Friction Variation**: [0.5, 1.25] range
- **Mass Randomization**: ±5kg for G1, ±10kg for H1
- **Motor Strength**: [0.9, 1.1] multiplier
- **Motor Offset**: ±0.02 rad random bias

## File Structure

```
legged_gym/
├── envs/
│   ├── base/
│   │   ├── base_config.py      # Base configuration
│   │   └── legged_robot.py     # Base environment
│   ├── g1/
│   │   ├── g1.py              # G1 environment
│   │   └── g1_config.py       # G1 configuration
│   └── h1/
│       ├── h1.py              # H1 environment
│       └── h1_config.py       # H1 configuration
├── scripts/
│   ├── train.py               # Training script
│   ├── play.py                # Evaluation script
│   └── eval_onnx.py          # ONNX verification
└── utils/
    ├── helpers.py             # Utility functions
    ├── task_registry.py       # Environment registry
    └── logger.py              # Training logger
```

## Usage Examples

### Training Commands
```bash
# G1 low-speed training
python legged_gym/scripts/train.py --task=g1 --run_name=my_g1_run

# H1 training with custom parameters
python legged_gym/scripts/train.py --task=h1 --num_envs=2048 --max_iterations=3000

# Resume training from checkpoint
python legged_gym/scripts/train.py --task=g1 --resume --checkpoint=1000
```

### Evaluation Commands
```bash
# Play trained policy
python legged_gym/scripts/play.py --task=g1 --run_name=my_g1_run

# Export to ONNX
python legged_gym/scripts/play.py --task=g1 --run_name=my_g1_run --export_policy

# Verify ONNX export
python legged_gym/scripts/eval_onnx.py --task=g1
```

## Performance Optimization

### For H100 GPU Training
The project is optimized for H100 GPU training with:
- Large batch sizes (4096+ environments)
- Efficient tensor operations
- Optimized reward computation
- Minimal Python overhead

### Training Tips
1. **Start with low-speed**: Train low-speed policies first
2. **Monitor symmetry**: Watch joint symmetry metrics during training
3. **Terrain curriculum**: Enable curriculum learning for better adaptation
4. **Style constraints**: Balance locomotion and style rewards
5. **Evaluation frequency**: Regular policy evaluation for progress tracking

## Troubleshooting

### Common Issues
1. **Isaac Gym Installation**: Check CUDA compatibility and driver versions
2. **Memory Issues**: Reduce num_envs if running out of GPU memory
3. **Slow Training**: Ensure headless mode and proper GPU utilization
4. **Export Errors**: Verify model checkpoint exists and is loadable

### Performance Tuning
- Adjust `num_envs` based on GPU memory
- Modify `max_iterations` for convergence
- Tune reward scales for desired behavior
- Optimize terrain difficulty progression

## License

This project is licensed under the BSD-3-Clause License. See LICENSE file for details.

## Acknowledgments

- NVIDIA Isaac Gym for physics simulation
- RSL-RL for reinforcement learning algorithms
- Unitree Robotics for G1 and H1 robot designs
- ETH Zurich RSL for legged robotics research

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{humanoid_terrain_locomotion,
  title={Isaac Gym Humanoid Robot Complex Terrain Locomotion},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/your-repo}}
}
``` 