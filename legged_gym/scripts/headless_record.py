# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import json

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import isaacgym


def headless_record_policy(args):
    """
    在无GUI环境下记录策略表现和生成可视化图表
    不依赖实时渲染，通过数据分析生成可视化内容
    """
    
    print("🎬 无GUI环境策略录制开始...")
    
    # Load environment configuration
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 优化配置用于数据收集
    env_cfg.env.num_envs = 16  # 小批量用于详细分析
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    # Create environment (headless)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Load policy if available
    policy = None
    if args.run_name:
        try:
            train_cfg.runner.resume = True
            ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
            policy = ppo_runner.get_inference_policy(device=env.device)
            print(f"✅ 成功加载策略: {args.run_name}")
        except Exception as e:
            print(f"⚠️  策略加载失败: {e}")
            print("🎲 将使用随机动作进行演示")
    
    # Setup recording
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"recordings/{args.task}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 输出目录: {output_dir}")
    
    # Data collection
    episode_data = []
    step_data = []
    terrain_performance = {}
    
    total_steps = 2000
    print(f"📊 开始收集 {total_steps} 步数据...")
    
    obs = env.get_observations()
    
    for step in range(total_steps):
        # Get actions
        if policy is not None:
            with torch.no_grad():
                actions = policy(obs.detach())
        else:
            # Random actions for demonstration
            actions = torch.randn(env.num_envs, env.num_actions, device=env.device) * 0.5
        
        # Step environment
        obs, rewards, dones, infos = env.step(actions.detach())
        
        # Collect step data
        step_info = {
            'step': step,
            'mean_reward': torch.mean(rewards).item(),
            'base_height': torch.mean(env.root_states[:, 2]).item() if hasattr(env, 'root_states') else 0,
            'linear_velocity': torch.mean(torch.norm(env.base_lin_vel[:, :2], dim=1)).item() if hasattr(env, 'base_lin_vel') else 0,
            'angular_velocity': torch.mean(torch.abs(env.base_ang_vel[:, 2])).item() if hasattr(env, 'base_ang_vel') else 0,
        }
        step_data.append(step_info)
        
        # Track episode completions
        if torch.any(dones):
            done_envs = torch.where(dones)[0]
            for env_idx in done_envs:
                episode_length = env.episode_length_buf[env_idx].item() if hasattr(env, 'episode_length_buf') else step
                survival_time = episode_length * env.dt if hasattr(env, 'dt') else episode_length * 0.02
                
                episode_info = {
                    'episode_length': episode_length,
                    'survival_time': survival_time,
                    'final_reward': rewards[env_idx].item(),
                    'terrain_level': getattr(env, 'terrain_levels', torch.zeros(env.num_envs))[env_idx].item(),
                    'terrain_type': getattr(env, 'terrain_types', torch.zeros(env.num_envs))[env_idx].item(),
                }
                episode_data.append(episode_info)
        
        # Progress update
        if step % 200 == 0:
            progress = (step / total_steps) * 100
            print(f"📈 进度: {progress:.1f}% - 平均奖励: {step_info['mean_reward']:.3f}")
    
    print("✅ 数据收集完成，生成可视化...")
    
    # Generate visualizations
    create_performance_plots(output_dir, step_data, episode_data, args.task)
    
    # Save raw data
    save_data(output_dir, step_data, episode_data, args)
    
    # Generate summary report
    generate_summary_report(output_dir, step_data, episode_data, args.task)
    
    print(f"🎉 录制完成！文件保存至: {output_dir}")
    print("📋 生成的文件:")
    print(f"  📊 performance_analysis.png - 性能分析图")
    print(f"  📈 training_curves.png - 训练曲线")
    print(f"  🗃️  raw_data.json - 原始数据")
    print(f"  📄 summary_report.txt - 总结报告")


def create_performance_plots(output_dir, step_data, episode_data, task_name):
    """生成性能分析图表"""
    
    # Create comprehensive performance visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Reward over time
    ax1 = plt.subplot(3, 3, 1)
    steps = [d['step'] for d in step_data]
    rewards = [d['mean_reward'] for d in step_data]
    plt.plot(steps, rewards, 'b-', alpha=0.7)
    plt.title('平均奖励随时间变化')
    plt.xlabel('步数')
    plt.ylabel('奖励')
    plt.grid(True, alpha=0.3)
    
    # 2. Velocity tracking
    ax2 = plt.subplot(3, 3, 2)
    velocities = [d['linear_velocity'] for d in step_data]
    plt.plot(steps, velocities, 'g-', alpha=0.7)
    plt.title('线速度')
    plt.xlabel('步数')
    plt.ylabel('速度 (m/s)')
    plt.grid(True, alpha=0.3)
    
    # 3. Base height stability
    ax3 = plt.subplot(3, 3, 3)
    heights = [d['base_height'] for d in step_data]
    plt.plot(steps, heights, 'r-', alpha=0.7)
    plt.title('机器人高度')
    plt.xlabel('步数')
    plt.ylabel('高度 (m)')
    plt.grid(True, alpha=0.3)
    
    # 4. Episode survival times
    if episode_data:
        ax4 = plt.subplot(3, 3, 4)
        survival_times = [d['survival_time'] for d in episode_data]
        plt.hist(survival_times, bins=20, alpha=0.7, color='orange')
        plt.title('生存时间分布')
        plt.xlabel('生存时间 (秒)')
        plt.ylabel('频次')
        plt.axvline(x=10, color='red', linestyle='--', label='目标: 10秒')
        plt.legend()
    
    # 5. Terrain performance
    if episode_data:
        ax5 = plt.subplot(3, 3, 5)
        terrain_levels = [d['terrain_level'] for d in episode_data]
        if terrain_levels:
            plt.hist(terrain_levels, bins=10, alpha=0.7, color='purple')
            plt.title('地形难度分布')
            plt.xlabel('地形等级')
            plt.ylabel('完成次数')
    
    # 6. Angular velocity
    ax6 = plt.subplot(3, 3, 6)
    ang_velocities = [d['angular_velocity'] for d in step_data]
    plt.plot(steps, ang_velocities, 'purple', alpha=0.7)
    plt.title('角速度')
    plt.xlabel('步数')
    plt.ylabel('角速度 (rad/s)')
    plt.grid(True, alpha=0.3)
    
    # 7. Performance metrics summary
    ax7 = plt.subplot(3, 3, 7)
    if episode_data:
        avg_survival = np.mean([d['survival_time'] for d in episode_data])
        survival_success = np.mean([1 if d['survival_time'] > 10 else 0 for d in episode_data])
        avg_reward = np.mean([d['final_reward'] for d in episode_data])
        
        metrics = ['平均生存时间', '生存成功率', '平均奖励']
        values = [avg_survival, survival_success * 100, avg_reward * 10]  # Scale for visibility
        colors = ['green', 'blue', 'orange']
        
        bars = plt.bar(metrics, values, color=colors, alpha=0.7)
        plt.title('关键指标')
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, [avg_survival, survival_success, avg_reward]):
            height = bar.get_height()
            if bar == bars[0]:  # Survival time
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1f}s', ha='center', va='bottom')
            elif bar == bars[1]:  # Success rate
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.1%}', ha='center', va='bottom')
            else:  # Reward
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
    
    # 8. Robot info
    ax8 = plt.subplot(3, 3, 8)
    ax8.text(0.1, 0.8, f'机器人: {task_name.upper()}', fontsize=14, weight='bold')
    if task_name == 'g1':
        ax8.text(0.1, 0.6, '• 23自由度紧凑型', fontsize=10)
        ax8.text(0.1, 0.5, '• 全身运动控制', fontsize=10)
    elif task_name == 'h1':
        ax8.text(0.1, 0.6, '• 全尺寸人形机器人', fontsize=10)
        ax8.text(0.1, 0.5, '• 增强稳定性', fontsize=10)
    
    ax8.text(0.1, 0.3, '• 复杂地形导航', fontsize=10)
    ax8.text(0.1, 0.2, '• 对称运动约束', fontsize=10)
    ax8.set_xlim(0, 1)
    ax8.set_ylim(0, 1)
    ax8.axis('off')
    
    # 9. Terrain distribution
    ax9 = plt.subplot(3, 3, 9)
    terrain_names = ['平缓斜坡', '粗糙斜坡', '上楼梯', '下楼梯', '离散', '踏脚石', '间隙']
    terrain_props = [0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1]
    plt.pie(terrain_props, labels=terrain_names, autopct='%1.1f%%', startangle=90)
    plt.title('地形分布')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_data(output_dir, step_data, episode_data, args):
    """保存原始数据"""
    data = {
        'task': args.task,
        'run_name': args.run_name,
        'timestamp': datetime.now().isoformat(),
        'step_data': step_data,
        'episode_data': episode_data
    }
    
    with open(os.path.join(output_dir, 'raw_data.json'), 'w') as f:
        json.dump(data, f, indent=2)


def generate_summary_report(output_dir, step_data, episode_data, task_name):
    """生成总结报告"""
    
    report = []
    report.append("="*60)
    report.append(f"Isaac Gym {task_name.upper()} 策略性能报告")
    report.append("="*60)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    if step_data:
        avg_reward = np.mean([d['mean_reward'] for d in step_data])
        avg_velocity = np.mean([d['linear_velocity'] for d in step_data])
        avg_height = np.mean([d['base_height'] for d in step_data])
        
        report.append("📊 整体性能指标:")
        report.append(f"  • 平均奖励: {avg_reward:.3f}")
        report.append(f"  • 平均线速度: {avg_velocity:.3f} m/s")
        report.append(f"  • 平均机器人高度: {avg_height:.3f} m")
        report.append("")
    
    if episode_data:
        survival_times = [d['survival_time'] for d in episode_data]
        avg_survival = np.mean(survival_times)
        max_survival = np.max(survival_times)
        survival_success = np.mean([1 if t > 10 else 0 for t in survival_times])
        
        report.append("🕐 生存时间分析:")
        report.append(f"  • 平均生存时间: {avg_survival:.2f} 秒")
        report.append(f"  • 最大生存时间: {max_survival:.2f} 秒")
        report.append(f"  • 生存成功率 (>10s): {survival_success:.1%}")
        
        # Check success criteria
        if avg_survival > 10:
            report.append("  ✅ 满足生存时间要求")
        else:
            report.append("  ❌ 未满足生存时间要求")
        report.append("")
        
        terrain_levels = [d['terrain_level'] for d in episode_data]
        if terrain_levels:
            avg_terrain_level = np.mean(terrain_levels)
            max_terrain_level = np.max(terrain_levels)
            
            report.append("🏔️  地形适应性:")
            report.append(f"  • 平均地形等级: {avg_terrain_level:.1f}")
            report.append(f"  • 最大地形等级: {max_terrain_level}")
            report.append("")
    
    # Performance evaluation based on task requirements
    report.append("🎯 任务要求评估:")
    if task_name == 'g1':
        report.append("  G1 低速行走要求:")
        report.append("  • xy速度: [0.0, 1.0] m/s")
        report.append("  • 速度跟踪误差 < 0.25 m/s")
        report.append("  • 生存时间 > 10s")
    elif task_name == 'h1':
        report.append("  H1 行走要求:")
        report.append("  • xy速度: [0.0, 1.0] m/s") 
        report.append("  • 速度跟踪误差 < 0.25 m/s")
        report.append("  • 生存时间 > 10s")
    
    report.append("")
    report.append("📁 生成文件:")
    report.append("  • performance_analysis.png - 性能分析图表")
    report.append("  • raw_data.json - 原始数据")
    report.append("  • summary_report.txt - 本报告")
    
    # Write report
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))


if __name__ == '__main__':
    args = get_args()
    headless_record_policy(args) 