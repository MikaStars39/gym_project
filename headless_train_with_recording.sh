#!/bin/bash

# Headless Training with Periodic Recording
# 在无GUI的Linux服务器上进行高性能训练，同时定期录制可视化

TASK=${1:-g1}
RUN_NAME=${2:-"headless_$(date +%Y%m%d_%H%M%S)"}
RECORD_INTERVAL=${3:-500}  # 每500次迭代录制一次

echo "🚀 启动无GUI高性能训练 + 定期录制"
echo "任务: $TASK"
echo "运行名称: $RUN_NAME" 
echo "录制间隔: 每 $RECORD_INTERVAL 次迭代"

# 创建输出目录
mkdir -p logs recordings

# 设置虚拟显示 (用于录制，不影响性能)
export DISPLAY=:99
if ! pgrep -x "Xvfb" > /dev/null; then
    echo "🖥️  启动虚拟显示服务器..."
    Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
    sleep 2
fi

# 设置Isaac Gym环境变量
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

echo "✅ 环境配置完成，开始训练..."

# 后台运行主训练 (headless高性能)
python legged_gym/scripts/train.py \
    --task=$TASK \
    --run_name=$RUN_NAME \
    --num_envs=4096 \
    --headless \
    --max_iterations=2000 &

TRAIN_PID=$!
echo "📊 训练进程启动 (PID: $TRAIN_PID)"

# 定期录制检查点
sleep 60  # 等待训练启动

ITERATION=0
while kill -0 $TRAIN_PID 2>/dev/null; do
    ITERATION=$((ITERATION + RECORD_INTERVAL))
    
    echo "🎥 录制检查点 - 迭代 $ITERATION"
    
    # 录制当前策略 (如果存在)
    if [ -d "logs" ]; then
        ./record_visualization.sh $TASK play $RUN_NAME 2>/dev/null || {
            echo "⏳ 策略尚未准备好，继续等待..."
        }
    fi
    
    # 等待下一次录制
    sleep 300  # 5分钟后再次检查
done

echo "✅ 训练完成！"
echo "📁 日志文件: logs/$RUN_NAME/"
echo "🎬 录制文件: recordings/"

# 最终录制
echo "🎬 录制最终策略..."
./record_visualization.sh $TASK play $RUN_NAME

echo "🎉 全部完成！可以下载recordings文件夹查看可视化结果" 