# Isaac Gym 可视化指南 (Linux服务器)

由于您在Linux服务器上运行Isaac Gym，有多种方法可以查看可视化效果。以下是详细的解决方案：

## 🖥️ **方法1: X11转发 (最简单)**

### SSH连接时启用X11转发

```bash
# 从本地连接服务器时使用-X参数
ssh -X username@your-server-ip

# 或者使用更安全的-Y参数
ssh -Y username@your-server-ip
```

### 在服务器上运行可视化

```bash
# 设置显示环境
export DISPLAY=:0.0

# 运行带可视化的训练 (少量环境用于观察)
./run_with_display.sh train g1 visual_test

# 运行已训练策略的可视化
./run_with_display.sh play g1 g1_low_speed
```

---

## 🖥️ **方法2: VNC远程桌面 (推荐)**

### 一键设置VNC服务器

```bash
# 安装并配置VNC
./setup_vnc.sh

# 这会自动：
# 1. 安装VNC服务器
# 2. 安装轻量桌面环境 (XFCE)
# 3. 配置Isaac Gym兼容设置
# 4. 启动VNC服务器在端口5901
```

### 连接VNC

**在本地电脑上：**

1. **SSH隧道 (推荐)**：
   ```bash
   ssh -L 5901:localhost:5901 username@your-server-ip
   ```

2. **VNC客户端连接**：
   - **Windows**: 使用RealVNC Viewer, TightVNC等
   - **macOS**: 系统自带Screen Sharing或RealVNC
   - **Linux**: Remmina, VNC Viewer等
   - 连接地址: `localhost:5901`

3. **运行Isaac Gym**：
   ```bash
   export DISPLAY=:1
   ./run_with_display.sh train g1
   ```

---

## 🎥 **方法3: 视频录制 (最实用)**

### 录制训练过程

```bash
# 录制G1训练过程
./record_visualization.sh g1 train

# 录制H1训练过程  
./record_visualization.sh h1 train

# 录制已训练策略
./record_visualization.sh g1 play g1_low_speed
```

### 输出内容

录制会生成以下文件：
- `recordings/g1_20241201_143022/locomotion.mp4` - 机器人运动视频
- `recordings/g1_20241201_143022/frame_0000.png` - 关键帧截图
- `recordings/g1_20241201_143022/summary.png` - 性能总结图

---

## 🔧 **环境配置检查**

### 检查显示设置

```bash
# 检查X11是否可用
echo $DISPLAY
xdpyinfo

# 检查OpenGL支持
glxinfo | grep "OpenGL version"

# 测试X11转发
xclock  # 应该弹出时钟窗口
```

### Isaac Gym环境变量

```bash
# 设置兼容性环境变量
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330
export DISPLAY=:0
```

---

## 📋 **具体使用步骤**

### 步骤1: 选择可视化方法

**简单快速**: 使用X11转发
```bash
ssh -X user@server
./run_with_display.sh train g1
```

**完整桌面**: 使用VNC
```bash
./setup_vnc.sh
# 然后通过VNC客户端连接
```

**离线查看**: 录制视频
```bash
./record_visualization.sh g1 train
# 下载recordings文件夹查看
```

### 步骤2: 运行可视化训练

```bash
# 短时间可视化训练 (100次迭代)
./run_with_display.sh train g1 visual_demo

# 可视化已训练策略
./run_with_display.sh play g1 your_run_name
```

### 步骤3: 调整可视化参数

编辑 `run_with_display.sh` 中的参数：
- `--num_envs=16`: 环境数量 (少量便于观察)
- `--max_iterations=100`: 训练迭代次数
- 相机位置和角度

---

## 🚀 **实际使用示例**

### 场景1: 快速查看训练效果

```bash
# SSH连接并启用X11
ssh -X user@your-server

# 快速可视化训练
export DISPLAY=:0
python legged_gym/scripts/train.py \
    --task=g1 \
    --run_name=quick_demo \
    --num_envs=4 \
    --max_iterations=50
```

### 场景2: 完整录制展示

```bash
# 训练一个完整策略
./train_g1_low_speed.sh

# 录制最佳策略表现
./record_visualization.sh g1 play g1_low_speed

# 下载recordings文件夹到本地查看
```

### 场景3: 实时远程观看

```bash
# 服务器端设置VNC
./setup_vnc.sh

# 本地建立隧道
ssh -L 5901:localhost:5901 user@server

# VNC连接localhost:5901
# 在VNC桌面中运行Isaac Gym
```

---

## ⚠️ **常见问题解决**

### 问题1: "Cannot connect to X server"

```bash
# 解决方案1: 重新设置DISPLAY
export DISPLAY=:0.0

# 解决方案2: 启动虚拟显示
Xvfb :0 -screen 0 1920x1080x24 &
export DISPLAY=:0
```

### 问题2: OpenGL错误

```bash
# 设置Mesa兼容性
export MESA_GL_VERSION_OVERRIDE=3.3
export MESA_GLSL_VERSION_OVERRIDE=330

# 或使用软件渲染
export LIBGL_ALWAYS_SOFTWARE=1
```

### 问题3: VNC连接失败

```bash
# 检查VNC状态
vncserver -list

# 重启VNC服务器
vncserver -kill :1
vncserver :1 -geometry 1920x1080 -depth 24
```

### 问题4: 视频录制失败

```bash
# 安装缺失的依赖
pip install opencv-python matplotlib

# 检查ffmpeg
sudo apt install ffmpeg
```

---

## 💡 **性能优化建议**

### 可视化时的设置建议

1. **减少环境数量**: `num_envs=4-16` (而不是4096)
2. **降低分辨率**: 1280x720 而不是 1920x1080
3. **关闭域随机化**: 减少计算开销
4. **使用录制模式**: 避免实时渲染的网络延迟

### 网络优化

```bash
# 压缩X11流量
ssh -X -C user@server

# VNC压缩设置
vncserver :1 -geometry 1280x720 -depth 16
```

---

## 🎯 **推荐工作流程**

### 日常开发调试
```bash
# 1. 使用headless训练 (高性能)
./train_g1_low_speed.sh

# 2. 定期录制检查点
./record_visualization.sh g1 play g1_low_speed

# 3. 本地查看录制结果
```

### 演示展示
```bash
# 1. 设置VNC完整桌面
./setup_vnc.sh

# 2. 实时演示训练过程
./run_with_display.sh train g1 demo

# 3. 展示最佳策略效果
./run_with_display.sh play g1 best_policy
```

这样您就可以在Linux服务器上完美地查看Isaac Gym的可视化效果了！选择最适合您需求的方法即可。 