# MuJoCo Robot Control with ROS2

这是一个使用MuJoCo物理引擎的机器人控制项目，支持PID控制和ROS2集成。项目包含实时PID曲线可视化功能，可以通过ROS2话题进行控制和状态监控。

## 功能特点

- 基于MuJoCo的机器人物理仿真
- 支持位置、速度和力矩控制模式
- 实时PID控制曲线可视化
- ROS2集成（发布机器人状态和接收控制命令）
- 自动保存每个关节的PID曲线图表

## 系统要求

- Ubuntu 22.04 或更高版本
- Python 3.8 或更高版本
- ROS2 Humble 或更高版本
- MuJoCo 3.0.0 或更高版本

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yx-xx/mujoco_use
cd mujoco_use
```

2. 创建并激活conda环境（推荐）：
```bash
conda create -n ros2_mujoco python=3.11
conda activate ros2_mujoco
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 编译ROS2消息：
```bash
colcon build --packages-select robot_msgs
source install/setup.bash
```

## 使用方法

### 1. 启动机器人控制节点

```bash
python3 robot_control_ros.py
```

### 2. 发送控制命令

通过ROS2话题`/robot_command`发送控制命令：
```python
ros2 topic pub /robot_command robot_msgs/msg/JointCommand "
control_mode: 'position'
position: [0.0, 0.0, 0.0, ...]  # 目标位置
"
```

### 3. 查看机器人状态

监听`/robot_state`话题：
```bash
ros2 topic echo /robot_state
```

### 4. PID可视化

- 实时显示选定关节的PID控制曲线
- 其他关节的曲线自动保存在`pid_plots`文件夹中
- 使用`controller.change_display_joint(joint_idx)`切换显示的关节

## 项目结构

```
mujoco_use/
├── robot_control.py        # 原始机器人控制代码
├── robot_control_ros.py    # ROS2集成版本
├── plot.py                 # PID可视化模块
├── robot_msgs/            # ROS2消息包
│   ├── msg/
│   │   ├── JointCommand.msg
│   │   └── JointState.msg
│   ├── CMakeLists.txt
│   └── package.xml
├── models/                # 机器人模型文件
│   └── ti5robot_x/
├── pid_plots/            # PID曲线图表保存目录
├── requirements.txt
└── README.md
```

## 自定义消息

### JointCommand.msg
```
std_msgs/Header header
float64[] position    # 目标位置
float64[] velocity    # 目标速度
float64[] effort      # 目标力矩
string control_mode   # 控制模式
```

### JointState.msg
```
std_msgs/Header header
float64[] position        # 当前位置
float64[] velocity        # 当前速度
float64[] effort          # 当前力矩
float64[] position_error  # 位置误差
float64[] velocity_error  # 速度误差
```

## 常见问题

1. 如果遇到MuJoCo导入错误，确保正确安装了MuJoCo：
```bash
pip install mujoco
```

2. 如果ROS2消息无法识别，确保已经编译并source了工作空间：
```bash
colcon build --packages-select robot_msgs
source install/setup.bash
```

## 贡献

欢迎提交Issue和Pull Request。

## 许可证

Apache License 2.0
