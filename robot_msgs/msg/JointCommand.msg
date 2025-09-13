# 控制命令消息
std_msgs/Header header

float64[] position    # 目标位置
float64[] velocity    # 目标速度（可选）
float64[] effort      # 目标力矩（可选）
string control_mode   # 控制模式：'position', 'velocity', 'torque'
