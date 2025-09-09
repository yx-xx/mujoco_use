import mujoco
import mujoco.viewer
import numpy as np
import time

# 加载模型
model = mujoco.MjModel.from_xml_path('./models/ti5robot_x/ti5robot.xml')
data = mujoco.MjData(model)

# 创建可视化窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    # 获取关节数量
    n_joints = model.nv
    
    # 初始化上次更新时间
    last_update = time.time()
    
    while viewer.is_running():
        step_start = time.time()
        
        # 生成控制量（使用正弦波）
        # time_now = time.time()
        # target_positions = np.sin(time_now) * np.ones(n_joints) * 0.5
        
        target_positions = np.zeros(n_joints)

        # 设置关节位置控制
        data.ctrl = target_positions
        
        # 步进仿真
        mujoco.mj_step(model, data)
        
        # 获取当前关节角度
        joint_angles = data.qpos[:n_joints]
        print(f"当前关节角度: {joint_angles}")
        
        # 更新可视化
        viewer.sync()
        
        # 控制仿真频率（1000Hz）
        time_until_next = 0.001 - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)


