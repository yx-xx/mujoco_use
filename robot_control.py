import mujoco
import mujoco.viewer
import numpy as np
import time

class RobotController:
    def __init__(self, model_path):
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 重置模型状态
        mujoco.mj_resetData(self.model, self.data)
        
        # 获取关节数量并设置关节参数
        self.n_joints = self.model.nv
        self.model.opt.timestep = 0.002  # 设置更大的时间步长以提高稳定性
        
        # 设置关节阻尼和刚度
        self.model.dof_damping[:] = 1.0  # 阻尼系数
        self.model.jnt_stiffness[:] = 0.0  # 关节刚度初始化为0
        
        # 控制器参数
        self.kp = np.array([1000.0] * self.n_joints)  # 每个关节的位置增益
        self.kd = np.array([10.0] * self.n_joints)   # 每个关节的速度增益
        self.ki = np.array([0.1] * self.n_joints)    # 每个关节的积分增益
        
        # 设置初始状态和目标
        self.target_positions = np.zeros(self.n_joints)
        self.target_velocities = np.zeros(self.n_joints)
        self.target_torques = np.zeros(self.n_joints)
        
        # 积分误差和限幅参数
        self.pos_error_integral = np.zeros(self.n_joints)
        self.integral_limit = 1.0  # 积分限幅值
        self.output_limit = 10.0  # 输出限幅值
        
        # 控制相关标志
        self.control_mode = 'position'  # 'position', 'velocity', 'torque'
        self.use_gravity_comp = False    # 是否使用重力补偿
        self.use_friction_comp = False   # 是否使用摩擦力补偿
    
    def get_compensation_torques(self):
        """计算补偿力矩（重力和摩擦力）"""
        comp_torque = np.zeros(self.n_joints)
        
        # 更新动力学
        mujoco.mj_forward(self.model, self.data)
        
        # 重力补偿
        if self.use_gravity_comp:
            comp_torque += self.data.qfrc_bias[:self.n_joints]
            
        # 摩擦力补偿（考虑阻尼和库仑摩擦）
        if self.use_friction_comp:
            friction = (self.model.dof_damping * self.data.qvel[:self.n_joints] +
                      0.1 * np.sign(self.data.qvel[:self.n_joints]))  # 简化的摩擦模型
            comp_torque -= friction
            
        return comp_torque
    
    def position_control(self):
        """位置控制：PID控制器 + 补偿控制"""
        # 获取当前状态
        q = self.data.qpos[:self.n_joints]
        qdot = self.data.qvel[:self.n_joints]
        
        # 计算误差
        pos_error = self.target_positions - q
        vel_error = -qdot  # 假设目标速度为0
        
        # 更新积分误差（带抗饱和）
        self.pos_error_integral += pos_error * self.model.opt.timestep
        self.pos_error_integral = np.clip(self.pos_error_integral, 
                                        -self.integral_limit, 
                                        self.integral_limit)
        
        # PID控制
        u = (self.kp * pos_error + 
             self.kd * vel_error + 
             self.ki * self.pos_error_integral)
        
        # 补偿力矩
        u += self.get_compensation_torques()
        
        # 输出限幅
        u = np.clip(u, -self.output_limit, self.output_limit)
        
        return u
    
    def velocity_control(self):
        """速度控制：PI控制器 + 补偿控制"""
        qdot = self.data.qvel[:self.n_joints]
        vel_error = self.target_velocities - qdot
        
        # PI控制
        u = self.kd * vel_error + self.ki * self.pos_error_integral
        u += self.get_compensation_torques()
        
        # 输出限幅
        u = np.clip(u, -self.output_limit, self.output_limit)
        
        return u
    
    def torque_control(self):
        """直接力矩控制（带补偿）"""
        u = self.target_torques + self.get_compensation_torques()
        u = np.clip(u, -self.output_limit, self.output_limit)
        return u
    
    def step(self):
        """执行一步控制"""
        # 计算控制输出
        if self.control_mode == 'position':
            self.data.ctrl = self.position_control()
        elif self.control_mode == 'velocity':
            self.data.ctrl = self.velocity_control()
        else:  # torque
            self.data.ctrl = self.torque_control()
        
        # 在每个显示帧内进行多次物理仿真步进以提高稳定性
        n_steps = 5  # 每帧进行5次物理仿真
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

def main():
    # 创建机器人控制器
    controller = RobotController('./models/ti5robot_x/ti5robot.xml')
    
    # 创建可视化窗口
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        # 重置到初始状态并等待稳定
        mujoco.mj_resetData(controller.model, controller.data)
        time.sleep(0.5)
        
        # 设置初始目标位置
        initial_pos = controller.data.qpos[:controller.n_joints].copy()
        controller.target_positions = initial_pos
        
        # 设置较小的初始阻尼和增益，然后逐渐增加
        controller.model.dof_damping[:] = 0.1
        controller.kp = np.array([10.0] * controller.n_joints)
        controller.kd = np.array([1.0] * controller.n_joints)
        
        print("正在初始化控制器...")
        
        # 主循环
        last_print_time = time.time()
        while viewer.is_running():
            try:
                # 记录循环开始时间
                step_start = time.time()
                
                # 逐渐增加阻尼和增益（在前5秒内）
                elapsed_time = time.time() - step_start
                if elapsed_time < 5.0:
                    ramp = min(1.0, elapsed_time / 5.0)
                    controller.model.dof_damping[:] = 0.1 + ramp * 0.9  # 从0.1渐增到1.0
                    controller.kp = np.array([10.0 + ramp * 90.0] * controller.n_joints)  # 从10渐增到100
                    controller.kd = np.array([1.0 + ramp * 9.0] * controller.n_joints)   # 从1渐增到10
                
                # 执行控制
                controller.step()
                
                # 计算误差
                pos_error = np.abs(controller.target_positions - 
                                 controller.data.qpos[:controller.n_joints])
                vel_error = np.abs(controller.data.qvel[:controller.n_joints])
                
                # 每0.5秒打印一次状态信息
                current_time = time.time()
                if current_time - last_print_time > 0.5:
                    # print(f"\n{'='*50}")
                    # print(f"控制模式: {controller.control_mode}")
                    print(f"最大位置误差: {pos_error.max():.6f}")
                    print(f"最大速度误差: {vel_error.max():.6f}")
                    print(f"增益 kp: {controller.kp[0]:.1f}, kd: {controller.kd[0]:.1f}")
                    print(f"阻尼系数: {controller.model.dof_damping[0]:.1f}")
                    last_print_time = current_time
                
                # 更新可视化
                viewer.sync()
                
                # 控制循环频率
                elapsed = time.time() - step_start
                if elapsed < 0.01:  # 100Hz的更新频率
                    time.sleep(0.01 - elapsed)
                    
            except KeyboardInterrupt:
                print("\n用户中断，正在退出...")
                break
            except Exception as e:
                print(f"\n发生错误: {str(e)}")
                break

if __name__ == "__main__":
    main()


