#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from robot_msgs.msg import JointCommand, JointState
from std_msgs.msg import Header
import mujoco
import mujoco.viewer
import numpy as np
import time

class RobotControllerROS(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # 加载模型
        model_path = './models/ti5robot_x/ti5robot.xml'
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 重置模型状态
        mujoco.mj_resetData(self.model, self.data)
        
        # 获取关节数量并设置关节参数
        self.n_joints = self.model.nv
        self.model.opt.timestep = 0.002
        
        # 设置关节阻尼和刚度
        self.model.dof_damping[:] = 1.0
        self.model.jnt_stiffness[:] = 0.0
        
        # 控制器参数
        self.kp = np.array([100.0] * self.n_joints)
        self.kd = np.array([0.0] * self.n_joints)
        self.ki = np.array([0.0] * self.n_joints)
        
        # 设置初始状态和目标
        self.target_positions = np.zeros(self.n_joints)
        self.target_velocities = np.zeros(self.n_joints)
        self.target_torques = np.zeros(self.n_joints)
        
        # 积分误差和限幅参数
        self.pos_error_integral = np.zeros(self.n_joints)
        self.integral_limit = 1.0
        self.output_limit = 10.0
        
        # 控制模式
        self.control_mode = 'position'
        self.use_gravity_comp = True
        self.use_friction_comp = False
        
        
        # 创建ROS2发布者和订阅者
        self.state_pub = self.create_publisher(
            JointState, 
            'robot_state', 
            10
        )
        self.cmd_sub = self.create_subscription(
            JointCommand,
            'robot_command',
            self.command_callback,
            10
        )
        
        # 创建定时器，控制更新频率
        self.timer = self.create_timer(0.01, self.timer_callback)  # 100Hz
        
        # 创建MuJoCo查看器
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
    def command_callback(self, msg):
        """处理接收到的控制命令"""
        self.control_mode = msg.control_mode
        
        if len(msg.position) == self.n_joints:
            self.target_positions = np.array(msg.position)
        if len(msg.velocity) == self.n_joints:
            self.target_velocities = np.array(msg.velocity)
        if len(msg.effort) == self.n_joints:
            self.target_torques = np.array(msg.effort)
    
    def publish_state(self):
        """发布机器人状态"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.position = self.data.qpos[:self.n_joints].tolist()
        msg.velocity = self.data.qvel[:self.n_joints].tolist()
        msg.effort = self.data.ctrl[:self.n_joints].tolist()
        
        pos_error = self.target_positions - self.data.qpos[:self.n_joints]
        vel_error = self.target_velocities - self.data.qvel[:self.n_joints]
        
        msg.position_error = pos_error.tolist()
        msg.velocity_error = vel_error.tolist()
        
        self.state_pub.publish(msg)
    
    def get_compensation_torques(self):
        """计算补偿力矩"""
        comp_torque = np.zeros(self.n_joints)
        mujoco.mj_forward(self.model, self.data)
        
        if self.use_gravity_comp:
            comp_torque += self.data.qfrc_bias[:self.n_joints]
        if self.use_friction_comp:
            friction = (self.model.dof_damping * self.data.qvel[:self.n_joints] +
                      0.1 * np.sign(self.data.qvel[:self.n_joints]))
            comp_torque -= friction
        return comp_torque
    
    def position_control(self):
        """位置控制"""
        q = self.data.qpos[:self.n_joints]
        qdot = self.data.qvel[:self.n_joints]
        
        pos_error = self.target_positions - q
        vel_error = -qdot
        
        self.pos_error_integral += pos_error * self.model.opt.timestep
        self.pos_error_integral = np.clip(self.pos_error_integral, 
                                        -self.integral_limit, 
                                        self.integral_limit)
        
        u = (self.kp * pos_error + 
             self.kd * vel_error + 
             self.ki * self.pos_error_integral)
        
        return u
    
    def velocity_control(self):
        """速度控制"""
        qdot = self.data.qvel[:self.n_joints]
        vel_error = self.target_velocities - qdot
        
        u = self.kd * vel_error + self.ki * self.pos_error_integral
        u += self.get_compensation_torques()
        
        return np.clip(u, -self.output_limit, self.output_limit)
    
    def torque_control(self):
        """力矩控制"""
        u = self.target_torques + self.get_compensation_torques()
        return np.clip(u, -self.output_limit, self.output_limit)
    
    def timer_callback(self):
        """定时器回调函数，执行控制循环"""
        if not self.viewer.is_running():
            self.get_logger().info('Viewer closed, shutting down...')
            rclpy.shutdown()
            return
        
        # 执行控制
        if self.control_mode == 'position':
            self.data.ctrl = self.position_control()
        elif self.control_mode == 'velocity':
            self.data.ctrl = self.velocity_control()
        else:  # torque
            self.data.ctrl = self.torque_control()
        
        # 执行仿真步进
        for _ in range(5):  # 每次控制周期执行5次物理仿真
            mujoco.mj_step(self.model, self.data)
        
        # 发布状态
        self.publish_state()

        self.viewer.sync()
    

def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerROS()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
