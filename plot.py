import matplotlib.pyplot as plt
from collections import deque
import time
import os

class PIDVisualizer:
    def __init__(self, num_joints, window_size=100, display_joint=0):
        self.num_joints = num_joints
        self.display_joint = display_joint  # 要实时显示的关节序号
        
        # 创建保存图片的文件夹
        self.save_dir = 'pid_plots'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        plt.ion()  # 开启交互模式
        # 只创建一个用于实时显示的图表
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 6))
        
        self.window_size = window_size
        self.time_data = deque(maxlen=window_size)
        self.target_data = [deque(maxlen=window_size) for _ in range(num_joints)]
        self.current_data = [deque(maxlen=window_size) for _ in range(num_joints)]
        self.error_data = [deque(maxlen=window_size) for _ in range(num_joints)]
        
        # 为所有关节创建数据存储
        self.all_data = {i: {'time': [], 'target': [], 'current': [], 'error': []} 
                        for i in range(num_joints)}
        
        # 只为显示的关节创建实时曲线
        target_line, = self.ax.plot([], [], 'g-', label='目标位置')
        current_line, = self.ax.plot([], [], 'b-', label='当前位置')
        error_line, = self.ax.plot([], [], 'r-', label='误差')
        self.lines = (target_line, current_line, error_line)
        
        self.ax.set_title(f'关节 {self.display_joint + 1} PID控制曲线')
        self.ax.set_xlabel('时间 (s)')
        self.ax.set_ylabel('角度 (rad)')
        self.ax.legend()
        self.ax.grid(True)
        
        plt.tight_layout()
        self.start_time = time.time()
    
    def update(self, targets, currents, errors):
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        
        # 更新所有关节的数据存储
        for i in range(self.num_joints):
            self.all_data[i]['time'].append(current_time)
            self.all_data[i]['target'].append(targets[i])
            self.all_data[i]['current'].append(currents[i])
            self.all_data[i]['error'].append(errors[i])
            
            # 每100个数据点保存一次图片
            if len(self.all_data[i]['time']) % 100 == 0:
                self.save_joint_plot(i)
        
        # 只更新显示的关节的实时图表
        i = self.display_joint
        target_line, current_line, error_line = self.lines
        x_data = list(self.time_data)
        
        self.target_data[i].append(targets[i])
        self.current_data[i].append(currents[i])
        self.error_data[i].append(errors[i])
        
        target_line.set_data(x_data, list(self.target_data[i]))
        current_line.set_data(x_data, list(self.current_data[i]))
        error_line.set_data(x_data, list(self.error_data[i]))
        
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.flush_events()
    
    def save_joint_plot(self, joint_idx):
        """保存指定关节的PID曲线图片"""
        if joint_idx == self.display_joint:
            return  # 不保存正在显示的关节图片
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.all_data[joint_idx]['time'], 
                self.all_data[joint_idx]['target'], 'g-', label='目标位置')
        plt.plot(self.all_data[joint_idx]['time'], 
                self.all_data[joint_idx]['current'], 'b-', label='当前位置')
        plt.plot(self.all_data[joint_idx]['time'], 
                self.all_data[joint_idx]['error'], 'r-', label='误差')
        
        plt.title(f'关节 {joint_idx + 1} PID控制曲线')
        plt.xlabel('时间 (s)')
        plt.ylabel('角度 (rad)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.save_dir, f'joint_{joint_idx + 1}_pid.png'))
        plt.close()
    
    def change_display_joint(self, new_joint_idx):
        """切换显示的关节"""
        if 0 <= new_joint_idx < self.num_joints:
            self.display_joint = new_joint_idx
            self.ax.set_title(f'关节 {self.display_joint + 1} PID控制曲线')
            self.time_data.clear()
            self.target_data[new_joint_idx].clear()
            self.current_data[new_joint_idx].clear()
            self.error_data[new_joint_idx].clear()