# -*- coding: utf-8 -*-
"""
Robot Chassis Kinematics and Physics Models

底盘运动学和物理模型实现
支持四种常见的机器人底盘类型
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, List
import math


class ChassisType(Enum):
    """底盘类型枚举"""
    DIFFERENTIAL_DRIVE = "differential_drive"  # 差速底盘
    ACKERMANN_STEERING = "ackermann_steering"  # 阿克曼底盘
    TRACKED_VEHICLE = "tracked_vehicle"        # 履带底盘
    MECANUM_WHEEL = "mecanum_wheel"            # 麦克纳姆轮底盘


@dataclass
class ChassisConfig:
    """底盘配置参数"""
    # 几何参数
    wheelbase: float = 0.5          # 轴距 (m)
    track_width: float = 0.4        # 轮距 (m)
    wheel_radius: float = 0.1       # 轮子半径 (m)
    body_length: float = 0.6        # 车体长度 (m)
    body_width: float = 0.4         # 车体宽度 (m)
    body_height: float = 0.2        # 车体高度 (m)
    
    # 质量参数
    body_mass: float = 10.0         # 车体质量 (kg)
    wheel_mass: float = 0.5         # 单个轮子质量 (kg)
    
    # 物理参数
    max_linear_velocity: float = 1.0     # 最大线速度 (m/s)
    max_angular_velocity: float = 2.0    # 最大角速度 (rad/s)
    max_linear_acceleration: float = 2.0 # 最大线加速度 (m/s²)
    max_angular_acceleration: float = 4.0 # 最大角加速度 (rad/s²)
    
    # 轮子摩擦系数
    wheel_friction: float = 0.8     # 轮子摩擦系数
    lateral_friction: float = 0.5   # 侧向摩擦系数
    
    # 阿克曼底盘专用参数
    max_steering_angle: float = math.pi / 4  # 最大转向角 (rad)
    
    # 履带底盘专用参数
    track_width_ratio: float = 0.8  # 履带宽度比例


@dataclass
class ChassisState:
    """底盘状态"""
    # 位姿
    x: float = 0.0          # 位置 x (m)
    y: float = 0.0          # 位置 y (m)
    theta: float = 0.0      # 朝向角 (rad)
    
    # 速度
    vx: float = 0.0         # 线速度 x (m/s)
    vy: float = 0.0         # 线速度 y (m/s)
    omega: float = 0.0      # 角速度 (rad/s)
    
    # 加速度
    ax: float = 0.0         # 线加速度 x (m/s²)
    ay: float = 0.0         # 线加速度 y (m/s²)
    alpha: float = 0.0      # 角加速度 (rad/s²)
    
    # 轮子状态 (用于可视化)
    wheel_velocities: Optional[np.ndarray] = None  # 轮子转速 (rad/s)
    wheel_angles: Optional[np.ndarray] = None      # 转向角 (rad, 仅阿克曼)
    
    def get_position(self) -> np.ndarray:
        """获取位置向量"""
        return np.array([self.x, self.y])
    
    def get_velocity(self) -> np.ndarray:
        """获取速度向量"""
        return np.array([self.vx, self.vy])
    
    def get_pose(self) -> np.ndarray:
        """获取位姿向量 [x, y, theta]"""
        return np.array([self.x, self.y, self.theta])


class ChassisBase(ABC):
    """底盘基类"""
    
    def __init__(self, chassis_type: ChassisType, config: ChassisConfig):
        self.chassis_type = chassis_type
        self.config = config
        self.state = ChassisState()
        self.trajectory: List[Tuple[float, float, float]] = []  # 轨迹记录 [(x, y, theta), ...]
        
    @abstractmethod
    def compute_kinematics(self, control_input: np.ndarray) -> np.ndarray:
        """
        计算运动学
        
        Args:
            control_input: 控制输入，具体格式由子类定义
            
        Returns:
            状态变化率 [dx, dy, dtheta, dvx, dvy, domega]
        """
        pass
    
    @abstractmethod
    def apply_control(self, control_input: np.ndarray, dt: float):
        """
        应用控制输入，更新状态
        
        Args:
            control_input: 控制输入
            dt: 时间步长 (s)
        """
        pass
    
    def update(self, dt: float):
        """
        更新底盘状态
        
        Args:
            dt: 时间步长 (s)
        """
        # 记录轨迹
        self.trajectory.append((self.state.x, self.state.y, self.state.theta))
        
    def get_transform_matrix(self) -> np.ndarray:
        """获取从底盘坐标系到世界坐标系的变换矩阵"""
        c, s = math.cos(self.state.theta), math.sin(self.state.theta)
        return np.array([
            [c, -s, self.state.x],
            [s, c, self.state.y],
            [0, 0, 1]
        ])
    
    def local_to_world(self, local_pos: np.ndarray) -> np.ndarray:
        """将局部坐标转换为世界坐标"""
        transform = self.get_transform_matrix()
        local_homo = np.append(local_pos, 1)
        world_homo = transform @ local_homo
        return world_homo[:2]
    
    def world_to_local(self, world_pos: np.ndarray) -> np.ndarray:
        """将世界坐标转换为局部坐标"""
        transform = self.get_transform_matrix()
        inv_transform = np.linalg.inv(transform)
        world_homo = np.append(world_pos, 1)
        local_homo = inv_transform @ world_homo
        return local_homo[:2]
    
    def get_wheel_positions(self) -> List[np.ndarray]:
        """
        获取各轮子相对于车体的位置（局部坐标）
        
        Returns:
            轮子位置列表，每个元素为 [x, y]
        """
        L = self.config.body_length
        W = self.config.body_width
        
        if self.chassis_type == ChassisType.DIFFERENTIAL_DRIVE:
            # 两个驱动轮 + 一个万向轮
            return [
                np.array([0, W/2]),     # 左轮
                np.array([0, -W/2]),    # 右轮
                np.array([-L/3, 0]),    # 万向轮
            ]
        elif self.chassis_type == ChassisType.ACKERMANN_STEERING:
            # 四个轮子
            return [
                np.array([L/2, W/2]),   # 左前
                np.array([L/2, -W/2]),  # 右前
                np.array([-L/2, W/2]),  # 左后
                np.array([-L/2, -W/2]), # 右后
            ]
        elif self.chassis_type == ChassisType.TRACKED_VEHICLE:
            # 履带的接地点
            return [
                np.array([L/2, W/2]),   # 左履带前
                np.array([-L/2, W/2]),  # 左履带后
                np.array([L/2, -W/2]),  # 右履带前
                np.array([-L/2, -W/2]), # 右履带后
            ]
        elif self.chassis_type == ChassisType.MECANUM_WHEEL:
            # 四个麦克纳姆轮
            return [
                np.array([L/2, W/2]),   # 左前
                np.array([L/2, -W/2]),  # 右前
                np.array([-L/2, W/2]),  # 左后
                np.array([-L/2, -W/2]), # 右后
            ]
        return []


class DifferentialDrive(ChassisBase):
    """
    差速底盘
    
    控制输入: [v_left, v_right] 左右轮线速度 (m/s)
    或: [v, omega] 线速度和角速度
    """
    
    def __init__(self, config: Optional[ChassisConfig] = None):
        super().__init__(ChassisType.DIFFERENTIAL_DRIVE, config or ChassisConfig())
        # 初始化轮子状态
        self.state.wheel_velocities = np.zeros(2)  # 左右轮转速
        
    def compute_kinematics(self, control_input: np.ndarray) -> np.ndarray:
        """
        计算差速底盘运动学
        
        Args:
            control_input: [v_left, v_right] 左右轮线速度
            
        Returns:
            状态变化率
        """
        v_left, v_right = control_input[0], control_input[1]
        r = self.config.wheel_radius
        W = self.config.track_width
        theta = self.state.theta
        
        # 转换为车体速度
        v = (v_left + v_right) / 2  # 线速度
        omega = (v_right - v_left) / W  # 角速度
        
        # 限制速度
        v = np.clip(v, -self.config.max_linear_velocity, self.config.max_linear_velocity)
        omega = np.clip(omega, -self.config.max_angular_velocity, self.config.max_angular_velocity)
        
        # 计算状态变化率
        dx = v * math.cos(theta)
        dy = v * math.sin(theta)
        dtheta = omega
        
        return np.array([dx, dy, dtheta, 0, 0, 0])
    
    def apply_control(self, control_input: np.ndarray, dt: float):
        """
        应用控制输入
        
        Args:
            control_input: [v_left, v_right] 或 [v, omega]
            dt: 时间步长
        """
        if len(control_input) == 2:
            v_left, v_right = control_input[0], control_input[1]
        else:
            raise ValueError("Control input must be [v_left, v_right]")
        
        # 限制轮速
        max_wheel_vel = self.config.max_linear_velocity
        v_left = np.clip(v_left, -max_wheel_vel, max_wheel_vel)
        v_right = np.clip(v_right, -max_wheel_vel, max_wheel_vel)
        
        # 更新轮子转速 (rad/s)
        r = self.config.wheel_radius
        self.state.wheel_velocities[0] = v_left / r
        self.state.wheel_velocities[1] = v_right / r
        
        # 计算状态变化
        kinematics = self.compute_kinematics(np.array([v_left, v_right]))
        
        # 更新状态 (欧拉积分)
        self.state.x += kinematics[0] * dt
        self.state.y += kinematics[1] * dt
        self.state.theta += kinematics[2] * dt
        self.state.theta = math.atan2(math.sin(self.state.theta), math.cos(self.state.theta))  # 归一化
        
        self.state.vx = kinematics[0]
        self.state.vy = kinematics[1]
        self.state.omega = kinematics[2]
        
        self.update(dt)
    
    def set_velocity(self, v: float, omega: float):
        """
        设置目标速度和角速度
        
        Args:
            v: 线速度 (m/s)
            omega: 角速度 (rad/s)
        """
        W = self.config.track_width
        # 逆运动学
        v_left = v - omega * W / 2
        v_right = v + omega * W / 2
        return np.array([v_left, v_right])


class AckermannSteering(ChassisBase):
    """
    阿克曼转向底盘
    
    控制输入: [v, steering_angle] 线速度和转向角
    """
    
    def __init__(self, config: Optional[ChassisConfig] = None):
        super().__init__(ChassisType.ACKERMANN_STEERING, config or ChassisConfig())
        self.state.wheel_velocities = np.zeros(4)  # 四个轮子转速
        self.state.wheel_angles = np.zeros(2)      # 前轮转向角 [左, 右]
        
    def compute_kinematics(self, control_input: np.ndarray) -> np.ndarray:
        """
        计算阿克曼底盘运动学
        
        Args:
            control_input: [v, steering_angle] 线速度和平均转向角
            
        Returns:
            状态变化率
        """
        v = control_input[0]
        steering_angle = control_input[1]
        
        # 限制输入
        v = np.clip(v, -self.config.max_linear_velocity, self.config.max_linear_velocity)
        steering_angle = np.clip(steering_angle, 
                                  -self.config.max_steering_angle, 
                                  self.config.max_steering_angle)
        
        L = self.config.wheelbase
        theta = self.state.theta
        
        # 计算角速度
        if abs(steering_angle) < 1e-6:
            omega = 0
            R = float('inf')  # 直线行驶
        else:
            R = L / math.tan(steering_angle)  # 转弯半径
            omega = v / R
        
        dx = v * math.cos(theta)
        dy = v * math.sin(theta)
        dtheta = omega
        
        return np.array([dx, dy, dtheta, 0, 0, 0])
    
    def apply_control(self, control_input: np.ndarray, dt: float):
        """
        应用控制输入
        
        Args:
            control_input: [v, steering_angle]
            dt: 时间步长
        """
        v, steering_angle = control_input[0], control_input[1]
        
        # 限制输入
        v = np.clip(v, -self.config.max_linear_velocity, self.config.max_linear_velocity)
        steering_angle = np.clip(steering_angle, 
                                  -self.config.max_steering_angle, 
                                  self.config.max_steering_angle)
        
        # 计算阿克曼转向几何
        L = self.config.wheelbase
        W = self.config.track_width
        
        if abs(steering_angle) < 1e-6:
            # 直线行驶
            inner_angle = outer_angle = 0
        else:
            # 计算内外轮转向角
            R = L / math.tan(steering_angle)
            inner_angle = math.atan(L / (R - W/2))
            outer_angle = math.atan(L / (R + W/2))
            
            if steering_angle < 0:
                inner_angle, outer_angle = -outer_angle, -inner_angle
        
        # 更新转向角 (假设左前轮转向角更大)
        if steering_angle >= 0:  # 左转
            self.state.wheel_angles[0] = inner_angle
            self.state.wheel_angles[1] = outer_angle
        else:  # 右转
            self.state.wheel_angles[0] = outer_angle
            self.state.wheel_angles[1] = inner_angle
        
        # 计算各轮速度 (考虑转弯时的速度差)
        r = self.config.wheel_radius
        if abs(steering_angle) < 1e-6:
            wheel_v = v
            self.state.wheel_velocities = np.full(4, wheel_v / r)
        else:
            R = L / math.tan(steering_angle)
            R_left = math.sqrt((R - W/2)**2 + L**2)
            R_right = math.sqrt((R + W/2)**2 + L**2)
            R_rear = abs(R)
            
            # 前轮转速
            v_left_front = v * R_left / R_rear
            v_right_front = v * R_right / R_rear
            
            self.state.wheel_velocities[0] = v_left_front / r
            self.state.wheel_velocities[1] = v_right_front / r
            self.state.wheel_velocities[2] = v / r  # 左后
            self.state.wheel_velocities[3] = v / r  # 右后
        
        # 更新状态
        kinematics = self.compute_kinematics(np.array([v, steering_angle]))
        
        self.state.x += kinematics[0] * dt
        self.state.y += kinematics[1] * dt
        self.state.theta += kinematics[2] * dt
        self.state.theta = math.atan2(math.sin(self.state.theta), math.cos(self.state.theta))
        
        self.state.vx = kinematics[0]
        self.state.vy = kinematics[1]
        self.state.omega = kinematics[2]
        
        self.update(dt)
    
    def get_turning_radius(self) -> float:
        """获取当前转弯半径"""
        if abs(self.state.wheel_angles[0]) < 1e-6:
            return float('inf')
        L = self.config.wheelbase
        return L / math.tan(self.state.wheel_angles[0])


class TrackedVehicle(ChassisBase):
    """
    履带底盘
    
    控制输入: [v_left, v_right] 左右履带速度
    """
    
    def __init__(self, config: Optional[ChassisConfig] = None):
        super().__init__(ChassisType.TRACKED_VEHICLE, config or ChassisConfig())
        self.state.wheel_velocities = np.zeros(2)  # 左右履带速度
        self.ground_contact_points: List[np.ndarray] = []  # 地面接触点
        
    def compute_kinematics(self, control_input: np.ndarray) -> np.ndarray:
        """
        计算履带底盘运动学
        
        Args:
            control_input: [v_left, v_right] 左右履带速度
            
        Returns:
            状态变化率
        """
        v_left, v_right = control_input[0], control_input[1]
        W = self.config.track_width
        theta = self.state.theta
        
        # 计算车体速度 (与差速底盘相同)
        v = (v_left + v_right) / 2
        omega = (v_right - v_left) / W
        
        # 限制速度
        v = np.clip(v, -self.config.max_linear_velocity, self.config.max_linear_velocity)
        omega = np.clip(omega, -self.config.max_angular_velocity, self.config.max_angular_velocity)
        
        dx = v * math.cos(theta)
        dy = v * math.sin(theta)
        dtheta = omega
        
        return np.array([dx, dy, dtheta, 0, 0, 0])
    
    def apply_control(self, control_input: np.ndarray, dt: float):
        """
        应用控制输入
        
        Args:
            control_input: [v_left, v_right]
            dt: 时间步长
        """
        v_left, v_right = control_input[0], control_input[1]
        
        # 限制履带速度
        max_vel = self.config.max_linear_velocity
        v_left = np.clip(v_left, -max_vel, max_vel)
        v_right = np.clip(v_right, -max_vel, max_vel)
        
        self.state.wheel_velocities[0] = v_left
        self.state.wheel_velocities[1] = v_right
        
        # 更新状态
        kinematics = self.compute_kinematics(np.array([v_left, v_right]))
        
        self.state.x += kinematics[0] * dt
        self.state.y += kinematics[1] * dt
        self.state.theta += kinematics[2] * dt
        self.state.theta = math.atan2(math.sin(self.state.theta), math.cos(self.state.theta))
        
        self.state.vx = kinematics[0]
        self.state.vy = kinematics[1]
        self.state.omega = kinematics[2]
        
        self.update(dt)
    
    def get_ground_pressure_distribution(self) -> np.ndarray:
        """
        获取地面压力分布（简化模型）
        
        Returns:
            压力分布数组 [左前, 左后, 右前, 右后]
        """
        total_weight = self.config.body_mass * 9.81
        # 简化为均匀分布
        return np.full(4, total_weight / 4)


class MecanumWheel(ChassisBase):
    """
    麦克纳姆轮底盘
    
    控制输入: [vx, vy, omega] 局部坐标系下的线速度和角速度
    或: [v_fl, v_fr, v_rl, v_rr] 四个轮子转速
    """
    
    def __init__(self, config: Optional[ChassisConfig] = None):
        super().__init__(ChassisType.MECANUM_WHEEL, config or ChassisConfig())
        self.state.wheel_velocities = np.zeros(4)  # 四个轮子转速
        
    def compute_kinematics(self, control_input: np.ndarray) -> np.ndarray:
        """
        计算麦克纳姆轮底盘运动学
        
        Args:
            control_input: [vx, vy, omega] 局部坐标系下的速度
            
        Returns:
            状态变化率
        """
        vx_local, vy_local, omega = control_input[0], control_input[1], control_input[2]
        theta = self.state.theta
        
        # 限制速度
        v_local = math.sqrt(vx_local**2 + vy_local**2)
        if v_local > self.config.max_linear_velocity:
            scale = self.config.max_linear_velocity / v_local
            vx_local *= scale
            vy_local *= scale
        omega = np.clip(omega, -self.config.max_angular_velocity, self.config.max_angular_velocity)
        
        # 转换到世界坐标系
        vx = vx_local * math.cos(theta) - vy_local * math.sin(theta)
        vy = vx_local * math.sin(theta) + vy_local * math.cos(theta)
        
        return np.array([vx, vy, omega, 0, 0, 0])
    
    def apply_control(self, control_input: np.ndarray, dt: float):
        """
        应用控制输入
        
        Args:
            control_input: [vx, vy, omega] 或 [v_fl, v_fr, v_rl, v_rr]
            dt: 时间步长
        """
        if len(control_input) == 3:
            vx, vy, omega = control_input[0], control_input[1], control_input[2]
            
            # 限制速度
            v = math.sqrt(vx**2 + vy**2)
            if v > self.config.max_linear_velocity:
                scale = self.config.max_linear_velocity / v
                vx *= scale
                vy *= scale
            omega = np.clip(omega, -self.config.max_angular_velocity, self.config.max_angular_velocity)
            
        elif len(control_input) == 4:
            # 从轮子速度解算
            v_fl, v_fr, v_rl, v_rr = control_input
            L = self.config.body_length
            W = self.config.track_width
            r = self.config.wheel_radius
            
            # 麦克纳姆轮逆运动学
            vx = (v_fl + v_fr + v_rl + v_rr) * r / 4
            vy = (-v_fl + v_fr + v_rl - v_rr) * r / 4
            omega = (-v_fl + v_fr - v_rl + v_rr) * r / (2 * (L + W))
        else:
            raise ValueError("Control input must be [vx, vy, omega] or 4 wheel velocities")
        
        # 更新状态
        kinematics = self.compute_kinematics(np.array([vx, vy, omega]))
        
        self.state.x += kinematics[0] * dt
        self.state.y += kinematics[1] * dt
        self.state.theta += kinematics[2] * dt
        self.state.theta = math.atan2(math.sin(self.state.theta), math.cos(self.state.theta))
        
        self.state.vx = kinematics[0]
        self.state.vy = kinematics[1]
        self.state.omega = kinematics[2]
        
        # 计算轮子速度（用于可视化）
        self.state.wheel_velocities = self.compute_wheel_velocities(vx, vy, omega)
        
        self.update(dt)
    
    def compute_wheel_velocities(self, vx: float, vy: float, omega: float) -> np.ndarray:
        """
        计算四个轮子的目标速度
        
        Args:
            vx: 局部坐标系线速度 x
            vy: 局部坐标系线速度 y
            omega: 角速度
            
        Returns:
            四个轮子的转速 [v_fl, v_fr, v_rl, v_rr]
        """
        L = self.config.body_length
        W = self.config.track_width
        r = self.config.wheel_radius
        
        # 麦克纳姆轮运动学
        # v_fl (前左)
        v_fl = (vx - vy - (L + W) * omega) / r
        # v_fr (前右)
        v_fr = (vx + vy + (L + W) * omega) / r
        # v_rl (后左)
        v_rl = (vx + vy - (L + W) * omega) / r
        # v_rr (后右)
        v_rr = (vx - vy + (L + W) * omega) / r
        
        return np.array([v_fl, v_fr, v_rl, v_rr])
