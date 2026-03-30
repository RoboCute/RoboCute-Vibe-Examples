# -*- coding: utf-8 -*-
"""
底盘模块 - 提供各种机器底盘类型的运动学模型

支持的底盘类型:
- Differential: 差速底盘(两轮差速)
- Ackermann: 阿克曼底盘(汽车转向模型)
- Tracked: 履带底盘(坦克模型)
- Mecanum: 麦轮底盘(全向移动)
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, List, Optional
import numpy as np

try:
    import robocute.rbc_ext.luisa as lc
    LUISA_AVAILABLE = True
except ImportError:
    LUISA_AVAILABLE = False


class ChassisType(Enum):
    """底盘类型枚举"""
    DIFFERENTIAL = auto()  # 差速底盘
    ACKERMANN = auto()     # 阿克曼底盘
    TRACKED = auto()       # 履带底盘
    MECANUM = auto()       # 麦轮底盘


@dataclass
class Pose2D:
    """2D位姿 (x, y, theta)"""
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0  # 朝向角度 (弧度)
    
    def to_array(self) -> np.ndarray:
        """转换为 numpy 数组"""
        return np.array([self.x, self.y, self.theta])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> Pose2D:
        """从 numpy 数组创建"""
        return Pose2D(float(arr[0]), float(arr[1]), float(arr[2]))


@dataclass
class WheelSpeeds:
    """轮子速度"""
    left: float = 0.0
    right: float = 0.0
    front_left: float = 0.0
    front_right: float = 0.0
    rear_left: float = 0.0
    rear_right: float = 0.0


@dataclass
class Velocity:
    """机器人速度 (vx, vy, omega)"""
    vx: float = 0.0     # 前进速度 (m/s)
    vy: float = 0.0     # 横向速度 (m/s)
    omega: float = 0.0  # 角速度 (rad/s)


class ChassisBase(ABC):
    """
    底盘基类
    
    所有底盘类型的抽象基类，定义通用接口。
    """
    
    def __init__(self, chassis_type: ChassisType, wheel_radius: float = 0.05, 
                 wheel_base: float = 0.3):
        """
        初始化底盘
        
        Args:
            chassis_type: 底盘类型
            wheel_radius: 轮子半径 (m)
            wheel_base: 轮距 (m，两轮之间距离)
        """
        self._type = chassis_type
        self._wheel_radius = wheel_radius
        self._wheel_base = wheel_base
        self._pose = Pose2D()
        self._velocity = Velocity()
        self._wheel_speeds = WheelSpeeds()
        self._entity = None  # World Entity handle
        self._transform_comp = None  # TransformComponent handle
        
    @property
    def type(self) -> ChassisType:
        """获取底盘类型"""
        return self._type
    
    @property
    def pose(self) -> Pose2D:
        """获取当前位姿"""
        return self._pose
    
    @pose.setter
    def pose(self, value: Pose2D):
        """设置位姿"""
        self._pose = value
        self._update_transform()
    
    @property
    def velocity(self) -> Velocity:
        """获取当前速度"""
        return self._velocity
    
    @property
    def wheel_speeds(self) -> WheelSpeeds:
        """获取轮子速度"""
        return self._wheel_speeds
    
    def set_entity(self, entity, transform_comp=None):
        """
        绑定 World Entity
        
        Args:
            entity: Entity handle
            transform_comp: TransformComponent handle
        """
        self._entity = entity
        self._transform_comp = transform_comp
        self._update_transform()
    
    def _update_transform(self):
        """更新 TransformComponent"""
        if self._transform_comp and LUISA_AVAILABLE:
            # 更新位置
            self._transform_comp.set_pos(
                lc.double3(self._pose.x, 0.0, self._pose.y),
                recursive=False
            )
            # 更新旋转 (绕 Y 轴)
            q = self._euler_to_quaternion(0.0, self._pose.theta, 0.0)
            self._transform_comp.set_rotation(q, recursive=False)
    
    def _euler_to_quaternion(self, roll: float, yaw: float, pitch: float) -> lc.float4:
        """欧拉角转四元数"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return lc.float4(x, y, z, w) if LUISA_AVAILABLE else (x, y, z, w)
    
    @abstractmethod
    def inverse_kinematics(self, vx: float, vy: float, omega: float) -> WheelSpeeds:
        """
        逆运动学: 速度 -> 轮速
        
        Args:
            vx: 前进速度 (m/s)
            vy: 横向速度 (m/s)
            omega: 角速度 (rad/s)
            
        Returns:
            各轮速度 (rad/s)
        """
        pass
    
    @abstractmethod
    def forward_kinematics(self, wheel_speeds: WheelSpeeds) -> Velocity:
        """
        正运动学: 轮速 -> 速度
        
        Args:
            wheel_speeds: 各轮速度
            
        Returns:
            机器人速度 (vx, vy, omega)
        """
        pass
    
    @abstractmethod
    def update_odometry(self, dt: float):
        """
        更新里程计
        
        Args:
            dt: 时间步长 (s)
        """
        pass
    
    def set_velocity(self, vx: float, vy: float, omega: float):
        """
        设置目标速度
        
        Args:
            vx: 前进速度 (m/s)
            vy: 横向速度 (m/s)
            omega: 角速度 (rad/s)
        """
        self._velocity = Velocity(vx, vy, omega)
        self._wheel_speeds = self.inverse_kinematics(vx, vy, omega)


class DifferentialChassis(ChassisBase):
    """
    差速底盘
    
    两轮差速驱动模型，广泛用于移动机器人。
    
    运动学模型:
        vx = (vr + vl) * r / 2
        omega = (vr - vl) * r / L
    
    其中:
        vr, vl: 右/左轮角速度 (rad/s)
        r: 轮子半径 (m)
        L: 轮距 (m)
    """
    
    def __init__(self, wheel_radius: float = 0.05, wheel_base: float = 0.3):
        super().__init__(ChassisType.DIFFERENTIAL, wheel_radius, wheel_base)
    
    def inverse_kinematics(self, vx: float, vy: float, omega: float) -> WheelSpeeds:
        """差速底盘逆运动学"""
        r = self._wheel_radius
        L = self._wheel_base
        
        # 差速底盘无法横向移动
        vl = (vx - omega * L / 2) / r
        vr = (vx + omega * L / 2) / r
        
        return WheelSpeeds(left=vl, right=vr)
    
    def forward_kinematics(self, wheel_speeds: WheelSpeeds) -> Velocity:
        """差速底盘正运动学"""
        r = self._wheel_radius
        L = self._wheel_base
        vl = wheel_speeds.left
        vr = wheel_speeds.right
        
        vx = (vr + vl) * r / 2
        vy = 0.0  # 差速底盘无法横向移动
        omega = (vr - vl) * r / L
        
        return Velocity(vx, vy, omega)
    
    def update_odometry(self, dt: float):
        """更新差速底盘里程计"""
        v = self._velocity
        
        # 更新位姿
        if abs(v.omega) < 1e-6:
            # 直线运动
            self._pose.x += v.vx * dt * math.cos(self._pose.theta)
            self._pose.y += v.vx * dt * math.sin(self._pose.theta)
        else:
            # 圆弧运动
            R = v.vx / v.omega  # 转弯半径
            theta_new = self._pose.theta + v.omega * dt
            self._pose.x += R * (math.sin(theta_new) - math.sin(self._pose.theta))
            self._pose.y -= R * (math.cos(theta_new) - math.cos(self._pose.theta))
            self._pose.theta = theta_new
        
        self._update_transform()


class AckermannChassis(ChassisBase):
    """
    阿克曼底盘
    
    汽车转向模型，前轮转向后轮驱动。
    
    运动学模型:
        转向角 delta 满足: tan(delta) = L / R
        其中 R 为转弯半径
    """
    
    def __init__(self, wheel_radius: float = 0.05, wheel_base: float = 0.3,
                 track_width: float = 0.25, max_steering_angle: float = math.pi / 4):
        super().__init__(ChassisType.ACKERMANN, wheel_radius, wheel_base)
        self._track_width = track_width  # 轴距 (前轮后轮距离)
        self._max_steering_angle = max_steering_angle
        self._steering_angle = 0.0
    
    @property
    def steering_angle(self) -> float:
        """获取当前转向角"""
        return self._steering_angle
    
    def set_steering(self, angle: float):
        """
        设置转向角
        
        Args:
            angle: 转向角 (rad)，限制在 [-max, max]
        """
        self._steering_angle = max(-self._max_steering_angle, 
                                   min(self._max_steering_angle, angle))
    
    def inverse_kinematics(self, vx: float, vy: float, omega: float) -> WheelSpeeds:
        """阿克曼底盘逆运动学"""
        r = self._wheel_radius
        L = self._wheel_base
        
        # 阿克曼底盘无法横向移动
        if abs(vx) < 1e-6:
            # 原地转向
            self._steering_angle = math.pi / 2 if omega > 0 else -math.pi / 2
        elif abs(omega) < 1e-6:
            # 直线行驶
            self._steering_angle = 0.0
        else:
            # 计算转向角
            R = vx / omega
            self._steering_angle = math.atan(L / R)
            self._steering_angle = max(-self._max_steering_angle,
                                       min(self._max_steering_angle, self._steering_angle))
        
        # 计算轮速
        v_rear = vx / r  # 后轮速度
        
        # 内侧和外侧前轮速度不同
        if abs(self._steering_angle) < 1e-6:
            v_front_left = v_front_right = v_rear
        else:
            R = L / math.tan(self._steering_angle)
            R_inner = R - self._track_width / 2
            R_outer = R + self._track_width / 2
            v_front_left = v_rear * R_inner / R
            v_front_right = v_rear * R_outer / R
        
        return WheelSpeeds(
            front_left=v_front_left,
            front_right=v_front_right,
            left=v_rear,
            right=v_rear
        )
    
    def forward_kinematics(self, wheel_speeds: WheelSpeeds) -> Velocity:
        """阿克曼底盘正运动学"""
        r = self._wheel_radius
        v_rear = (wheel_speeds.left + wheel_speeds.right) * r / 2
        
        vx = v_rear * math.cos(self._steering_angle)
        vy = 0.0
        omega = v_rear * math.sin(self._steering_angle) / self._wheel_base
        
        return Velocity(vx, vy, omega)
    
    def update_odometry(self, dt: float):
        """更新阿克曼底盘里程计"""
        v = self._velocity
        
        # 更新位姿
        self._pose.x += v.vx * dt * math.cos(self._pose.theta)
        self._pose.y += v.vx * dt * math.sin(self._pose.theta)
        self._pose.theta += v.omega * dt
        
        self._update_transform()


class TrackedChassis(ChassisBase):
    """
    履带底盘
    
    坦克模型，两侧履带独立驱动。
    运动学与差速底盘类似，但考虑履带滑动。
    """
    
    def __init__(self, wheel_radius: float = 0.05, wheel_base: float = 0.3,
                 slip_factor: float = 0.9):
        super().__init__(ChassisType.TRACKED, wheel_radius, wheel_base)
        self._slip_factor = slip_factor  # 滑动系数 (0-1)
    
    def inverse_kinematics(self, vx: float, vy: float, omega: float) -> WheelSpeeds:
        """履带底盘逆运动学"""
        r = self._wheel_radius
        L = self._wheel_base
        s = self._slip_factor
        
        # 履带底盘无法横向移动
        vl = (vx - omega * L / 2) / (r * s)
        vr = (vx + omega * L / 2) / (r * s)
        
        return WheelSpeeds(left=vl, right=vr)
    
    def forward_kinematics(self, wheel_speeds: WheelSpeeds) -> Velocity:
        """履带底盘正运动学"""
        r = self._wheel_radius
        L = self._wheel_base
        s = self._slip_factor
        vl = wheel_speeds.left * s
        vr = wheel_speeds.right * s
        
        vx = (vr + vl) * r / 2
        vy = 0.0
        omega = (vr - vl) * r / L
        
        return Velocity(vx, vy, omega)
    
    def update_odometry(self, dt: float):
        """更新履带底盘里程计"""
        # 与差速底盘相同
        v = self._velocity
        
        if abs(v.omega) < 1e-6:
            self._pose.x += v.vx * dt * math.cos(self._pose.theta)
            self._pose.y += v.vx * dt * math.sin(self._pose.theta)
        else:
            R = v.vx / v.omega
            theta_new = self._pose.theta + v.omega * dt
            self._pose.x += R * (math.sin(theta_new) - math.sin(self._pose.theta))
            self._pose.y -= R * (math.cos(theta_new) - math.cos(self._pose.theta))
            self._pose.theta = theta_new
        
        self._update_transform()


class MecanumChassis(ChassisBase):
    r"""
    麦轮底盘
    
    麦克纳姆轮全向移动底盘，可实现任意方向平移和旋转。
    
    轮子布局:
        FL: Front Left  (\)
        FR: Front Right (/)
        RL: Rear Left   (/)
        RR: Rear Right  (\)
    
    运动学模型:
        vx = (vFL + vFR + vRL + vRR) * r / 4
        vy = (-vFL + vFR + vRL - vRR) * r / 4
        omega = (-vFL + vFR - vRL + vRR) * r / (2 * (L + W))
    """
    
    def __init__(self, wheel_radius: float = 0.05, wheel_base: float = 0.3,
                 track_width: float = 0.3):
        super().__init__(ChassisType.MECANUM, wheel_radius, wheel_base)
        self._track_width = track_width  # 左右轮距
    
    def inverse_kinematics(self, vx: float, vy: float, omega: float) -> WheelSpeeds:
        """麦轮底盘逆运动学"""
        r = self._wheel_radius
        L = self._wheel_base
        W = self._track_width
        
        # 计算各轮速度
        vFL = (vx - vy - omega * (L + W) / 2) / r
        vFR = (vx + vy + omega * (L + W) / 2) / r
        vRL = (vx + vy - omega * (L + W) / 2) / r
        vRR = (vx - vy + omega * (L + W) / 2) / r
        
        return WheelSpeeds(
            front_left=vFL,
            front_right=vFR,
            rear_left=vRL,
            rear_right=vRR
        )
    
    def forward_kinematics(self, wheel_speeds: WheelSpeeds) -> WheelSpeeds:
        """麦轮底盘正运动学"""
        r = self._wheel_radius
        L = self._wheel_base
        W = self._track_width
        
        vFL = wheel_speeds.front_left
        vFR = wheel_speeds.front_right
        vRL = wheel_speeds.rear_left
        vRR = wheel_speeds.rear_right
        
        vx = (vFL + vFR + vRL + vRR) * r / 4
        vy = (-vFL + vFR + vRL - vRR) * r / 4
        omega = (-vFL + vFR - vRL + vRR) * r / (2 * (L + W))
        
        return Velocity(vx, vy, omega)
    
    def update_odometry(self, dt: float):
        """更新麦轮底盘里程计"""
        v = self._velocity
        
        # 在世界坐标系中更新位姿
        cos_theta = math.cos(self._pose.theta)
        sin_theta = math.sin(self._pose.theta)
        
        # 速度转换到世界坐标系
        vx_world = v.vx * cos_theta - v.vy * sin_theta
        vy_world = v.vx * sin_theta + v.vy * cos_theta
        
        self._pose.x += vx_world * dt
        self._pose.y += vy_world * dt
        self._pose.theta += v.omega * dt
        
        self._update_transform()


def create_chassis(chassis_type: ChassisType, **kwargs) -> BaseChassis:
    """
    创建底盘实例的工厂函数
    
    Args:
        chassis_type: 底盘类型
        **kwargs: 底盘特定参数
            - wheel_base: 轮距 (差速、麦轮)
            - wheel_radius: 轮子半径
            - track_width: 轮距宽度 (阿克曼)
            - max_steering_angle: 最大转向角 (阿克曼, 默认 30°)
    
    Returns:
        底盘实例
        
    Examples:
        >>> chassis = create_chassis(ChassisType.DIFFERENTIAL, wheel_base=0.3)
        >>> chassis = create_chassis(ChassisType.ACKERMANN, wheel_base=0.4, track_width=0.3)
    """
    if chassis_type == ChassisType.DIFFERENTIAL:
        return DifferentialChassis(
            wheel_base=kwargs.get('wheel_base', 0.3),
            wheel_radius=kwargs.get('wheel_radius', 0.05)
        )
    elif chassis_type == ChassisType.ACKERMANN:
        return AckermannChassis(
            wheel_base=kwargs.get('wheel_base', 0.4),
            track_width=kwargs.get('track_width', 0.3),
            wheel_radius=kwargs.get('wheel_radius', 0.05),
            max_steering_angle=kwargs.get('max_steering_angle', math.radians(30))
        )
    elif chassis_type == ChassisType.TRACKED:
        return TrackedChassis(
            wheel_base=kwargs.get('wheel_base', 0.3),
            wheel_radius=kwargs.get('wheel_radius', 0.05),
            slip_factor=kwargs.get('slip_factor', 0.9)
        )
    elif chassis_type == ChassisType.MECANUM:
        return MecanumChassis(
            wheel_base=kwargs.get('wheel_base', 0.3),
            wheel_radius=kwargs.get('wheel_radius', 0.05)
        )
    else:
        raise ValueError(f"不支持的底盘类型: {chassis_type}")
