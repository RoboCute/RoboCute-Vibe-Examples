# -*- coding: utf-8 -*-
"""
机器人主模块 - 整合底盘、路径规划和地图功能

提供完整的机器人仿真控制接口。
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any, Tuple
import numpy as np

try:
    import robocute as rbc
    import robocute.rbc_ext as re
    import robocute.rbc_ext.luisa as lc
    RBC_AVAILABLE = True
except ImportError:
    RBC_AVAILABLE = False

from chassis import (
    ChassisType, ChassisBase, DifferentialChassis,
    AckermannChassis, TrackedChassis, MecanumChassis,
    Pose2D, Velocity
)
from path_planner import PathPlanner, Path, PathPoint, AStarPlanner
from map_editor import MapEditor


@dataclass
class RobotConfig:
    """机器人配置"""
    name: str = "robot"
    chassis_type: ChassisType = ChassisType.DIFFERENTIAL
    wheel_radius: float = 0.05
    wheel_base: float = 0.3
    track_width: float = 0.3
    max_linear_speed: float = 1.0  # m/s
    max_angular_speed: float = 1.0  # rad/s
    position_tolerance: float = 0.1  # m
    angle_tolerance: float = 0.1  # rad


class Robot:
    """
    机器人主类
    
    整合底盘控制、路径规划和地图交互功能。
    """
    
    def __init__(self, config: Optional[RobotConfig] = None):
        """
        初始化机器人
        
        Args:
            config: 机器人配置
        """
        self._config = config or RobotConfig()
        self._chassis = self._create_chassis()
        self._planner: Optional[PathPlanner] = None
        self._current_path: Optional[Path] = None
        self._path_index = 0
        self._is_moving = False
        self._goal_reached = False
        
        # World 相关
        self._scene = None
        self._entity = None
        self._transform_comp = None
        self._data_comp = None
        
        # 回调函数
        self._callbacks: Dict[str, List[Callable]] = {
            'on_goal_reached': [],
            'on_path_complete': [],
            'on_collision': [],
        }
        
        # 运行状态
        self._running = False
        self._last_time = 0.0
    
    def _create_chassis(self) -> ChassisBase:
        """根据配置创建底盘"""
        cfg = self._config
        
        if cfg.chassis_type == ChassisType.DIFFERENTIAL:
            return DifferentialChassis(cfg.wheel_radius, cfg.wheel_base)
        elif cfg.chassis_type == ChassisType.ACKERMANN:
            return AckermannChassis(cfg.wheel_radius, cfg.wheel_base, cfg.track_width)
        elif cfg.chassis_type == ChassisType.TRACKED:
            return TrackedChassis(cfg.wheel_radius, cfg.wheel_base)
        elif cfg.chassis_type == ChassisType.MECANUM:
            return MecanumChassis(cfg.wheel_radius, cfg.wheel_base, cfg.track_width)
        else:
            raise ValueError(f"Unknown chassis type: {cfg.chassis_type}")
    
    @property
    def config(self) -> RobotConfig:
        """获取配置"""
        return self._config
    
    @property
    def chassis(self) -> ChassisBase:
        """获取底盘"""
        return self._chassis
    
    @property
    def pose(self) -> Pose2D:
        """获取当前位姿"""
        return self._chassis.pose
    
    @property
    def velocity(self) -> Velocity:
        """获取当前速度"""
        return self._chassis.velocity
    
    @property
    def is_moving(self) -> bool:
        """是否正在移动"""
        return self._is_moving
    
    def set_planner(self, planner: PathPlanner):
        """设置路径规划器"""
        self._planner = planner
    
    def set_map(self, map_editor: MapEditor):
        """设置地图"""
        if self._planner:
            self._planner.set_map(map_editor.data, map_editor.resolution,
                                  (map_editor.metadata.origin_x, map_editor.metadata.origin_y))
    
    def create_in_world(self, scene, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        """
        在 World 中创建机器人实体
        
        Args:
            scene: World Scene
            x, y, theta: 初始位姿
        """
        if not RBC_AVAILABLE:
            print("Warning: RoboCute not available, running in simulation mode")
            self._chassis.pose = Pose2D(x, y, theta)
            return
        
        self._scene = scene
        self._entity = scene.add_entity()
        self._entity.set_name(self._config.name)
        
        # 添加 TransformComponent
        self._transform_comp = re.world.TransformComponent(
            self._entity.add_component("TransformComponent")
        )
        
        # 添加 DataComponent 用于 tick 回调
        self._data_comp = re.world.DataComponent(
            self._entity.add_component("DataComponent")
        )
        
        # 设置初始位姿
        self._chassis.set_entity(self._entity, self._transform_comp)
        self._chassis.pose = Pose2D(x, y, theta)
    
    def set_pose(self, x: float, y: float, theta: float = 0.0):
        """设置位姿"""
        self._chassis.pose = Pose2D(x, y, theta)
    
    def set_velocity(self, vx: float, vy: float = 0.0, omega: float = 0.0):
        """
        设置速度
        
        Args:
            vx: 前进速度 (m/s)
            vy: 横向速度 (m/s，仅麦轮有效)
            omega: 角速度 (rad/s)
        """
        # 限制最大速度
        vx = max(-self._config.max_linear_speed, min(self._config.max_linear_speed, vx))
        omega = max(-self._config.max_angular_speed, min(self._config.max_angular_speed, omega))
        
        self._chassis.set_velocity(vx, vy, omega)
    
    def stop(self):
        """停止机器人"""
        self._chassis.set_velocity(0.0, 0.0, 0.0)
        self._is_moving = False
    
    def move_to(self, x: float, y: float, theta: Optional[float] = None) -> bool:
        """
        移动到目标位置
        
        Args:
            x, y: 目标位置
            theta: 目标朝向 (可选)
            
        Returns:
            是否成功开始移动
        """
        if self._planner is None:
            # 无规划器时直接前往
            self._current_path = Path([PathPoint(self.pose.x, self.pose.y, self.pose.theta),
                                       PathPoint(x, y, theta or self.pose.theta)])
        else:
            # 使用路径规划
            start = PathPoint(self.pose.x, self.pose.y, self.pose.theta)
            goal = PathPoint(x, y, theta or self.pose.theta)
            
            path = self._planner.plan(start, goal)
            if path is None:
                print(f"Path planning failed from ({start.x}, {start.y}) to ({goal.x}, {goal.y})")
                return False
            
            self._current_path = path
        
        self._path_index = 0
        self._is_moving = True
        self._goal_reached = False
        return True
    
    def follow_path(self, path: Path):
        """
        跟随指定路径
        
        Args:
            path: 要跟随的路径
        """
        self._current_path = path
        self._path_index = 0
        self._is_moving = True
        self._goal_reached = False
    
    def update(self, dt: float):
        """
        更新机器人状态
        
        Args:
            dt: 时间步长 (s)
        """
        if not self._is_moving:
            return
        
        # 更新底盘里程计
        self._chassis.update_odometry(dt)
        
        # 路径跟随
        if self._current_path and not self._current_path.is_empty():
            self._update_path_following(dt)
        
        self._last_time = time.time()
    
    def _update_path_following(self, dt: float):
        """更新路径跟随"""
        if self._path_index >= len(self._current_path.points):
            # 路径完成
            self.stop()
            self._goal_reached = True
            self._is_moving = False
            self._trigger_callback('on_path_complete')
            self._trigger_callback('on_goal_reached')
            return
        
        # 获取目标点
        target = self._current_path.points[self._path_index]
        current = self.pose
        
        # 计算到目标点的距离和角度
        dx = target.x - current.x
        dy = target.y - current.y
        dist = math.sqrt(dx * dx + dy * dy)
        target_angle = math.atan2(dy, dx)
        angle_diff = self._normalize_angle(target_angle - current.theta)
        
        # 检查是否到达当前路径点
        if dist < self._config.position_tolerance:
            self._path_index += 1
            if self._path_index >= len(self._current_path.points):
                self.stop()
                self._goal_reached = True
                self._is_moving = False
                self._trigger_callback('on_path_complete')
                self._trigger_callback('on_goal_reached')
            return
        
        # 计算控制指令
        vx, vy, omega = self._compute_control(dist, angle_diff, target)
        self.set_velocity(vx, vy, omega)
    
    def _compute_control(self, dist: float, angle_diff: float, 
                         target: PathPoint) -> Tuple[float, float, float]:
        """
        计算控制指令
        
        Args:
            dist: 到目标点的距离
            angle_diff: 角度差
            target: 目标点
            
        Returns:
            (vx, vy, omega)
        """
        # 改进的 P 控制器
        Kp_v = 1.0
        Kp_w = 2.5
        
        # 计算速度因子 (角度偏差大时减速，但不完全停止)
        angle_factor = max(0.1, math.cos(angle_diff))  # 至少保持 10% 速度
        
        # 前进速度
        vx = min(Kp_v * dist, self._config.max_linear_speed) * angle_factor
        
        # 角速度 - 快速转向
        omega = Kp_w * angle_diff
        omega = max(-self._config.max_angular_speed, 
                    min(self._config.max_angular_speed, omega))
        
        # 麦轮底盘可以横向移动
        vy = 0.0
        if self._config.chassis_type == ChassisType.MECANUM:
            # 简单的横向控制
            pass
        
        return vx, vy, omega
    
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def register_callback(self, event: str, callback: Callable):
        """
        注册事件回调
        
        Args:
            event: 事件名称 ('on_goal_reached', 'on_path_complete', 'on_collision')
            callback: 回调函数
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """触发回调"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Callback error: {e}")
    
    def is_at_goal(self, goal: Optional[PathPoint] = None) -> bool:
        """
        检查是否到达目标
        
        Args:
            goal: 目标点，None 表示当前路径终点
            
        Returns:
            是否到达
        """
        if goal is None:
            return self._goal_reached
        
        dist = math.sqrt((self.pose.x - goal.x) ** 2 + (self.pose.y - goal.y) ** 2)
        return dist < self._config.position_tolerance
    
    def get_path_progress(self) -> float:
        """
        获取路径进度
        
        Returns:
            0.0 - 1.0 的进度值
        """
        if not self._current_path or self._current_path.is_empty():
            return 1.0
        return self._path_index / len(self._current_path.points)
    
    def get_remaining_distance(self) -> float:
        """
        获取剩余路径长度
        
        Returns:
            剩余距离 (米)
        """
        if not self._current_path or self._current_path.is_empty():
            return 0.0
        
        remaining = 0.0
        current_pos = PathPoint(self.pose.x, self.pose.y)
        
        # 到当前路径点的距离
        if self._path_index < len(self._current_path.points):
            remaining += current_pos.distance_to(self._current_path.points[self._path_index])
        
        # 剩余路径点之间的距离
        for i in range(self._path_index, len(self._current_path.points) - 1):
            remaining += self._current_path.points[i].distance_to(
                self._current_path.points[i + 1]
            )
        
        return remaining
    
    def destroy(self):
        """销毁机器人"""
        self.stop()
        
        if self._entity and RBC_AVAILABLE:
            # 从场景中移除实体
            if self._scene:
                self._scene.remove_entity(self._entity)
        
        self._entity = None
        self._transform_comp = None
        self._data_comp = None


class RobotFleet:
    """
    机器人群组管理
    
    管理多个机器人的协同工作。
    """
    
    def __init__(self):
        self._robots: Dict[str, Robot] = {}
        self._tasks: Dict[str, Any] = {}
    
    def add_robot(self, robot: Robot) -> str:
        """
        添加机器人
        
        Args:
            robot: 机器人实例
            
        Returns:
            机器人 ID
        """
        robot_id = robot.config.name
        self._robots[robot_id] = robot
        return robot_id
    
    def remove_robot(self, robot_id: str):
        """移除机器人"""
        if robot_id in self._robots:
            self._robots[robot_id].destroy()
            del self._robots[robot_id]
    
    def get_robot(self, robot_id: str) -> Optional[Robot]:
        """获取机器人"""
        return self._robots.get(robot_id)
    
    def update_all(self, dt: float):
        """更新所有机器人"""
        for robot in self._robots.values():
            robot.update(dt)
    
    def stop_all(self):
        """停止所有机器人"""
        for robot in self._robots.values():
            robot.stop()
    
    def assign_task(self, robot_id: str, target_x: float, target_y: float, 
                    target_theta: Optional[float] = None) -> bool:
        """
        分配任务给机器人
        
        Args:
            robot_id: 机器人 ID
            target_x, target_y: 目标位置
            target_theta: 目标朝向
            
        Returns:
            是否成功
        """
        robot = self._robots.get(robot_id)
        if robot is None:
            return False
        
        return robot.move_to(target_x, target_y, target_theta)
    
    def get_all_poses(self) -> Dict[str, Pose2D]:
        """获取所有机器人的位姿"""
        return {rid: robot.pose for rid, robot in self._robots.items()}
    
    def clear(self):
        """清空所有机器人"""
        for robot in self._robots.values():
            robot.destroy()
        self._robots.clear()
        self._tasks.clear()


if __name__ == "__main__":
    """简单的机器人演示"""
    print("=" * 50)
    print("机器人模块演示")
    print("=" * 50)
    
    # 创建差速机器人
    config = RobotConfig(
        name="demo_robot",
        chassis_type=ChassisType.DIFFERENTIAL,
        wheel_radius=0.05,
        wheel_base=0.3,
        max_linear_speed=1.0,
        max_angular_speed=1.0
    )
    
    robot = Robot(config)
    print(f"\n创建机器人: {robot.config.name}")
    print(f"底盘类型: {robot.config.chassis_type.name}")
    
    # 设置初始位姿
    robot.set_pose(0.0, 0.0, 0.0)
    print(f"初始位姿: ({robot.pose.x}, {robot.pose.y}, {robot.pose.theta:.2f})")
    
    # 设置速度并更新
    print("\n设置速度: vx=0.5 m/s, omega=0.3 rad/s")
    robot.set_velocity(0.5, 0.0, 0.3)
    
    # 模拟运动
    dt = 0.1
    print("\n模拟运动:")
    for i in range(20):
        robot.chassis.update_odometry(dt)
        pose = robot.pose
        print(f"  t={i*dt:.1f}s: x={pose.x:.3f}, y={pose.y:.3f}, theta={pose.theta:.3f}")
    
    # 停止
    robot.stop()
    print(f"\n停止后速度: ({robot.velocity.vx}, {robot.velocity.omega})")
    
    print("\n" + "=" * 50)
    print("演示完成!")
    print("=" * 50)
