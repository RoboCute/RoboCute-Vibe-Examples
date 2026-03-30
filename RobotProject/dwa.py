# -*- coding: utf-8 -*-
"""
DWA (Dynamic Window Approach) Algorithm

动态窗口法路径规划算法实现
用于机器人局部路径规划和动态避障

Features:
    - Differential drive and Ackermann steering kinematics support
    - Dynamic velocity window based on acceleration constraints
    - Multi-objective trajectory evaluation (heading, clearance, velocity)
    - Obstacle distance computation
    - Real-time local path planning

References:
    - Fox, D., Burgard, W., & Thrun, S. (1997). The dynamic window approach to collision avoidance.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Callable, Set, Dict
from dataclasses import dataclass, field
from enum import Enum, auto


class KinematicModel(Enum):
    """运动学模型类型"""
    DIFFERENTIAL_DRIVE = "differential_drive"  # 差速底盘
    ACKERMANN_STEERING = "ackermann_steering"  # 阿克曼底盘


@dataclass
class DWAConfig:
    """DWA 算法配置参数"""
    
    # 速度约束 (m/s, rad/s)
    max_linear_velocity: float = 1.0        # 最大线速度
    min_linear_velocity: float = 0.0        # 最小线速度 (允许倒车则设为负值)
    max_angular_velocity: float = 2.0       # 最大角速度
    
    # 加速度约束 (m/s², rad/s²)
    max_linear_accel: float = 2.0           # 最大线加速度
    max_angular_accel: float = 4.0          # 最大角加速度
    
    # 采样参数
    velocity_resolution: int = 20           # 线速度采样数量
    angular_resolution: int = 20            # 角速度采样数量
    
    # 预测参数
    dt: float = 0.1                         # 时间步长 (s)
    predict_time: float = 3.0               # 预测时间 (s)
    
    # 评价函数权重
    weight_heading: float = 1.0             # 航向角权重
    weight_obstacle: float = 2.0            # 障碍物权重
    weight_velocity: float = 0.5            # 速度权重
    weight_distance: float = 0.3            # 距离权重
    
    # 安全参数
    robot_radius: float = 0.3               # 机器人半径 (m)
    safety_margin: float = 0.1              # 安全边距 (m)
    
    # 停止条件
    goal_tolerance: float = 0.2             # 到达目标容差 (m)
    goal_angle_tolerance: float = 0.1       # 到达目标角度容差 (rad)
    
    # 底盘参数 (用于阿克曼模型)
    wheelbase: float = 0.5                  # 轴距 (m)


@dataclass
class RobotState:
    """机器人状态"""
    x: float = 0.0          # 位置 x (m)
    y: float = 0.0          # 位置 y (m)
    theta: float = 0.0      # 朝向角 (rad)
    v: float = 0.0          # 线速度 (m/s)
    omega: float = 0.0      # 角速度 (rad/s)
    
    def get_position(self) -> Tuple[float, float]:
        """获取位置"""
        return (self.x, self.y)
    
    def get_pose(self) -> Tuple[float, float, float]:
        """获取位姿 (x, y, theta)"""
        return (self.x, self.y, self.theta)
    
    def copy(self) -> 'RobotState':
        """复制状态"""
        return RobotState(self.x, self.y, self.theta, self.v, self.omega)


@dataclass
class Trajectory:
    """轨迹数据结构"""
    states: List[RobotState] = field(default_factory=list)
    control: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))  # (v, omega)
    score: float = 0.0
    
    # 评分组成
    heading_score: float = 0.0
    obstacle_score: float = 0.0
    velocity_score: float = 0.0
    distance_score: float = 0.0
    
    def __len__(self) -> int:
        return len(self.states)
    
    def get_waypoints(self) -> List[Tuple[float, float]]:
        """获取轨迹路点"""
        return [(s.x, s.y) for s in self.states]
    
    def get_final_position(self) -> Optional[Tuple[float, float]]:
        """获取终点位置"""
        if self.states:
            return (self.states[-1].x, self.states[-1].y)
        return None


@dataclass
class DWAResult:
    """DWA 规划结果"""
    success: bool = False
    best_velocity: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    best_trajectory: Optional[Trajectory] = None
    all_trajectories: List[Trajectory] = field(default_factory=list)
    min_obstacle_dist: float = float('inf')
    goal_reached: bool = False
    
    def __repr__(self) -> str:
        status = "Success" if self.success else "Failed"
        v, omega = self.best_velocity
        return f"DWAResult({status}, v={v:.2f}, ω={omega:.2f}, goal_reached={self.goal_reached})"


class DWAPlanner:
    """
    DWA (Dynamic Window Approach) 动态窗口法路径规划器
    
    DWA 算法是一种局部路径规划方法，适用于动态环境下的实时避障。
    它通过在速度空间 (v, ω) 中采样，评估每条轨迹，选择最优的控制指令。
    
    算法流程:
    1. 根据当前速度和加速度限制计算动态窗口
    2. 在窗口内采样多组速度 (v, ω)
    3. 预测每组速度对应的轨迹
    4. 使用评价函数评估轨迹
    5. 选择评分最高的轨迹对应的速度作为控制指令
    
    Attributes:
        config: DWA 配置参数
        collision_check: 碰撞检测函数
        kinematic_model: 运动学模型类型
    
    Example:
        >>> # 创建简单的碰撞检测函数
        >>> obstacles = [(3.0, 3.0), (4.0, 4.0)]
        >>> def collision_fn(x, y):
        ...     for ox, oy in obstacles:
        ...         if math.hypot(x - ox, y - oy) < 0.5:
        ...             return True
        ...     return False
        >>> 
        >>> # 创建 DWA 规划器
        >>> planner = DWAPlanner(collision_fn, DWAConfig())
        >>> 
        >>> # 执行规划
        >>> state = RobotState(x=0, y=0, theta=0, v=0, omega=0)
        >>> result = planner.plan(state, goal=(5, 5))
        >>> print(result.best_velocity)
    """
    
    def __init__(
        self,
        collision_check: Callable[[float, float], bool],
        config: Optional[DWAConfig] = None,
        kinematic_model: KinematicModel = KinematicModel.DIFFERENTIAL_DRIVE,
        goal_checker: Optional[Callable[[RobotState, Tuple[float, float]], bool]] = None
    ):
        """
        初始化 DWA 规划器
        
        Args:
            collision_check: 碰撞检测函数，输入 (x, y)，返回 True 表示碰撞
            config: DWA 配置参数，默认使用默认配置
            kinematic_model: 运动学模型类型
            goal_checker: 自定义目标检测函数 (可选)
        """
        self.config = config or DWAConfig()
        self.collision_check = collision_check
        self.kinematic_model = kinematic_model
        self.goal_checker = goal_checker
        
        # 缓存动态窗口，避免重复计算
        self._dynamic_window_cache: Optional[Tuple[float, float, float, float]] = None
        self._cache_key: Optional[Tuple[float, float]] = None
    
    def plan(
        self,
        current_state: RobotState,
        goal: Tuple[float, float],
        obstacles: Optional[List[Tuple[float, float]]] = None,
        debug: bool = False
    ) -> DWAResult:
        """
        执行 DWA 路径规划
        
        Args:
            current_state: 当前机器人状态
            goal: 目标位置 (x, y)
            obstacles: 障碍物列表 [(x, y), ...] (可选，用于距离计算)
            debug: 是否输出调试信息
            
        Returns:
            DWAResult 规划结果
        """
        result = DWAResult()
        
        # 检查是否已到达目标
        if self._check_goal_reached(current_state, goal):
            result.success = True
            result.goal_reached = True
            result.best_velocity = (0.0, 0.0)
            return result
        
        # 计算动态窗口
        dynamic_window = self._calculate_dynamic_window(current_state)
        
        # 采样并评估所有可能的速度
        best_trajectory = None
        best_score = -float('inf')
        
        v_samples = np.linspace(dynamic_window[0], dynamic_window[1], 
                                self.config.velocity_resolution)
        omega_samples = np.linspace(dynamic_window[2], dynamic_window[3], 
                                    self.config.angular_resolution)
        
        if debug:
            print(f"Dynamic Window: v=[{dynamic_window[0]:.2f}, {dynamic_window[1]:.2f}], "
                  f"ω=[{dynamic_window[2]:.2f}, {dynamic_window[3]:.2f}]")
            print(f"Samples: {len(v_samples)} x {len(omega_samples)} = {len(v_samples) * len(omega_samples)}")
        
        for v in v_samples:
            for omega in omega_samples:
                # 预测轨迹
                trajectory = self._predict_trajectory(current_state, v, omega)
                
                if len(trajectory) == 0:
                    continue
                
                # 评估轨迹
                score = self._evaluate_trajectory(trajectory, goal, obstacles)
                trajectory.score = score
                
                result.all_trajectories.append(trajectory)
                
                if score > best_score:
                    best_score = score
                    best_trajectory = trajectory
        
        # 设置结果
        if best_trajectory is not None:
            result.success = True
            result.best_velocity = best_trajectory.control
            result.best_trajectory = best_trajectory
            
            # 计算最小障碍物距离
            if obstacles:
                result.min_obstacle_dist = self._calculate_min_obstacle_distance(
                    best_trajectory, obstacles)
        
        if debug and result.success:
            print(f"Best control: v={result.best_velocity[0]:.2f}, "
                  f"ω={result.best_velocity[1]:.2f}, score={best_score:.3f}")
        
        return result
    
    def plan_with_recovery(
        self,
        current_state: RobotState,
        goal: Tuple[float, float],
        obstacles: Optional[List[Tuple[float, float]]] = None,
        max_attempts: int = 3
    ) -> DWAResult:
        """
        带恢复策略的 DWA 规划
        
        当规划失败时，尝试降低速度要求或切换方向
        
        Args:
            current_state: 当前机器人状态
            goal: 目标位置
            obstacles: 障碍物列表
            max_attempts: 最大尝试次数
            
        Returns:
            DWAResult 规划结果
        """
        # 第一次尝试
        result = self.plan(current_state, goal, obstacles)
        
        if result.success:
            return result
        
        # 尝试恢复策略
        original_config = self.config
        
        for attempt in range(1, max_attempts):
            # 创建临时配置
            temp_config = DWAConfig(
                max_linear_velocity=original_config.max_linear_velocity * (0.5 ** attempt),
                max_angular_velocity=original_config.max_angular_velocity,
                max_linear_accel=original_config.max_linear_accel * (0.5 ** attempt),
                max_angular_accel=original_config.max_angular_accel * (0.5 ** attempt),
                velocity_resolution=original_config.velocity_resolution,
                angular_resolution=original_config.angular_resolution * 2,
                dt=original_config.dt,
                predict_time=original_config.predict_time * (1.0 + 0.2 * attempt),
                weight_heading=original_config.weight_heading,
                weight_obstacle=original_config.weight_obstacle * (0.5 ** attempt),
                weight_velocity=0.0,  # 忽略速度评分
                robot_radius=original_config.robot_radius,
                safety_margin=original_config.safety_margin * (0.5 ** attempt)
            )
            
            self.config = temp_config
            result = self.plan(current_state, goal, obstacles)
            
            if result.success:
                break
        
        # 恢复原始配置
        self.config = original_config
        
        return result
    
    def _calculate_dynamic_window(
        self,
        state: RobotState
    ) -> Tuple[float, float, float, float]:
        """
        计算动态窗口
        
        动态窗口由当前速度和加速度限制决定
        
        Args:
            state: 当前机器人状态
            
        Returns:
            (min_v, max_v, min_omega, max_omega) 动态窗口边界
        """
        cache_key = (round(state.v, 3), round(state.omega, 3))
        
        if self._cache_key == cache_key and self._dynamic_window_cache is not None:
            return self._dynamic_window_cache
        
        # 速度限制
        vs = (
            self.config.min_linear_velocity,
            self.config.max_linear_velocity,
            -self.config.max_angular_velocity,
            self.config.max_angular_velocity
        )
        
        # 加速度限制 (在一个 dt 时间内能达到的速度范围)
        vd = (
            state.v - self.config.max_linear_accel * self.config.dt,
            state.v + self.config.max_linear_accel * self.config.dt,
            state.omega - self.config.max_angular_accel * self.config.dt,
            state.omega + self.config.max_angular_accel * self.config.dt
        )
        
        # 动态窗口为两者的交集
        dw = (
            max(vs[0], vd[0]),  # min_v
            min(vs[1], vd[1]),  # max_v
            max(vs[2], vd[2]),  # min_omega
            min(vs[3], vd[3])   # max_omega
        )
        
        # 确保窗口有效
        if dw[0] > dw[1]:
            dw = (dw[1], dw[1], dw[2], dw[3])
        if dw[2] > dw[3]:
            dw = (dw[0], dw[1], dw[3], dw[3])
        
        self._dynamic_window_cache = dw
        self._cache_key = cache_key
        
        return dw
    
    def _predict_trajectory(
        self,
        initial_state: RobotState,
        v: float,
        omega: float
    ) -> Trajectory:
        """
        预测给定速度下的轨迹
        
        Args:
            initial_state: 初始状态
            v: 线速度
            omega: 角速度
            
        Returns:
            Trajectory 预测的轨迹
        """
        trajectory = Trajectory(control=(v, omega))
        
        state = initial_state.copy()
        trajectory.states.append(state.copy())
        
        num_steps = int(self.config.predict_time / self.config.dt)
        
        for _ in range(num_steps):
            # 根据运动学模型更新状态
            if self.kinematic_model == KinematicModel.DIFFERENTIAL_DRIVE:
                state = self._update_differential_drive(state, v, omega)
            elif self.kinematic_model == KinematicModel.ACKERMANN_STEERING:
                state = self._update_ackermann(state, v, omega)
            
            # 检查碰撞
            if self._check_collision(state.x, state.y):
                break
            
            trajectory.states.append(state.copy())
        
        return trajectory
    
    def _update_differential_drive(
        self,
        state: RobotState,
        v: float,
        omega: float
    ) -> RobotState:
        """
        更新差速底盘状态
        
        Args:
            state: 当前状态
            v: 线速度
            omega: 角速度
            
        Returns:
            新状态
        """
        new_state = state.copy()
        
        # 使用差分驱动运动学模型
        new_state.x += v * math.cos(state.theta) * self.config.dt
        new_state.y += v * math.sin(state.theta) * self.config.dt
        new_state.theta += omega * self.config.dt
        new_state.theta = math.atan2(math.sin(new_state.theta), math.cos(new_state.theta))
        new_state.v = v
        new_state.omega = omega
        
        return new_state
    
    def _update_ackermann(
        self,
        state: RobotState,
        v: float,
        omega: float
    ) -> RobotState:
        """
        更新阿克曼底盘状态
        
        Args:
            state: 当前状态
            v: 线速度
            omega: 角速度
            
        Returns:
            新状态
        """
        new_state = state.copy()
        
        # 限制转弯半径 (阿克曼底盘有最小转弯半径)
        if abs(omega) > 1e-6:
            max_omega = v / (self.config.wheelbase / math.tan(math.radians(30)))
            omega = np.clip(omega, -max_omega, max_omega)
        
        new_state.x += v * math.cos(state.theta) * self.config.dt
        new_state.y += v * math.sin(state.theta) * self.config.dt
        new_state.theta += omega * self.config.dt
        new_state.theta = math.atan2(math.sin(new_state.theta), math.cos(new_state.theta))
        new_state.v = v
        new_state.omega = omega
        
        return new_state
    
    def _evaluate_trajectory(
        self,
        trajectory: Trajectory,
        goal: Tuple[float, float],
        obstacles: Optional[List[Tuple[float, float]]]
    ) -> float:
        """
        评估轨迹
        
        综合考虑航向角、障碍物距离、速度和目标距离
        
        Args:
            trajectory: 待评估轨迹
            goal: 目标位置
            obstacles: 障碍物列表
            
        Returns:
            综合评分 (越高越好)
        """
        if len(trajectory) == 0:
            return -float('inf')
        
        final_state = trajectory.states[-1]
        
        # 1. 航向角评分 (heading score)
        heading_score = self._calc_heading_score(final_state, goal)
        trajectory.heading_score = heading_score
        
        # 2. 障碍物评分 (obstacle score)
        obstacle_score = self._calc_obstacle_score(trajectory, obstacles)
        trajectory.obstacle_score = obstacle_score
        
        # 3. 速度评分 (velocity score)
        velocity_score = self._calc_velocity_score(final_state)
        trajectory.velocity_score = velocity_score
        
        # 4. 距离评分 (distance score)
        distance_score = self._calc_distance_score(final_state, goal)
        trajectory.distance_score = distance_score
        
        # 综合评分
        total_score = (
            self.config.weight_heading * heading_score +
            self.config.weight_obstacle * obstacle_score +
            self.config.weight_velocity * velocity_score +
            self.config.weight_distance * distance_score
        )
        
        return total_score
    
    def _calc_heading_score(
        self,
        state: RobotState,
        goal: Tuple[float, float]
    ) -> float:
        """
        计算航向角评分
        
        评估机器人朝向与目标方向的差异
        
        Args:
            state: 机器人状态
            goal: 目标位置
            
        Returns:
            航向角评分 [0, 1]
        """
        dx = goal[0] - state.x
        dy = goal[1] - state.y
        
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 1.0
        
        goal_angle = math.atan2(dy, dx)
        angle_diff = goal_angle - state.theta
        
        # 归一化到 [-pi, pi]
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        
        # 转换为评分 (1 表示完全对准，0 表示完全反向)
        score = (math.pi - abs(angle_diff)) / math.pi
        
        return score
    
    def _calc_obstacle_score(
        self,
        trajectory: Trajectory,
        obstacles: Optional[List[Tuple[float, float]]]
    ) -> float:
        """
        计算障碍物评分
        
        评估轨迹与障碍物的距离
        
        Args:
            trajectory: 轨迹
            obstacles: 障碍物列表
            
        Returns:
            障碍物评分 [0, 1]
        """
        min_dist = float('inf')
        
        # 使用提供的障碍物列表或依赖碰撞检测函数
        if obstacles:
            for state in trajectory.states:
                for obs in obstacles:
                    dist = math.hypot(state.x - obs[0], state.y - obs[1])
                    dist = dist - self.config.robot_radius - self.config.safety_margin
                    if dist < min_dist:
                        min_dist = dist
        
        # 检查轨迹上的点是否碰撞
        for state in trajectory.states:
            if self._check_collision(state.x, state.y):
                return 0.0  # 碰撞，最低评分
        
        if min_dist == float('inf'):
            return 1.0  # 无障碍物，最高评分
        
        # 距离越近评分越低
        if min_dist < 0:
            return 0.0
        
        # 使用指数衰减函数
        score = min(1.0, min_dist / (self.config.robot_radius * 2))
        
        return score
    
    def _calc_velocity_score(self, state: RobotState) -> float:
        """
        计算速度评分
        
        鼓励高速运动
        
        Args:
            state: 机器人状态
            
        Returns:
            速度评分 [0, 1]
        """
        if self.config.max_linear_velocity > 0:
            return abs(state.v) / self.config.max_linear_velocity
        return 0.0
    
    def _calc_distance_score(
        self,
        state: RobotState,
        goal: Tuple[float, float]
    ) -> float:
        """
        计算距离评分
        
        评估与目标的接近程度
        
        Args:
            state: 机器人状态
            goal: 目标位置
            
        Returns:
            距离评分 [0, 1]
        """
        dist = math.hypot(state.x - goal[0], state.y - goal[1])
        
        # 越近越好，使用指数衰减
        score = math.exp(-dist / 5.0)  # 5m 为特征距离
        
        return score
    
    def _check_collision(self, x: float, y: float) -> bool:
        """
        检查位置是否碰撞
        
        Args:
            x: x 坐标
            y: y 坐标
            
        Returns:
            True 如果碰撞
        """
        return self.collision_check(x, y)
    
    def _check_goal_reached(
        self,
        state: RobotState,
        goal: Tuple[float, float]
    ) -> bool:
        """
        检查是否到达目标
        
        Args:
            state: 机器人状态
            goal: 目标位置
            
        Returns:
            True 如果到达目标
        """
        if self.goal_checker:
            return self.goal_checker(state, goal)
        
        dist = math.hypot(state.x - goal[0], state.y - goal[1])
        return dist < self.config.goal_tolerance
    
    def _calculate_min_obstacle_distance(
        self,
        trajectory: Trajectory,
        obstacles: List[Tuple[float, float]]
    ) -> float:
        """
        计算轨迹与障碍物的最小距离
        
        Args:
            trajectory: 轨迹
            obstacles: 障碍物列表
            
        Returns:
            最小距离
        """
        min_dist = float('inf')
        
        for state in trajectory.states:
            for obs in obstacles:
                dist = math.hypot(state.x - obs[0], state.y - obs[1])
                dist = dist - self.config.robot_radius
                if dist < min_dist:
                    min_dist = dist
        
        return min_dist if min_dist != float('inf') else -1.0
    
    def simulate_navigation(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float],
        obstacles: List[Tuple[float, float]],
        max_steps: int = 500,
        dt: float = 0.1,
        debug: bool = False
    ) -> Tuple[List[RobotState], bool]:
        """
        模拟完整的导航过程
        
        Args:
            start: 起点 (x, y, theta)
            goal: 目标位置 (x, y)
            obstacles: 障碍物列表
            max_steps: 最大步数
            dt: 时间步长
            debug: 是否输出调试信息
            
        Returns:
            (轨迹状态列表, 是否成功到达目标)
        """
        state = RobotState(x=start[0], y=start[1], theta=start[2])
        trajectory_states = [state.copy()]
        
        for step in range(max_steps):
            # 检查是否到达目标
            if self._check_goal_reached(state, goal):
                if debug:
                    print(f"Goal reached at step {step}")
                return trajectory_states, True
            
            # 执行 DWA 规划
            result = self.plan(state, goal, obstacles, debug=(debug and step % 50 == 0))
            
            if not result.success:
                # 尝试恢复策略
                result = self.plan_with_recovery(state, goal, obstacles)
                
                if not result.success:
                    if debug:
                        print(f"Planning failed at step {step}")
                    break
            
            # 应用控制指令
            v, omega = result.best_velocity
            
            if self.kinematic_model == KinematicModel.DIFFERENTIAL_DRIVE:
                state = self._update_differential_drive(state, v, omega)
            else:
                state = self._update_ackermann(state, v, omega)
            
            trajectory_states.append(state.copy())
            
            if debug and step % 50 == 0:
                dist_to_goal = math.hypot(state.x - goal[0], state.y - goal[1])
                print(f"Step {step}: pos=({state.x:.2f}, {state.y:.2f}), "
                      f"v={v:.2f}, ω={omega:.2f}, dist_to_goal={dist_to_goal:.2f}")
        
        return trajectory_states, False


# ==================== 工具函数 ====================

def create_circular_obstacle_checker(
    obstacles: List[Tuple[float, float]],
    obstacle_radius: float,
    bounds: Optional[Tuple[float, float, float, float]] = None
) -> Callable[[float, float], bool]:
    """
    创建圆形障碍物的碰撞检测函数
    
    Args:
        obstacles: 障碍物中心列表 [(x, y), ...]
        obstacle_radius: 障碍物半径
        bounds: 地图边界 (min_x, max_x, min_y, max_y)
        
    Returns:
        碰撞检测函数
    """
    def collision_check(x: float, y: float) -> bool:
        # 检查边界
        if bounds:
            min_x, max_x, min_y, max_y = bounds
            if x < min_x or x > max_x or y < min_y or y > max_y:
                return True
        
        # 检查障碍物
        for obs in obstacles:
            if math.hypot(x - obs[0], y - obs[1]) < obstacle_radius:
                return True
        
        return False
    
    return collision_check


def create_grid_obstacle_checker(
    grid: np.ndarray,
    resolution: float,
    origin: Tuple[float, float] = (0.0, 0.0)
) -> Callable[[float, float], bool]:
    """
    创建基于栅格地图的碰撞检测函数
    
    Args:
        grid: 二值栅格地图 (True 表示障碍物)
        resolution: 栅格分辨率
        origin: 地图原点偏移
        
    Returns:
        碰撞检测函数
    """
    def collision_check(x: float, y: float) -> bool:
        # 转换为栅格坐标
        gx = int((x - origin[0]) / resolution)
        gy = int((y - origin[1]) / resolution)
        
        # 检查边界
        if gx < 0 or gx >= grid.shape[1] or gy < 0 or gy >= grid.shape[0]:
            return True
        
        return bool(grid[gy, gx])
    
    return collision_check


# ==================== 示例和测试 ====================

def example_basic():
    """基础使用示例"""
    print("=" * 60)
    print("DWA 路径规划 - 基础示例")
    print("=" * 60)
    
    # 定义障碍物
    obstacles = [
        (3.0, 3.0), (3.5, 3.0), (4.0, 3.0),
        (3.0, 3.5), (3.5, 3.5), (4.0, 3.5),
        (6.0, 6.0), (6.5, 6.0), (7.0, 6.0),
    ]
    
    obstacle_radius = 0.4
    
    # 创建碰撞检测函数
    bounds = (0, 10, 0, 10)
    collision_fn = create_circular_obstacle_checker(obstacles, obstacle_radius, bounds)
    
    # 配置 DWA
    config = DWAConfig(
        max_linear_velocity=1.0,
        max_angular_velocity=2.0,
        max_linear_accel=2.0,
        max_angular_accel=4.0,
        dt=0.1,
        predict_time=3.0,
        velocity_resolution=15,
        angular_resolution=15,
        robot_radius=0.3,
        safety_margin=0.1
    )
    
    # 创建规划器
    planner = DWAPlanner(collision_fn, config)
    
    # 初始状态
    start = RobotState(x=1.0, y=1.0, theta=0.0, v=0.0, omega=0.0)
    goal = (8.0, 8.0)
    
    print(f"起点: ({start.x}, {start.y})")
    print(f"目标: {goal}")
    print(f"障碍物数量: {len(obstacles)}")
    print()
    
    # 执行规划
    result = planner.plan(start, goal, obstacles, debug=True)
    
    if result.success:
        v, omega = result.best_velocity
        print(f"\n[OK] 规划成功!")
        print(f"  最佳控制指令: v={v:.2f} m/s, ω={omega:.2f} rad/s")
        print(f"  轨迹点数: {len(result.best_trajectory)}")
        print(f"  综合评分: {result.best_trajectory.score:.3f}")
        print(f"  航向评分: {result.best_trajectory.heading_score:.3f}")
        print(f"  障碍物评分: {result.best_trajectory.obstacle_score:.3f}")
        print(f"  速度评分: {result.best_trajectory.velocity_score:.3f}")
    else:
        print("\n[FAIL] 规划失败")
    
    return result


def example_full_navigation():
    """完整导航模拟示例"""
    print("\n" + "=" * 60)
    print("DWA 路径规划 - 完整导航模拟")
    print("=" * 60)
    
    # 创建地图
    obstacles = []
    
    # 添加一些随机障碍物
    import random
    random.seed(42)
    for _ in range(20):
        x = random.uniform(2, 18)
        y = random.uniform(2, 18)
        obstacles.append((x, y))
    
    # 添加墙壁
    for i in range(20):
        obstacles.append((i, 0))
        obstacles.append((i, 20))
        obstacles.append((0, i))
        obstacles.append((20, i))
    
    obstacle_radius = 0.5
    bounds = (0, 20, 0, 20)
    collision_fn = create_circular_obstacle_checker(obstacles, obstacle_radius, bounds)
    
    # 配置
    config = DWAConfig(
        max_linear_velocity=1.5,
        max_angular_velocity=2.5,
        robot_radius=0.4,
        predict_time=4.0,
        weight_heading=1.0,
        weight_obstacle=2.5,
        weight_velocity=0.3
    )
    
    planner = DWAPlanner(collision_fn, config)
    
    # 导航参数
    start = (1.0, 1.0, math.pi / 4)  # (x, y, theta)
    goal = (18.0, 18.0)
    
    print(f"起点: {start}")
    print(f"目标: {goal}")
    print(f"障碍物数量: {len(obstacles)}")
    print()
    
    # 模拟导航
    trajectory, success = planner.simulate_navigation(
        start, goal, obstacles, max_steps=300, dt=0.1, debug=True
    )
    
    print(f"\n导航结果: {'成功' if success else '失败'}")
    print(f"总步数: {len(trajectory)}")
    
    if trajectory:
        final = trajectory[-1]
        final_dist = math.hypot(final.x - goal[0], final.y - goal[1])
        print(f"最终位置: ({final.x:.2f}, {final.y:.2f})")
        print(f"最终距离目标: {final_dist:.2f} m")
    
    return trajectory, success


def example_comparison():
    """不同参数对比示例"""
    print("\n" + "=" * 60)
    print("DWA 路径规划 - 参数对比")
    print("=" * 60)
    
    # 简单场景
    obstacles = [(5.0, 5.0), (5.5, 5.0), (5.0, 5.5)]
    collision_fn = create_circular_obstacle_checker(obstacles, 0.5, (0, 10, 0, 10))
    
    start = RobotState(x=1.0, y=1.0, theta=0.0)
    goal = (8.0, 8.0)
    
    configs = [
        ("Conservative", DWAConfig(
            max_linear_velocity=0.5,
            max_angular_velocity=1.5,
            weight_obstacle=3.0,
            predict_time=4.0
        )),
        ("Aggressive", DWAConfig(
            max_linear_velocity=2.0,
            max_angular_velocity=3.0,
            weight_obstacle=1.0,
            predict_time=2.0
        )),
        ("Balanced", DWAConfig(
            max_linear_velocity=1.0,
            max_angular_velocity=2.0,
            weight_obstacle=2.0,
            predict_time=3.0
        )),
    ]
    
    print(f"起点: ({start.x}, {start.y})")
    print(f"目标: {goal}")
    print()
    
    for name, config in configs:
        planner = DWAPlanner(collision_fn, config)
        result = planner.plan(start, goal, obstacles)
        
        if result.success:
            v, omega = result.best_velocity
            print(f"{name:12s}: v={v:.2f}, ω={omega:.2f}, "
                  f"score={result.best_trajectory.score:.3f}")
        else:
            print(f"{name:12s}: Failed")


def example_different_models():
    """不同运动学模型对比"""
    print("\n" + "=" * 60)
    print("DWA 路径规划 - 运动学模型对比")
    print("=" * 60)
    
    obstacles = [(4.0, 4.0), (6.0, 6.0)]
    collision_fn = create_circular_obstacle_checker(obstacles, 0.5, (0, 10, 0, 10))
    
    start = RobotState(x=1.0, y=1.0, theta=math.pi/4)
    goal = (8.0, 8.0)
    config = DWAConfig(max_linear_velocity=1.0, max_angular_velocity=2.0)
    
    models = [
        ("Differential Drive", KinematicModel.DIFFERENTIAL_DRIVE),
        ("Ackermann Steering", KinematicModel.ACKERMANN_STEERING),
    ]
    
    print(f"起点: ({start.x}, {start.y}, {start.theta:.2f})")
    print(f"目标: {goal}")
    print()
    
    for name, model in models:
        planner = DWAPlanner(collision_fn, config, kinematic_model=model)
        result = planner.plan(start, goal, obstacles)
        
        if result.success:
            v, omega = result.best_velocity
            print(f"{name:20s}: v={v:.2f}, ω={omega:.2f}")
        else:
            print(f"{name:20s}: Failed")


def example_visualization_ascii():
    """ASCII 可视化示例"""
    print("\n" + "=" * 60)
    print("DWA 路径规划 - ASCII 可视化")
    print("=" * 60)
    
    # 创建场景
    obstacles = [(3, 3), (4, 3), (5, 3), (6, 4), (6, 5), (6, 6)]
    collision_fn = create_circular_obstacle_checker(obstacles, 0.4, (0, 10, 0, 10))
    
    config = DWAConfig(
        max_linear_velocity=1.0,
        robot_radius=0.3,
        predict_time=3.0
    )
    
    planner = DWAPlanner(collision_fn, config)
    
    start = (0.5, 0.5, math.pi / 4)
    goal = (8.0, 8.0)
    
    # 模拟导航
    trajectory, success = planner.simulate_navigation(
        start, goal, obstacles, max_steps=100, dt=0.1
    )
    
    # ASCII 可视化
    grid_size = 20
    scale = 10.0 / grid_size
    
    grid = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
    
    # 绘制障碍物
    for ox, oy in obstacles:
        gx = int(ox / scale)
        gy = int(oy / scale)
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            grid[grid_size - 1 - gy][gx] = '#'
    
    # 绘制轨迹
    for state in trajectory:
        gx = int(state.x / scale)
        gy = int(state.y / scale)
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            if grid[grid_size - 1 - gy][gx] == '.':
                grid[grid_size - 1 - gy][gx] = '*'
    
    # 绘制起点和终点
    sx, sy = int(start[0] / scale), int(start[1] / scale)
    gx, gy = int(goal[0] / scale), int(goal[1] / scale)
    grid[grid_size - 1 - sy][sx] = 'S'
    grid[grid_size - 1 - gy][gx] = 'G'
    
    print(f"起点: ({start[0]}, {start[1]}), 目标: {goal}")
    print(f"结果: {'成功' if success else '失败'}, 步数: {len(trajectory)}")
    print()
    print("图例: S=起点, G=目标, #=障碍物, *=路径")
    print("-" * (grid_size + 2))
    for row in grid:
        print("|" + "".join(row) + "|")
    print("-" * (grid_size + 2))


def example_recovery():
    """恢复策略示例"""
    print("\n" + "=" * 60)
    print("DWA 路径规划 - 恢复策略测试")
    print("=" * 60)
    
    # 创建一个困难场景：机器人被困
    obstacles = []
    for i in range(10):
        obstacles.append((i, 3))
        obstacles.append((i, -3))
    for i in range(-3, 4):
        obstacles.append((9, i))
    
    collision_fn = create_circular_obstacle_checker(obstacles, 0.5, (-1, 10, -5, 5))
    
    config = DWAConfig(
        max_linear_velocity=2.0,
        robot_radius=0.4,
        predict_time=2.0,
        weight_obstacle=5.0  # 非常保守
    )
    
    planner = DWAPlanner(collision_fn, config)
    
    # 机器人位置靠近出口
    start = RobotState(x=7.0, y=0.0, theta=0.0)
    goal = (0.0, 0.0)
    
    print(f"起点: ({start.x}, {start.y})")
    print(f"目标: {goal}")
    print("场景: 机器人位于障碍物包围的狭窄通道中")
    print()
    
    # 普通规划
    result1 = planner.plan(start, goal, obstacles)
    print(f"普通规划: {'成功' if result1.success else '失败'}")
    
    # 带恢复策略的规划
    result2 = planner.plan_with_recovery(start, goal, obstacles, max_attempts=3)
    print(f"恢复策略: {'成功' if result2.success else '失败'}")
    
    if result2.success:
        v, omega = result2.best_velocity
        print(f"  最佳速度: v={v:.2f}, ω={omega:.2f}")


if __name__ == "__main__":
    # 运行所有示例
    example_basic()
    example_full_navigation()
    example_comparison()
    example_different_models()
    example_visualization_ascii()
    example_recovery()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
