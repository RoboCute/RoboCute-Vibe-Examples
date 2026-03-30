# -*- coding: utf-8 -*-
"""
Scene 2: Off-road Terrain

越野地形场景
- 不平整地面
- 斜坡和台阶
- 底盘性能对比
"""

import numpy as np
import math
from typing import List, Tuple, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from robot import (
    ChassisType, ChassisConfig,
    DifferentialDrive, AckermannSteering, TrackedVehicle, MecanumWheel,
    OccupancyGrid, CircularObstacle, RectangularObstacle,
    AStarPlanner, PurePursuitController, Path
)


class TerrainMap:
    """地形地图"""
    
    def __init__(self, width: float = 15.0, height: float = 10.0):
        self.width = width
        self.height = height
        self.resolution = 0.2
        self.origin = (-width/2, -height/2)
        
        # 高程图
        self.elevation = np.zeros((int(height/self.resolution), 
                                   int(width/self.resolution)))
        
        # 坡度图
        self.slope = np.zeros_like(self.elevation)
        
        self._generate_terrain()
    
    def _generate_terrain(self):
        """生成地形"""
        rows, cols = self.elevation.shape
        
        for i in range(rows):
            for j in range(cols):
                x = self.origin[0] + j * self.resolution
                y = self.origin[1] + i * self.resolution
                
                # 组合多种地形特征
                h = 0.0
                
                # 1. 中央丘陵
                dist_to_center = math.sqrt(x**2 + y**2)
                h += 0.5 * math.exp(-dist_to_center**2 / 8)
                
                # 2. 斜坡区域（右侧）
                if x > 2:
                    h += 0.3 * (x - 2) / (self.width/2 - 2)
                
                # 3. 台阶（左侧）
                if x < -3:
                    h += 0.2 * int((-x - 3) / 1.5)
                
                # 4. 小起伏
                h += 0.05 * math.sin(x * 2) * math.cos(y * 2)
                
                self.elevation[i, j] = h
        
        # 计算坡度
        self._compute_slope()
    
    def _compute_slope(self):
        """计算坡度"""
        rows, cols = self.elevation.shape
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # x方向梯度
                dx = (self.elevation[i, j+1] - self.elevation[i, j-1]) / (2 * self.resolution)
                # y方向梯度
                dy = (self.elevation[i+1, j] - self.elevation[i-1, j]) / (2 * self.resolution)
                
                # 坡度角
                slope_rad = math.atan(math.sqrt(dx**2 + dy**2))
                self.slope[i, j] = slope_rad
    
    def get_elevation(self, x: float, y: float) -> float:
        """获取位置高程"""
        j = int((x - self.origin[0]) / self.resolution)
        i = int((y - self.origin[1]) / self.resolution)
        
        if 0 <= i < self.elevation.shape[0] and 0 <= j < self.elevation.shape[1]:
            return self.elevation[i, j]
        return 0.0
    
    def get_slope(self, x: float, y: float) -> float:
        """获取位置坡度"""
        j = int((x - self.origin[0]) / self.resolution)
        i = int((y - self.origin[1]) / self.resolution)
        
        if 0 <= i < self.slope.shape[0] and 0 <= j < self.slope.shape[1]:
            return self.slope[i, j]
        return 0.0
    
    def create_obstacle_grid(self, max_slope: float = 0.3) -> OccupancyGrid:
        """
        根据坡度创建障碍物栅格
        
        Args:
            max_slope: 最大可通行坡度（弧度）
        """
        grid = OccupancyGrid(self.width, self.height, self.resolution, self.origin)
        
        rows, cols = self.slope.shape
        for i in range(rows):
            for j in range(cols):
                if self.slope[i, j] > max_slope:
                    # 创建圆形障碍物
                    x = self.origin[0] + j * self.resolution
                    y = self.origin[1] + i * self.resolution
                    obstacle = CircularObstacle(
                        id=i * cols + j,
                        position=(x, y),
                        radius=self.resolution / 2
                    )
                    grid.set_obstacle(obstacle)
        
        return grid


def create_offroad_obstacles(grid: OccupancyGrid):
    """添加越野专用障碍物"""
    # 石头
    stones = [
        CircularObstacle(id=1000, position=(-4, 2), radius=0.4),
        CircularObstacle(id=1001, position=(-2, 3), radius=0.3),
        CircularObstacle(id=1002, position=(3, -2), radius=0.5),
        CircularObstacle(id=1003, position=(5, 1), radius=0.35),
    ]
    
    for stone in stones:
        grid.set_obstacle(stone)
    
    # 树干
    trees = [
        RectangularObstacle(id=2000, position=(-5, -3), width=0.3, height=0.3),
        RectangularObstacle(id=2001, position=(4, 3), width=0.3, height=0.3),
    ]
    
    for tree in trees:
        grid.set_obstacle(tree)


def calculate_terrain_penalty(chassis: ChassisBase, terrain: TerrainMap, 
                               x: float, y: float) -> float:
    """
    计算地形惩罚
    
    不同底盘对地形的适应性不同
    
    Returns:
        速度惩罚因子 (0-1)
    """
    slope = terrain.get_slope(x, y)
    elevation = terrain.get_elevation(x, y)
    
    chassis_type = chassis.chassis_type
    
    if chassis_type == ChassisType.TRACKED_VEHICLE:
        # 履带底盘地形适应性最好
        if slope < 0.5:
            return 1.0
        elif slope < 0.8:
            return 0.8
        else:
            return 0.5
    
    elif chassis_type == ChassisType.MECANUM_WHEEL:
        # 麦克纳姆轮对坡度和不平整地面敏感
        if slope < 0.15:
            return 1.0
        elif slope < 0.3:
            return 0.7
        else:
            return 0.3
    
    elif chassis_type == ChassisType.ACKERMANN_STEERING:
        # 阿克曼底盘对坡度有一定适应性
        if slope < 0.25:
            return 1.0
        elif slope < 0.45:
            return 0.75
        else:
            return 0.4
    
    else:  # DIFFERENTIAL_DRIVE
        # 差速底盘中等适应性
        if slope < 0.2:
            return 1.0
        elif slope < 0.4:
            return 0.8
        else:
            return 0.5


def simulate_offroad_navigation(chassis_type: ChassisType, 
                                 terrain: TerrainMap,
                                 grid: OccupancyGrid,
                                 start: Tuple[float, float] = (-6, 0),
                                 goal: Tuple[float, float] = (6, 0)) -> Dict:
    """
    仿真越野导航
    
    Args:
        chassis_type: 底盘类型
        terrain: 地形
        grid: 栅格地图
        start: 起点
        goal: 终点
        
    Returns:
        仿真结果字典
    """
    # 创建底盘
    config = ChassisConfig(
        body_length=0.7,
        body_width=0.5,
        wheel_radius=0.12,
        max_linear_velocity=1.0,
        max_angular_velocity=2.0,
        wheel_friction=0.9  # 越野需要更好的摩擦
    )
    
    if chassis_type == ChassisType.DIFFERENTIAL_DRIVE:
        chassis = DifferentialDrive(config)
    elif chassis_type == ChassisType.ACKERMANN_STEERING:
        chassis = AckermannSteering(config)
    elif chassis_type == ChassisType.TRACKED_VEHICLE:
        chassis = TrackedVehicle(config)
    elif chassis_type == ChassisType.MECANUM_WHEEL:
        chassis = MecanumWheel(config)
    else:
        chassis = DifferentialDrive(config)
    
    chassis.state.x = start[0]
    chassis.state.y = start[1]
    
    # 规划路径
    collision_fn = lambda x, y: not grid.is_in_bounds_world(x, y) or grid.check_collision(x, y)
    planner = AStarPlanner(collision_fn, resolution=0.2)
    path = planner.plan(start, goal)
    
    if not path.is_valid:
        return {
            'chassis_type': chassis_type.value,
            'success': False,
            'reason': '路径规划失败'
        }
    
    # 跟踪路径
    controller = PurePursuitController(
        lookahead_distance=0.5,
        wheelbase=config.wheelbase,
        max_linear_velocity=config.max_linear_velocity
    )
    
    trajectory = []
    elevations = []
    slopes = []
    penalties = []
    
    current_idx = 0
    dt = 0.1
    max_steps = 800
    
    for step in range(max_steps):
        current_pose = (chassis.state.x, chassis.state.y, chassis.state.theta)
        trajectory.append(current_pose)
        
        # 记录地形信息
        elev = terrain.get_elevation(chassis.state.x, chassis.state.y)
        slope = terrain.get_slope(chassis.state.x, chassis.state.y)
        elevations.append(elev)
        slopes.append(slope)
        
        # 计算地形惩罚
        penalty = calculate_terrain_penalty(chassis, terrain, 
                                            chassis.state.x, chassis.state.y)
        penalties.append(penalty)
        
        # 检查是否到达终点
        if current_idx >= len(path) - 1:
            dist_to_goal = math.sqrt((chassis.state.x - goal[0])**2 + 
                                     (chassis.state.y - goal[1])**2)
            if dist_to_goal < 0.3:
                break
        
        # 计算控制命令
        v, omega, current_idx = controller.compute_control(
            current_pose, path, current_idx
        )
        
        # 应用地形惩罚
        v *= penalty
        
        # 根据底盘类型应用控制
        if chassis_type == ChassisType.DIFFERENTIAL_DRIVE:
            control = chassis.set_velocity(v, omega)
        elif chassis_type == ChassisType.ACKERMANN_STEERING:
            if abs(v) > 0.01:
                R = v / (omega + 1e-6)
                steering_angle = math.atan(config.wheelbase / R)
            else:
                steering_angle = 0.0
            control = np.array([v, steering_angle])
        elif chassis_type == ChassisType.TRACKED_VEHICLE:
            W = config.track_width
            v_left = v - omega * W / 2
            v_right = v + omega * W / 2
            control = np.array([v_left, v_right])
        elif chassis_type == ChassisType.MECANUM_WHEEL:
            control = np.array([v, 0, omega])
        else:
            control = chassis.set_velocity(v, omega)
        
        chassis.apply_control(control, dt)
    
    # 计算性能指标
    total_distance = 0.0
    for i in range(len(trajectory) - 1):
        p1 = np.array(trajectory[i][:2])
        p2 = np.array(trajectory[i + 1][:2])
        total_distance += np.linalg.norm(p2 - p1)
    
    avg_elevation = np.mean(elevations)
    max_slope_encountered = max(slopes) if slopes else 0
    avg_penalty = np.mean(penalties) if penalties else 1.0
    
    final_dist_to_goal = math.sqrt((chassis.state.x - goal[0])**2 + 
                                   (chassis.state.y - goal[1])**2)
    
    return {
        'chassis_type': chassis_type.value,
        'success': final_dist_to_goal < 0.5,
        'trajectory': trajectory,
        'total_distance': total_distance,
        'steps': len(trajectory),
        'avg_elevation': avg_elevation,
        'max_slope': max_slope_encountered,
        'avg_speed_factor': avg_penalty,
        'final_position': (chassis.state.x, chassis.state.y),
        'final_distance_to_goal': final_dist_to_goal
    }


def run_offroad_simulation():
    """运行越野地形仿真"""
    print("=" * 60)
    print("越野地形场景")
    print("=" * 60)
    
    # 创建地形
    print("\n生成地形...")
    terrain = TerrainMap()
    print(f"地形大小: {terrain.width}m x {terrain.height}m")
    print(f"最大高程: {terrain.elevation.max():.2f}m")
    print(f"最大坡度: {np.degrees(terrain.slope.max()):.1f}°")
    
    # 创建栅格地图
    grid = terrain.create_obstacle_grid(max_slope=0.35)
    create_offroad_obstacles(grid)
    print(f"障碍物数量: {len(grid.obstacles)}")
    
    # 仿真所有底盘类型
    chassis_types = [
        ChassisType.DIFFERENTIAL_DRIVE,
        ChassisType.ACKERMANN_STEERING,
        ChassisType.TRACKED_VEHICLE,
        ChassisType.MECANUM_WHEEL,
    ]
    
    results = []
    
    for chassis_type in chassis_types:
        print(f"\n仿真 {chassis_type.value}...")
        result = simulate_offroad_navigation(chassis_type, terrain, grid)
        results.append(result)
        
        print(f"  成功: {result['success']}")
        print(f"  总距离: {result.get('total_distance', 0):.2f}m")
        print(f"  平均速度因子: {result.get('avg_speed_factor', 0):.2f}")
        print(f"  遇到的最大坡度: {np.degrees(result.get('max_slope', 0)):.1f}°")
    
    # 对比结果
    print("\n" + "=" * 60)
    print("越野性能对比")
    print("=" * 60)
    print(f"{'底盘类型':<25} {'成功率':>10} {'距离(m)':>10} {'速度因子':>10}")
    print("-" * 60)
    
    for result in results:
        success = "✓" if result['success'] else "✗"
        dist = result.get('total_distance', 0)
        penalty = result.get('avg_speed_factor', 0)
        print(f"{result['chassis_type']:<25} {success:>10} {dist:>10.2f} {penalty:>10.2f}")
    
    return terrain, grid, results


if __name__ == "__main__":
    run_offroad_simulation()
