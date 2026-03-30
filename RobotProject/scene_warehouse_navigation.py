# -*- coding: utf-8 -*-
"""
Scene 1: Warehouse Navigation

仓库导航场景
- 货架障碍物
- 多个拣货点
- 路径优化
"""

import numpy as np
import math
from typing import List, Tuple
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from robot import (
    ChassisType, ChassisConfig, DifferentialDrive, AckermannSteering,
    OccupancyGrid, CircularObstacle, RectangularObstacle,
    AStarPlanner, RRTStarPlanner, Path, PurePursuitController
)


def create_warehouse_map(width: float = 20.0, height: float = 15.0) -> OccupancyGrid:
    """
    创建仓库地图
    
    仓库布局：
    - 墙壁围绕
    - 货架排列
    - 通道
    """
    resolution = 0.2
    origin = (-width/2, -height/2)
    
    grid = OccupancyGrid(width, height, resolution, origin)
    
    # 墙壁（四周）
    wall_thickness = 0.5
    walls = [
        # 上墙
        RectangularObstacle(id=0, position=(0, height/2 - wall_thickness/2), 
                          width=width, height=wall_thickness),
        # 下墙
        RectangularObstacle(id=1, position=(0, -height/2 + wall_thickness/2), 
                          width=width, height=wall_thickness),
        # 左墙
        RectangularObstacle(id=2, position=(-width/2 + wall_thickness/2, 0), 
                          width=wall_thickness, height=height),
        # 右墙
        RectangularObstacle(id=3, position=(width/2 - wall_thickness/2, 0), 
                          width=wall_thickness, height=height),
    ]
    
    for wall in walls:
        grid.set_obstacle(wall)
    
    # 货架（多排）
    shelf_width = 1.2
    shelf_depth = 0.6
    aisle_width = 1.5
    
    shelf_id = 4
    num_rows = 3
    num_cols = 4
    
    start_x = -width/2 + 2
    start_y = -height/2 + 2
    
    for row in range(num_rows):
        for col in range(num_cols):
            x = start_x + col * (shelf_width + aisle_width)
            y = start_y + row * (shelf_depth + aisle_width * 2)
            
            shelf = RectangularObstacle(
                id=shelf_id,
                position=(x, y),
                width=shelf_width,
                height=shelf_depth
            )
            grid.set_obstacle(shelf)
            shelf_id += 1
    
    return grid


def get_pick_points() -> List[Tuple[float, float]]:
    """获取拣货点列表"""
    return [
        (-6, -4),   # 拣货点 1
        (0, -4),    # 拣货点 2
        (6, -4),    # 拣货点 3
        (-6, 0),    # 拣货点 4
        (6, 0),     # 拣货点 5
        (-6, 4),    # 拣货点 6
        (0, 4),     # 拣货点 7
        (6, 4),     # 拣货点 8
    ]


def plan_warehouse_route(grid: OccupancyGrid, 
                        pick_points: List[Tuple[float, float]],
                        start_point: Tuple[float, float] = (-8, -6)) -> List[Path]:
    """
    规划仓库路线
    
    使用最近邻算法优化拣货顺序
    """
    collision_fn = lambda x, y: not grid.is_in_bounds_world(x, y) or grid.check_collision(x, y)
    planner = AStarPlanner(collision_fn, resolution=0.2)
    
    # 最近邻算法优化拣货顺序
    unvisited = pick_points.copy()
    current_pos = start_point
    route_order = []
    
    while unvisited:
        # 找到最近的未访问点
        nearest = min(unvisited, key=lambda p: math.sqrt((p[0]-current_pos[0])**2 + 
                                                         (p[1]-current_pos[1])**2))
        route_order.append(nearest)
        unvisited.remove(nearest)
        current_pos = nearest
    
    # 规划路径
    paths = []
    current_pos = start_point
    
    for target in route_order:
        path = planner.plan(current_pos, target)
        if path.is_valid:
            paths.append(path)
            current_pos = target
        else:
            print(f"警告: 无法规划到 {target} 的路径")
    
    return paths


def simulate_chassis_in_warehouse(grid: OccupancyGrid, 
                                   paths: List[Path],
                                   chassis_type: ChassisType = ChassisType.DIFFERENTIAL_DRIVE):
    """
    在仓库场景中仿真底盘
    
    Args:
        grid: 栅格地图
        paths: 路径列表
        chassis_type: 底盘类型
    """
    # 创建底盘
    config = ChassisConfig(
        body_length=0.6,
        body_width=0.4,
        wheel_radius=0.08,
        max_linear_velocity=0.8,
        max_angular_velocity=1.5
    )
    
    if chassis_type == ChassisType.DIFFERENTIAL_DRIVE:
        chassis = DifferentialDrive(config)
    elif chassis_type == ChassisType.ACKERMANN_STEERING:
        chassis = AckermannSteering(config)
    else:
        chassis = DifferentialDrive(config)
    
    # 设置初始位置
    if paths and paths[0].waypoints:
        chassis.state.x = paths[0].waypoints[0][0]
        chassis.state.y = paths[0].waypoints[0][1]
    
    # 跟踪所有路径
    controller = PurePursuitController(
        lookahead_distance=0.4,
        wheelbase=config.wheelbase,
        max_linear_velocity=config.max_linear_velocity
    )
    
    all_trajectories = []
    dt = 0.1
    
    for path in paths:
        if not path.is_valid:
            continue
        
        trajectory = []
        current_idx = 0
        max_steps = 500
        
        for _ in range(max_steps):
            current_pose = (chassis.state.x, chassis.state.y, chassis.state.theta)
            trajectory.append(current_pose)
            
            # 检查是否到达终点
            if current_idx >= len(path) - 1:
                break
            
            # 计算控制命令
            v, omega, current_idx = controller.compute_control(
                current_pose, path, current_idx
            )
            
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
            else:
                control = chassis.set_velocity(v, omega)
            
            chassis.apply_control(control, dt)
        
        all_trajectories.append(trajectory)
    
    return all_trajectories, chassis


def run_warehouse_simulation():
    """运行仓库仿真"""
    print("=" * 60)
    print("仓库导航场景")
    print("=" * 60)
    
    # 创建地图
    print("\n创建仓库地图...")
    grid = create_warehouse_map()
    print(f"地图大小: {grid.width}m x {grid.height}m")
    print(f"栅格数量: {grid.grid_width} x {grid.grid_height}")
    print(f"障碍物数量: {len(grid.obstacles)}")
    
    # 获取拣货点
    pick_points = get_pick_points()
    print(f"\n拣货点数量: {len(pick_points)}")
    
    # 规划路线
    print("\n规划路线...")
    paths = plan_warehouse_route(grid, pick_points[:4])  # 只取前4个点
    print(f"成功规划路径数: {len(paths)}")
    
    # 仿真不同底盘
    chassis_types = [
        ChassisType.DIFFERENTIAL_DRIVE,
        ChassisType.ACKERMANN_STEERING,
    ]
    
    results = {}
    
    for chassis_type in chassis_types:
        print(f"\n仿真 {chassis_type.value}...")
        trajectories, chassis = simulate_chassis_in_warehouse(grid, paths, chassis_type)
        
        # 计算性能指标
        total_distance = 0
        for traj in trajectories:
            for i in range(len(traj) - 1):
                p1 = np.array(traj[i][:2])
                p2 = np.array(traj[i + 1][:2])
                total_distance += np.linalg.norm(p2 - p1)
        
        results[chassis_type.value] = {
            'total_distance': total_distance,
            'trajectories': trajectories,
            'final_position': (chassis.state.x, chassis.state.y)
        }
        
        print(f"  总行驶距离: {total_distance:.2f}m")
        print(f"  最终位置: ({chassis.state.x:.2f}, {chassis.state.y:.2f})")
    
    # 对比结果
    print("\n" + "=" * 60)
    print("性能对比")
    print("=" * 60)
    
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  总行驶距离: {data['total_distance']:.2f}m")
    
    return grid, paths, results


if __name__ == "__main__":
    run_warehouse_simulation()
