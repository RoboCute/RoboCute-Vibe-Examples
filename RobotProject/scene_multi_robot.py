# -*- coding: utf-8 -*-
"""
Scene 3: Multi-robot Collaboration

多机器人协同场景
- 多个底盘同时仿真
- 避碰算法
- 任务分配
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Set
import sys
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from robot import (
    ChassisType, ChassisConfig,
    DifferentialDrive, AckermannSteering, MecanumWheel,
    OccupancyGrid, CircularObstacle, DynamicObstacle,
    AStarPlanner, Path, PurePursuitController
)


@dataclass
class Robot:
    """机器人数据类"""
    id: int
    chassis: any
    task: Tuple[float, float] = None
    path: Path = None
    path_index: int = 0
    status: str = "idle"  # idle, moving, waiting, completed
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    priority: int = 0


class MultiRobotCoordinator:
    """多机器人协调器"""
    
    def __init__(self, robots: List[Robot], grid: OccupancyGrid,
                 safety_radius: float = 0.5):
        """
        初始化协调器
        
        Args:
            robots: 机器人列表
            grid: 栅格地图
            safety_radius: 安全半径
        """
        self.robots = robots
        self.grid = grid
        self.safety_radius = safety_radius
        self.collision_pairs: Set[Tuple[int, int]] = set()
        self.time = 0.0
        
    def check_collision(self, robot1: Robot, robot2: Robot) -> bool:
        """检查两个机器人是否会发生碰撞"""
        pos1 = np.array([robot1.chassis.state.x, robot1.chassis.state.y])
        pos2 = np.array([robot2.chassis.state.x, robot2.chassis.state.y])
        
        distance = np.linalg.norm(pos2 - pos1)
        return distance < self.safety_radius * 2
    
    def predict_collision(self, robot1: Robot, robot2: Robot, 
                          prediction_time: float = 2.0) -> bool:
        """预测未来是否会发生碰撞"""
        # 简化的预测：基于当前速度和方向
        v1 = np.array([robot1.chassis.state.vx, robot1.chassis.state.vy])
        v2 = np.array([robot2.chassis.state.vx, robot2.chassis.state.vy])
        
        pos1 = np.array([robot1.chassis.state.x, robot1.chassis.state.y])
        pos2 = np.array([robot2.chassis.state.x, robot2.chassis.state.y])
        
        # 预测位置
        future_pos1 = pos1 + v1 * prediction_time
        future_pos2 = pos2 + v2 * prediction_time
        
        future_dist = np.linalg.norm(future_pos2 - future_pos1)
        return future_dist < self.safety_radius * 2
    
    def resolve_conflict(self, robot1: Robot, robot2: Robot):
        """解决冲突（优先级低的机器人等待）"""
        if robot1.priority >= robot2.priority:
            higher, lower = robot1, robot2
        else:
            higher, lower = robot2, robot1
        
        # 低优先级机器人等待
        if lower.status == "moving":
            lower.status = "waiting"
    
    def update_priorities(self):
        """更新优先级（基于任务紧急度和等待时间）"""
        for robot in self.robots:
            if robot.status == "waiting":
                # 等待时间越长，优先级越高
                robot.priority += 1
    
    def step(self, dt: float):
        """执行一步协调"""
        self.time += dt
        
        # 更新优先级
        self.update_priorities()
        
        # 检测碰撞
        self.collision_pairs.clear()
        for i, robot1 in enumerate(self.robots):
            for robot2 in self.robots[i+1:]:
                if self.check_collision(robot1, robot2):
                    self.collision_pairs.add((robot1.id, robot2.id))
                    self.resolve_conflict(robot1, robot2)
                elif self.predict_collision(robot1, robot2):
                    # 预测到潜在碰撞，提前处理
                    pass
        
        # 更新等待中的机器人
        for robot in self.robots:
            if robot.status == "waiting":
                # 检查是否可以恢复移动
                can_resume = True
                for other in self.robots:
                    if other.id != robot.id:
                        dist = np.linalg.norm([
                            other.chassis.state.x - robot.chassis.state.x,
                            other.chassis.state.y - robot.chassis.state.y
                        ])
                        if dist < self.safety_radius * 2.5:
                            can_resume = False
                            break
                
                if can_resume:
                    robot.status = "moving"


def create_multi_robot_map(width: float = 15.0, height: float = 10.0) -> OccupancyGrid:
    """创建多机器人场景地图"""
    resolution = 0.2
    origin = (-width/2, -height/2)
    
    grid = OccupancyGrid(width, height, resolution, origin)
    
    # 中央障碍物（分隔区域）
    central_obstacles = [
        RectangularObstacle(id=0, position=(0, -3), width=2, height=1),
        RectangularObstacle(id=1, position=(0, 0), width=2, height=1),
        RectangularObstacle(id=2, position=(0, 3), width=2, height=1),
    ]
    
    # 随机障碍物
    np.random.seed(42)
    for i in range(10):
        x = np.random.uniform(-width/2 + 1, width/2 - 1)
        y = np.random.uniform(-height/2 + 1, height/2 - 1)
        
        # 避开中央通道
        if abs(x) < 2:
            continue
        
        obstacle = CircularObstacle(
            id=100 + i,
            position=(x, y),
            radius=np.random.uniform(0.3, 0.6)
        )
        grid.set_obstacle(obstacle)
    
    for obs in central_obstacles:
        grid.set_obstacle(obs)
    
    return grid


def assign_tasks(robots: List[Robot], tasks: List[Tuple[float, float]], 
                 grid: OccupancyGrid) -> Dict[int, Tuple[float, float]]:
    """
    任务分配（最近邻算法）
    
    Args:
        robots: 机器人列表
        tasks: 任务位置列表
        grid: 栅格地图
        
    Returns:
        任务分配字典 {robot_id: task}
    """
    collision_fn = lambda x, y: not grid.is_in_bounds_world(x, y) or grid.check_collision(x, y)
    planner = AStarPlanner(collision_fn, resolution=0.2)
    
    assignments = {}
    unassigned_robots = set(r.id for r in robots)
    unassigned_tasks = tasks.copy()
    
    while unassigned_robots and unassigned_tasks:
        best_cost = float('inf')
        best_pair = None
        
        for robot_id in unassigned_robots:
            robot = next(r for r in robots if r.id == robot_id)
            start = (robot.chassis.state.x, robot.chassis.state.y)
            
            for task in unassigned_tasks:
                # 使用欧几里得距离作为近似
                cost = math.sqrt((task[0] - start[0])**2 + (task[1] - start[1])**2)
                
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (robot_id, task)
        
        if best_pair:
            robot_id, task = best_pair
            assignments[robot_id] = task
            unassigned_robots.remove(robot_id)
            unassigned_tasks.remove(task)
    
    return assignments


def create_robot_fleet(num_robots: int, start_positions: List[Tuple[float, float]],
                       chassis_types: List[ChassisType] = None) -> List[Robot]:
    """创建机器人车队"""
    if chassis_types is None:
        chassis_types = [ChassisType.DIFFERENTIAL_DRIVE] * num_robots
    
    robots = []
    
    for i in range(num_robots):
        config = ChassisConfig(
            body_length=0.5,
            body_width=0.35,
            wheel_radius=0.08,
            max_linear_velocity=0.6,
            max_angular_velocity=1.5
        )
        
        chassis_type = chassis_types[i % len(chassis_types)]
        
        if chassis_type == ChassisType.DIFFERENTIAL_DRIVE:
            chassis = DifferentialDrive(config)
        elif chassis_type == ChassisType.ACKERMANN_STEERING:
            chassis = AckermannSteering(config)
        elif chassis_type == ChassisType.MECANUM_WHEEL:
            chassis = MecanumWheel(config)
        else:
            chassis = DifferentialDrive(config)
        
        # 设置初始位置
        if i < len(start_positions):
            chassis.state.x = start_positions[i][0]
            chassis.state.y = start_positions[i][1]
        
        robot = Robot(
            id=i,
            chassis=chassis,
            priority=num_robots - i  # 编号越小，优先级越高
        )
        robots.append(robot)
    
    return robots


def simulate_multi_robot(robots: List[Robot], tasks: List[Tuple[float, float]], 
                        grid: OccupancyGrid, max_time: float = 60.0) -> Dict:
    """
    多机器人仿真
    
    Args:
        robots: 机器人列表
        tasks: 任务列表
        grid: 栅格地图
        max_time: 最大仿真时间
        
    Returns:
        仿真结果
    """
    # 任务分配
    print("分配任务...")
    assignments = assign_tasks(robots, tasks, grid)
    
    for robot in robots:
        if robot.id in assignments:
            robot.task = assignments[robot.id]
            robot.status = "moving"
    
    # 创建协调器
    coordinator = MultiRobotCoordinator(robots, grid, safety_radius=0.4)
    
    # 规划路径
    print("规划路径...")
    collision_fn = lambda x, y: not grid.is_in_bounds_world(x, y) or grid.check_collision(x, y)
    planner = AStarPlanner(collision_fn, resolution=0.2)
    
    for robot in robots:
        if robot.task:
            start = (robot.chassis.state.x, robot.chassis.state.y)
            path = planner.plan(start, robot.task)
            robot.path = path
            robot.path_index = 0
    
    # 创建控制器
    controllers = {
        robot.id: PurePursuitController(
            lookahead_distance=0.4,
            wheelbase=robot.chassis.config.wheelbase,
            max_linear_velocity=robot.chassis.config.max_linear_velocity
        )
        for robot in robots
    }
    
    # 仿真
    print("运行仿真...")
    dt = 0.1
    num_steps = int(max_time / dt)
    completed_tasks = 0
    
    for step in range(num_steps):
        # 协调器步进
        coordinator.step(dt)
        
        all_completed = True
        
        for robot in robots:
            if robot.status == "completed":
                continue
            
            all_completed = False
            
            # 记录轨迹
            robot.trajectory.append((
                robot.chassis.state.x,
                robot.chassis.state.y,
                robot.chassis.state.theta
            ))
            
            if robot.status != "moving" or not robot.path or not robot.path.is_valid:
                continue
            
            # 检查是否到达目标
            if robot.path_index >= len(robot.path) - 1:
                dist_to_goal = math.sqrt(
                    (robot.chassis.state.x - robot.task[0])**2 +
                    (robot.chassis.state.y - robot.task[1])**2
                )
                if dist_to_goal < 0.3:
                    robot.status = "completed"
                    completed_tasks += 1
                    continue
            
            # 计算控制命令
            current_pose = (robot.chassis.state.x, robot.chassis.state.y, 
                          robot.chassis.state.theta)
            controller = controllers[robot.id]
            
            v, omega, robot.path_index = controller.compute_control(
                current_pose, robot.path, robot.path_index
            )
            
            # 根据底盘类型应用控制
            chassis_type = robot.chassis.chassis_type
            if chassis_type == ChassisType.DIFFERENTIAL_DRIVE:
                control = robot.chassis.set_velocity(v, omega)
            elif chassis_type == ChassisType.ACKERMANN_STEERING:
                if abs(v) > 0.01:
                    R = v / (omega + 1e-6)
                    steering_angle = math.atan(robot.chassis.config.wheelbase / R)
                else:
                    steering_angle = 0.0
                control = np.array([v, steering_angle])
            elif chassis_type == ChassisType.MECANUM_WHEEL:
                control = np.array([v, 0, omega])
            else:
                control = robot.chassis.set_velocity(v, omega)
            
            robot.chassis.apply_control(control, dt)
        
        if all_completed:
            print(f"所有任务完成！用时: {step * dt:.1f}s")
            break
    
    # 计算统计信息
    total_distance = sum(
        sum(math.sqrt(
            (robot.trajectory[i+1][0] - robot.trajectory[i][0])**2 +
            (robot.trajectory[i+1][1] - robot.trajectory[i][1])**2
        ) for i in range(len(robot.trajectory) - 1))
        for robot in robots
    )
    
    collision_count = len(coordinator.collision_pairs)
    
    return {
        'completed_tasks': completed_tasks,
        'total_tasks': len(tasks),
        'completion_rate': completed_tasks / len(tasks) if tasks else 0,
        'total_distance': total_distance,
        'collision_count': collision_count,
        'robots': robots,
        'final_time': step * dt
    }


def run_multi_robot_simulation():
    """运行多机器人仿真"""
    print("=" * 60)
    print("多机器人协同场景")
    print("=" * 60)
    
    # 创建地图
    print("\n创建地图...")
    grid = create_multi_robot_map()
    print(f"障碍物数量: {len(grid.obstacles)}")
    
    # 创建机器人
    num_robots = 4
    start_positions = [
        (-6, -3),
        (-6, 3),
        (6, -3),
        (6, 3),
    ]
    chassis_types = [
        ChassisType.DIFFERENTIAL_DRIVE,
        ChassisType.ACKERMANN_STEERING,
        ChassisType.MECANUM_WHEEL,
        ChassisType.DIFFERENTIAL_DRIVE,
    ]
    
    print(f"\n创建 {num_robots} 个机器人...")
    robots = create_robot_fleet(num_robots, start_positions, chassis_types)
    
    for robot in robots:
        print(f"  机器人 {robot.id}: {robot.chassis.chassis_type.value}")
    
    # 创建任务
    tasks = [
        (0, -4),
        (-4, 0),
        (4, 0),
        (0, 4),
    ]
    
    print(f"\n任务数量: {len(tasks)}")
    
    # 运行仿真
    results = simulate_multi_robot(robots, tasks, grid)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("仿真结果")
    print("=" * 60)
    print(f"完成任务数: {results['completed_tasks']}/{results['total_tasks']}")
    print(f"完成率: {results['completion_rate']*100:.1f}%")
    print(f"总行驶距离: {results['total_distance']:.2f}m")
    print(f"碰撞次数: {results['collision_count']}")
    print(f"仿真时间: {results['final_time']:.1f}s")
    
    # 每个机器人的统计
    print("\n各机器人统计:")
    for robot in robots:
        distance = sum(math.sqrt(
            (robot.trajectory[i+1][0] - robot.trajectory[i][0])**2 +
            (robot.trajectory[i+1][1] - robot.trajectory[i][1])**2
        ) for i in range(len(robot.trajectory) - 1)) if len(robot.trajectory) > 1 else 0
        
        print(f"  机器人 {robot.id}: {robot.status}, 行驶距离: {distance:.2f}m")
    
    return grid, robots, results


if __name__ == "__main__":
    run_multi_robot_simulation()
