# -*- coding: utf-8 -*-
"""
Robot Chassis Simulation Demo

机器人底盘仿真演示主程序

运行方式:
    python -m samples.robot.demo

或:
    python samples/robot/demo.py
"""

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from robot import (
        ChassisType, ChassisConfig,
        DifferentialDrive, AckermannSteering, TrackedVehicle, MecanumWheel,
        AStarPlanner, RRTPlanner, PurePursuitController, Path,
        OccupancyGrid, CircularObstacle, RectangularObstacle,
        create_chassis_mesh, create_path_mesh, create_trajectory_line,
        Color
    )
    from robot.nodes import (
        RobotChassisNode, ObstacleMapNode, PathPlanningNode,
        PathFollowingNode, TrajectoryVisualizerNode
    )
    HAS_ROBOT = True
except ImportError as e:
    print(f"导入 robot 模块失败: {e}")
    HAS_ROBOT = False


def demo_single_chassis(chassis_type: ChassisType = ChassisType.DIFFERENTIAL_DRIVE):
    """
    单底盘演示
    
    展示基本的底盘运动学和路径跟踪
    """
    print("=" * 60)
    print(f"单底盘演示: {chassis_type.value}")
    print("=" * 60)
    
    # 创建底盘
    config = ChassisConfig(
        body_length=0.6,
        body_width=0.4,
        wheel_radius=0.1,
        max_linear_velocity=1.0,
        max_angular_velocity=2.0
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
    
    print(f"\n底盘类型: {chassis_type.value}")
    print(f"车体尺寸: {config.body_length}m x {config.body_width}m")
    print(f"最大速度: {config.max_linear_velocity}m/s")
    
    # 创建地图
    print("\n创建地图...")
    grid = OccupancyGrid(10, 10, 0.1, (-5, -5))
    
    # 添加障碍物
    obstacles = [
        CircularObstacle(id=0, position=(2, 2), radius=0.5),
        CircularObstacle(id=1, position=(-1, 3), radius=0.4),
        RectangularObstacle(id=2, position=(0, -2), width=2, height=0.5),
    ]
    
    for obs in obstacles:
        grid.set_obstacle(obs)
    
    print(f"障碍物数量: {len(grid.obstacles)}")
    
    # 规划路径
    print("\n规划路径...")
    start = (-3, -3)
    goal = (3, 3)
    
    collision_fn = lambda x, y: not grid.is_in_bounds_world(x, y) or grid.check_collision(x, y)
    planner = AStarPlanner(collision_fn, resolution=0.1)
    path = planner.plan(start, goal)
    
    if path.is_valid:
        print(f"路径规划成功！路径点数: {len(path)}")
        print(f"路径长度: {path.get_remaining_distance(0):.2f}m")
    else:
        print("路径规划失败！")
        return
    
    # 路径跟踪仿真
    print("\n路径跟踪仿真...")
    controller = PurePursuitController(
        lookahead_distance=0.5,
        wheelbase=config.wheelbase,
        max_linear_velocity=config.max_linear_velocity
    )
    
    # 设置初始位置
    chassis.state.x = start[0]
    chassis.state.y = start[1]
    chassis.state.theta = 0.0
    
    trajectory = []
    current_idx = 0
    dt = 0.1
    max_steps = 500
    
    for step in range(max_steps):
        current_pose = (chassis.state.x, chassis.state.y, chassis.state.theta)
        trajectory.append(current_pose)
        
        # 检查是否到达终点
        if current_idx >= len(path) - 1:
            dist_to_goal = ((chassis.state.x - goal[0])**2 + 
                          (chassis.state.y - goal[1])**2)**0.5
            if dist_to_goal < 0.2:
                print(f"到达目标！步数: {step}")
                break
        
        # 计算控制命令
        v, omega, current_idx = controller.compute_control(current_pose, path, current_idx)
        
        # 根据底盘类型应用控制
        if chassis_type == ChassisType.DIFFERENTIAL_DRIVE:
            control = chassis.set_velocity(v, omega)
        elif chassis_type == ChassisType.ACKERMANN_STEERING:
            if abs(v) > 0.01:
                R = v / (omega + 1e-6)
                steering_angle = __import__('math').atan(config.wheelbase / R)
            else:
                steering_angle = 0.0
            control = __import__('numpy').array([v, steering_angle])
        elif chassis_type == ChassisType.TRACKED_VEHICLE:
            W = config.track_width
            v_left = v - omega * W / 2
            v_right = v + omega * W / 2
            control = __import__('numpy').array([v_left, v_right])
        elif chassis_type == ChassisType.MECANUM_WHEEL:
            control = __import__('numpy').array([v, 0, omega])
        else:
            control = chassis.set_velocity(v, omega)
        
        chassis.apply_control(control, dt)
    
    print(f"\n仿真完成！")
    print(f"轨迹点数: {len(trajectory)}")
    print(f"最终位置: ({chassis.state.x:.2f}, {chassis.state.y:.2f})")
    print(f"最终朝向: {chassis.state.theta:.2f} rad")
    
    # 计算跟踪误差
    final_error = ((chassis.state.x - goal[0])**2 + 
                   (chassis.state.y - goal[1])**2)**0.5
    print(f"终点误差: {final_error:.3f}m")
    
    return chassis, path, trajectory


def demo_path_planning_comparison():
    """
    路径规划算法对比演示
    
    比较 A*, RRT, RRT* 等算法
    """
    print("=" * 60)
    print("路径规划算法对比")
    print("=" * 60)
    
    # 创建地图
    grid = OccupancyGrid(10, 10, 0.1, (-5, -5))
    
    # 添加随机障碍物
    import numpy as np
    np.random.seed(42)
    for i in range(15):
        x = np.random.uniform(-4, 4)
        y = np.random.uniform(-4, 4)
        r = np.random.uniform(0.3, 0.8)
        obstacle = CircularObstacle(id=i, position=(x, y), radius=r)
        grid.set_obstacle(obstacle)
    
    start = (-4, -4)
    goal = (4, 4)
    
    collision_fn = lambda x, y: not grid.is_in_bounds_world(x, y) or grid.check_collision(x, y)
    
    # 测试不同算法
    algorithms = [
        ("A*", AStarPlanner(collision_fn, resolution=0.1)),
        ("RRT", RRTPlanner(collision_fn, max_iter=5000)),
        ("RRT*", RRTPlanner(collision_fn, max_iter=5000)),  # 简化处理
    ]
    
    results = []
    
    for name, planner in algorithms:
        print(f"\n测试 {name}...")
        
        import time
        t0 = time.time()
        path = planner.plan(start, goal)
        elapsed = time.time() - t0
        
        if path.is_valid:
            length = path.get_remaining_distance(0)
            print(f"  成功！时间: {elapsed*1000:.1f}ms, 路径长度: {length:.2f}m, 点数: {len(path)}")
            results.append({
                'name': name,
                'success': True,
                'time_ms': elapsed * 1000,
                'length': length,
                'num_points': len(path)
            })
        else:
            print(f"  失败！时间: {elapsed*1000:.1f}ms")
            results.append({
                'name': name,
                'success': False,
                'time_ms': elapsed * 1000
            })
    
    # 打印对比表
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(f"{'算法':<15} {'成功':>8} {'时间(ms)':>12} {'长度(m)':>12} {'点数':>8}")
    print("-" * 60)
    
    for r in results:
        success = "✓" if r['success'] else "✗"
        time_str = f"{r['time_ms']:.1f}"
        length_str = f"{r['length']:.2f}" if r['success'] else "N/A"
        points_str = str(r['num_points']) if r['success'] else "N/A"
        print(f"{r['name']:<15} {success:>8} {time_str:>12} {length_str:>12} {points_str:>8}")
    
    return results


def demo_all_chassis_types():
    """
    所有底盘类型演示
    
    展示四种底盘的运动特性
    """
    print("=" * 60)
    print("所有底盘类型演示")
    print("=" * 60)
    
    chassis_types = [
        ChassisType.DIFFERENTIAL_DRIVE,
        ChassisType.ACKERMANN_STEERING,
        ChassisType.TRACKED_VEHICLE,
        ChassisType.MECANUM_WHEEL,
    ]
    
    results = {}
    
    for chassis_type in chassis_types:
        print(f"\n{'='*40}")
        print(f"测试 {chassis_type.value}")
        print(f"{'='*40}")
        
        try:
            chassis, path, trajectory = demo_single_chassis(chassis_type)
            
            # 计算性能指标
            total_distance = 0
            for i in range(len(trajectory) - 1):
                dx = trajectory[i+1][0] - trajectory[i][0]
                dy = trajectory[i+1][1] - trajectory[i][1]
                total_distance += (dx**2 + dy**2)**0.5
            
            results[chassis_type.value] = {
                'success': True,
                'total_distance': total_distance,
                'trajectory_length': len(trajectory)
            }
        except Exception as e:
            print(f"错误: {e}")
            results[chassis_type.value] = {
                'success': False,
                'error': str(e)
            }
    
    # 打印对比
    print("\n" + "=" * 60)
    print("底盘性能对比")
    print("=" * 60)
    print(f"{'底盘类型':<25} {'状态':>10} {'行驶距离':>12}")
    print("-" * 60)
    
    for name, data in results.items():
        status = "✓" if data['success'] else "✗"
        dist = f"{data.get('total_distance', 0):.2f}m" if data['success'] else "N/A"
        print(f"{name:<25} {status:>10} {dist:>12}")
    
    return results


def demo_node_system():
    """
    节点系统演示
    
    展示 RBC 节点工作流
    """
    print("=" * 60)
    print("节点系统演示")
    print("=" * 60)
    
    if not HAS_ROBOT:
        print("Robot 模块不可用")
        return
    
    # 创建节点
    print("\n1. 创建底盘节点...")
    chassis_node = RobotChassisNode("chassis_1")
    chassis_node.set_input("chassis_type", "differential_drive")
    chassis_node.set_input("initial_pose", [0, 0, 0])
    chassis_output = chassis_node.run()
    
    print(f"   底盘类型: {chassis_output['chassis_type']}")
    print(f"   配置: {chassis_output['config']}")
    
    print("\n2. 创建地图节点...")
    map_node = ObstacleMapNode("map_1")
    map_node.set_input("width", 10)
    map_node.set_input("height", 10)
    map_node.set_input("obstacles", [
        {"type": "circular", "position": [2, 2], "radius": 0.5},
        {"type": "circular", "position": [-2, -2], "radius": 0.4},
    ])
    map_output = map_node.run()
    
    print(f"   栅格地图: {map_output['grid']}")
    print(f"   碰撞函数: {map_output['collision_fn']}")
    
    print("\n3. 创建路径规划节点...")
    planning_node = PathPlanningNode("planner_1")
    planning_node.set_input("start", [-3, -3])
    planning_node.set_input("goal", [3, 3])
    planning_node.set_input("collision_fn", map_output['collision_fn'])
    planning_node.set_input("algorithm", "astar")
    planning_output = planning_node.run()
    
    print(f"   路径有效: {planning_output['is_valid']}")
    print(f"   路径长度: {planning_output['path_length']:.2f}m")
    print(f"   路径点数: {len(planning_output['waypoints'])}")
    
    print("\n4. 创建路径跟踪节点...")
    following_node = PathFollowingNode("follower_1")
    following_node.set_input("chassis", chassis_output['chassis'])
    following_node.set_input("path", planning_output['path'])
    following_node.set_input("dt", 0.1)
    following_node.set_input("simulation_time", 20)
    following_output = following_node.run()
    
    print(f"   轨迹点数: {len(following_output['trajectory'])}")
    print(f"   最终位置: {following_output['final_pose']}")
    print(f"   跟踪误差: {following_output['tracking_error']:.3f}m")
    
    print("\n5. 创建可视化节点...")
    vis_node = TrajectoryVisualizerNode("vis_1")
    vis_node.set_input("trajectory", following_output['trajectory'])
    vis_node.set_input("path", planning_output['path'])
    vis_node.set_input("chassis", chassis_output['chassis'])
    vis_output = vis_node.run()
    
    print(f"   可视化数据键: {list(vis_output['visualization_data'].keys())}")
    
    print("\n节点系统演示完成！")
    
    return {
        'chassis': chassis_output,
        'map': map_output,
        'planning': planning_output,
        'following': following_output,
        'visualization': vis_output
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="机器人底盘仿真演示")
    parser.add_argument(
        "--demo",
        type=str,
        default="all",
        choices=["all", "single", "planning", "chassis", "nodes", "warehouse", "offroad", "multi"],
        help="选择演示类型"
    )
    parser.add_argument(
        "--chassis",
        type=str,
        default="differential_drive",
        choices=["differential_drive", "ackermann_steering", "tracked_vehicle", "mecanum_wheel"],
        help="选择底盘类型"
    )
    
    args = parser.parse_args()
    
    if not HAS_ROBOT:
        print("错误: Robot 模块不可用，请检查安装")
        return 1
    
    # 底盘类型映射
    chassis_map = {
        "differential_drive": ChassisType.DIFFERENTIAL_DRIVE,
        "ackermann_steering": ChassisType.ACKERMANN_STEERING,
        "tracked_vehicle": ChassisType.TRACKED_VEHICLE,
        "mecanum_wheel": ChassisType.MECANUM_WHEEL,
    }
    
    try:
        if args.demo == "all":
            print("\n" + "=" * 70)
            print("  机器人底盘仿真演示 - 完整演示")
            print("=" * 70 + "\n")
            
            demo_single_chassis(ChassisType.DIFFERENTIAL_DRIVE)
            print("\n")
            demo_path_planning_comparison()
            print("\n")
            demo_node_system()
        
        elif args.demo == "single":
            demo_single_chassis(chassis_map[args.chassis])
        
        elif args.demo == "planning":
            demo_path_planning_comparison()
        
        elif args.demo == "chassis":
            demo_all_chassis_types()
        
        elif args.demo == "nodes":
            demo_node_system()
        
        elif args.demo == "warehouse":
            from scene_warehouse_navigation import run_warehouse_simulation
            run_warehouse_simulation()
        
        elif args.demo == "offroad":
            from scene_offroad_terrain import run_offroad_simulation
            run_offroad_simulation()
        
        elif args.demo == "multi":
            from scene_multi_robot import run_multi_robot_simulation
            run_multi_robot_simulation()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
        return 1
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
