# -*- coding: utf-8 -*-
"""
机器人工程示例 - 展示如何使用 RoboCute Robot API

本示例展示:
1. 创建地图
2. 创建不同类型的机器人底盘
3. 路径规划
4. 机器人运动控制
5. 与 World API 集成
"""

import time
import math
import random
from pathlib import Path

try:
    import robocute as rbc
    import robocute.rbc_ext as re
    import robocute.rbc_ext.luisa as lc
    RBC_AVAILABLE = True
except ImportError:
    RBC_AVAILABLE = False
    print("RoboCute not available, running in standalone mode")

from chassis import (
    ChassisType, DifferentialChassis, AckermannChassis,
    TrackedChassis, MecanumChassis, Pose2D
)
from path_planner import AStarPlanner, RRTPlanner, PathPoint
from map_editor import MapEditor
from robot import Robot, RobotConfig, RobotFleet


def demo_chassis():
    """底盘运动学演示"""
    print("=" * 50)
    print("底盘运动学演示")
    print("=" * 50)
    
    # 差速底盘
    print("\n1. 差速底盘 (Differential)")
    diff = DifferentialChassis(wheel_radius=0.05, wheel_base=0.3)
    
    # 设置速度 (vx=0.5 m/s, omega=0.3 rad/s)
    diff.set_velocity(0.5, 0.0, 0.3)
    
    print(f"   目标速度: vx={diff.velocity.vx:.2f}, omega={diff.velocity.omega:.2f}")
    print(f"   左轮速度: {diff.wheel_speeds.left:.2f} rad/s")
    print(f"   右轮速度: {diff.wheel_speeds.right:.2f} rad/s")
    
    # 模拟运动
    dt = 0.1
    for i in range(20):
        diff.update_odometry(dt)
        pose = diff.pose
        print(f"   t={i*dt:.1f}s: x={pose.x:.3f}, y={pose.y:.3f}, theta={pose.theta:.3f}")
    
    # 阿克曼底盘
    print("\n2. 阿克曼底盘 (Ackermann)")
    acker = AckermannChassis(wheel_radius=0.05, wheel_base=0.3)
    acker.set_steering(math.radians(20))  # 20度转向角
    acker.set_velocity(0.5, 0.0, 0.0)
    
    for i in range(10):
        acker.update_odometry(dt)
        pose = acker.pose
        print(f"   t={i*dt:.1f}s: x={pose.x:.3f}, y={pose.y:.3f}, theta={pose.theta:.3f}")
    
    # 麦轮底盘
    print("\n3. 麦轮底盘 (Mecanum)")
    mec = MecanumChassis(wheel_radius=0.05, wheel_base=0.3, track_width=0.3)
    
    # 斜向移动
    mec.set_velocity(0.3, 0.3, 0.0)
    print(f"   目标速度: vx={mec.velocity.vx:.2f}, vy={mec.velocity.vy:.2f}")
    
    for i in range(10):
        mec.update_odometry(dt)
        pose = mec.pose
        print(f"   t={i*dt:.1f}s: x={pose.x:.3f}, y={pose.y:.3f}, theta={pose.theta:.3f}")


def demo_map_editor():
    """地图编辑器演示"""
    print("\n" + "=" * 50)
    print("地图编辑器演示")
    print("=" * 50)
    
    # 创建带围墙的地图
    map_editor = MapEditor.create_with_walls(
        width=10.0, height=10.0, resolution=0.2
    )
    
    print(f"\n地图尺寸: {map_editor.width}x{map_editor.height} 格子")
    print(f"分辨率: {map_editor.resolution} m/cell")
    
    # 添加障碍物
    map_editor.add_rect_obstacle_world(2.0, 2.0, 1.0, 3.0)
    map_editor.add_rect_obstacle_world(6.0, 4.0, 2.0, 1.0)
    map_editor.add_circle_obstacle(5.0, 7.0, 1.0)
    
    # 统计障碍物数量
    obstacle_count = 0
    for y in range(map_editor.height):
        for x in range(map_editor.width):
            if map_editor.is_occupied(x, y):
                obstacle_count += 1
    
    print(f"障碍物格子数: {obstacle_count}")
    
    # 膨胀地图
    map_editor.inflate(radius=0.3, cost_scaling=2.0)
    
    # 保存地图
    # map_editor.save("demo_map.json")
    print("地图已创建")
    
    return map_editor


def demo_path_planning(map_editor: MapEditor):
    """路径规划演示"""
    print("\n" + "=" * 50)
    print("路径规划演示")
    print("=" * 50)
    
    # A* 规划
    print("\n1. A* 算法")
    astar = AStarPlanner(
        map_data=map_editor.data,
        resolution=map_editor.resolution,
        origin=(map_editor.metadata.origin_x, map_editor.metadata.origin_y)
    )
    
    start = PathPoint(1.0, 1.0)
    goal = PathPoint(8.0, 8.0)
    
    print(f"   起点: ({start.x}, {start.y})")
    print(f"   终点: ({goal.x}, {goal.y})")
    
    path = astar.plan(start, goal)
    
    if path:
        print(f"   找到路径! 长度: {len(path)} 点, 总距离: {path.total_length():.2f} m")
        print(f"   路径点: ", end="")
        for p in path.points[:5]:
            print(f"({p.x:.1f}, {p.y:.1f})", end=" ")
        print("...")
    else:
        print("   未找到路径")
    
    # RRT 规划
    print("\n2. RRT 算法")
    rrt = RRTPlanner(
        map_data=map_editor.data,
        resolution=map_editor.resolution,
        origin=(map_editor.metadata.origin_x, map_editor.metadata.origin_y),
        max_iter=2000,
        step_size=0.3
    )
    
    path_rrt = rrt.plan(start, goal)
    
    if path_rrt:
        print(f"   找到路径! 长度: {len(path_rrt)} 点, 总距离: {path_rrt.total_length():.2f} m")
    else:
        print("   未找到路径")
    
    return path


def demo_robot_control(map_editor: MapEditor, path):
    """机器人控制演示"""
    print("\n" + "=" * 50)
    print("机器人控制演示")
    print("=" * 50)
    
    # 创建差速机器人
    config = RobotConfig(
        name="diff_robot_1",
        chassis_type=ChassisType.DIFFERENTIAL,
        wheel_radius=0.05,
        wheel_base=0.3,
        max_linear_speed=1.0,
        max_angular_speed=1.0
    )
    
    robot = Robot(config)
    robot.set_pose(1.0, 1.0, 0.0)
    
    # 设置规划器
    planner = AStarPlanner(
        map_data=map_editor.data,
        resolution=map_editor.resolution
    )
    robot.set_planner(planner)
    
    # 设置目标
    print("\n机器人移动到目标位置...")
    success = robot.move_to(8.0, 8.0)
    
    if success:
        # 模拟运行
        dt = 0.1
        for i in range(200):  # 最多 20 秒
            robot.update(dt)
            
            if not robot.is_moving:
                print(f"到达目标! 最终位置: ({robot.pose.x:.3f}, {robot.pose.y:.3f})")
                break
            
            if i % 10 == 0:
                print(f"   位置: ({robot.pose.x:.3f}, {robot.pose.y:.3f}), "
                      f"速度: ({robot.velocity.vx:.2f}, {robot.velocity.omega:.2f})")
        else:
            print("运行超时")
            robot.stop()
    
    return robot


def demo_robot_fleet():
    """机器人群组演示"""
    print("\n" + "=" * 50)
    print("机器人群组演示")
    print("=" * 50)
    
    fleet = RobotFleet()
    
    # 添加多个机器人
    for i, chassis_type in enumerate([
        ChassisType.DIFFERENTIAL,
        ChassisType.ACKERMANN,
        ChassisType.MECANUM
    ]):
        config = RobotConfig(
            name=f"robot_{i}",
            chassis_type=chassis_type
        )
        robot = Robot(config)
        robot.set_pose(i * 2.0, 0.0, 0.0)
        fleet.add_robot(robot)
    
    print(f"\n创建了 {len(fleet._robots)} 个机器人")
    
    # 分配任务
    for robot_id in fleet._robots.keys():
        target_x = 5.0 + random.uniform(-1.0, 1.0)
        target_y = 5.0 + random.uniform(-1.0, 1.0)
        fleet.assign_task(robot_id, target_x, target_y)
        print(f"   {robot_id}: 任务 -> ({target_x:.1f}, {target_y:.1f})")
    
    # 模拟运行
    dt = 0.1
    for _ in range(100):
        fleet.update_all(dt)
    
    # 获取最终位置
    poses = fleet.get_all_poses()
    print("\n最终位置:")
    for rid, pose in poses.items():
        print(f"   {rid}: ({pose.x:.3f}, {pose.y:.3f})")
    
    fleet.clear()


def demo_world_integration():
    """World API 集成演示"""
    print("\n" + "=" * 50)
    print("World API 集成演示")
    print("=" * 50)
    
    if not RBC_AVAILABLE:
        print("RoboCute 不可用，跳过 World API 演示")
        return
    
    print("\n注意: 此演示需要有效的 RoboCute 项目路径")
    print("使用方法:")
    print("  python -m robo.main --project <path> --backend dx")


def main():
    """主函数"""
    random.seed(42)
    
    print("\n" + "=" * 60)
    print("RoboCute 机器人工程示例")
    print("=" * 60)
    
    # 1. 底盘演示
    demo_chassis()
    
    # 2. 地图编辑器演示
    map_editor = demo_map_editor()
    
    # 3. 路径规划演示
    path = demo_path_planning(map_editor)
    
    # 4. 机器人控制演示
    robot = demo_robot_control(map_editor, path)
    
    # 5. 机器人群组演示
    demo_robot_fleet()
    
    # 6. World API 集成
    demo_world_integration()
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
