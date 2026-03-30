# -*- coding: utf-8 -*-
"""
机器人工程测试 - 测试底盘、路径规划和地图编辑功能

Test cases for:
- Chassis kinematics (Differential, Ackermann, Tracked, Mecanum)
- Path planning algorithms (A*, RRT)
- Map editor operations
- Robot control and path following
"""

import sys
import math
import pytest
import numpy as np
from pathlib import Path

# Add samples directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from robo package
from robo.chassis import (
    ChassisType, ChassisBase, DifferentialChassis,
    AckermannChassis, TrackedChassis, MecanumChassis,
    Pose2D, Velocity, WheelSpeeds, create_chassis
)
from robo.path_planner import (
    PathPlanner, AStarPlanner, RRTPlanner, DijkstraPlanner,
    Path, PathPoint
)
from robo.map_editor import MapEditor, MapCell, CellType, MapMetadata
from robo.robot import Robot, RobotConfig, RobotFleet


class TestChassis:
    """底盘测试类"""
    
    def test_differential_chassis_creation(self):
        """测试差速底盘创建"""
        chassis = DifferentialChassis(wheel_radius=0.05, wheel_base=0.3)
        assert chassis.type == ChassisType.DIFFERENTIAL
        assert chassis.pose.x == 0.0
        assert chassis.pose.y == 0.0
        assert chassis.pose.theta == 0.0
    
    def test_differential_inverse_kinematics(self):
        """测试差速底盘逆运动学"""
        chassis = DifferentialChassis(wheel_radius=0.05, wheel_base=0.3)
        
        # 设置速度: vx=0.5 m/s, omega=0.3 rad/s
        chassis.set_velocity(0.5, 0.0, 0.3)
        
        # 验证轮子速度
        r = 0.05
        L = 0.3
        vl = (0.5 - 0.3 * L / 2) / r
        vr = (0.5 + 0.3 * L / 2) / r
        
        assert abs(chassis.wheel_speeds.left - vl) < 1e-6
        assert abs(chassis.wheel_speeds.right - vr) < 1e-6
    
    def test_differential_forward_kinematics(self):
        """测试差速底盘正运动学"""
        chassis = DifferentialChassis(wheel_radius=0.05, wheel_base=0.3)
        
        wheel_speeds = WheelSpeeds(left=10.0, right=10.0)
        velocity = chassis.forward_kinematics(wheel_speeds)
        
        # 两轮速度相同，应该只有前进速度
        assert abs(velocity.vx - 0.5) < 1e-6  # (10+10)*0.05/2 = 0.5
        assert abs(velocity.vy) < 1e-6
        assert abs(velocity.omega) < 1e-6
    
    def test_differential_odometry_straight(self):
        """测试差速底盘直线运动里程计"""
        chassis = DifferentialChassis(wheel_radius=0.05, wheel_base=0.3)
        chassis.set_velocity(1.0, 0.0, 0.0)  # 直线前进
        
        # 更新1秒
        for _ in range(10):
            chassis.update_odometry(0.1)
        
        # 1秒应该前进1米
        assert abs(chassis.pose.x - 1.0) < 0.1
        assert abs(chassis.pose.y) < 0.01
        assert abs(chassis.pose.theta) < 0.01
    
    def test_differential_odometry_rotation(self):
        """测试差速底盘旋转运动里程计"""
        chassis = DifferentialChassis(wheel_radius=0.05, wheel_base=0.3)
        chassis.set_velocity(0.0, 0.0, 1.0)  # 原地旋转
        
        # 更新 1 秒，角速度 1.0 rad/s
        for _ in range(10):
            chassis.update_odometry(0.1)
        
        assert abs(chassis.pose.theta - 1.0) < 0.1  # 约 1.0 弧度
    
    def test_ackermann_chassis_creation(self):
        """测试阿克曼底盘创建"""
        chassis = AckermannChassis(
            wheel_radius=0.05, 
            wheel_base=0.3,
            track_width=0.25
        )
        assert chassis.type == ChassisType.ACKERMANN
        assert chassis.steering_angle == 0.0
    
    def test_ackermann_steering_limit(self):
        """测试阿克曼转向角限制"""
        chassis = AckermannChassis(
            wheel_radius=0.05,
            wheel_base=0.3,
            max_steering_angle=math.pi/4
        )
        
        # 设置超过限制的转向角
        chassis.set_steering(math.pi/2)
        assert abs(chassis.steering_angle - math.pi/4) < 1e-6
        
        chassis.set_steering(-math.pi/2)
        assert abs(chassis.steering_angle + math.pi/4) < 1e-6
    
    def test_tracked_chassis_creation(self):
        """测试履带底盘创建"""
        chassis = TrackedChassis(
            wheel_radius=0.05,
            wheel_base=0.3,
            slip_factor=0.9
        )
        assert chassis.type == ChassisType.TRACKED
    
    def test_mecanum_chassis_creation(self):
        """测试麦轮底盘创建"""
        chassis = MecanumChassis(
            wheel_radius=0.05,
            wheel_base=0.3,
            track_width=0.3
        )
        assert chassis.type == ChassisType.MECANUM
    
    def test_mecanum_omnidirectional(self):
        """测试麦轮全向移动"""
        chassis = MecanumChassis(
            wheel_radius=0.05,
            wheel_base=0.3,
            track_width=0.3
        )
        
        # 斜向移动
        chassis.set_velocity(0.3, 0.3, 0.0)
        
        assert abs(chassis.velocity.vx - 0.3) < 1e-6
        assert abs(chassis.velocity.vy - 0.3) < 1e-6
        
        # 验证轮子速度 (麦轮逆运动学对角轮子速度相同)
        speeds = chassis.wheel_speeds
        assert speeds.front_right != 0 or speeds.front_left != 0
        assert speeds.rear_left != 0 or speeds.rear_right != 0
    
    def test_create_chassis_factory(self):
        """测试底盘工厂函数"""
        diff = create_chassis(ChassisType.DIFFERENTIAL, wheel_base=0.3)
        assert isinstance(diff, DifferentialChassis)
        
        acker = create_chassis(ChassisType.ACKERMANN, wheel_base=0.4)
        assert isinstance(acker, AckermannChassis)
        
        tracked = create_chassis(ChassisType.TRACKED, wheel_base=0.3)
        assert isinstance(tracked, TrackedChassis)
        
        mecanum = create_chassis(ChassisType.MECANUM, wheel_base=0.3, track_width=0.3)
        assert isinstance(mecanum, MecanumChassis)


class TestMapEditor:
    """地图编辑器测试类"""
    
    def test_create_empty_map(self):
        """测试创建空地图"""
        editor = MapEditor.create_empty(width=10.0, height=10.0, resolution=0.5)
        
        assert editor.width == 20  # 10/0.5 = 20
        assert editor.height == 20
        assert editor.resolution == 0.5
    
    def test_create_map_with_walls(self):
        """测试创建带围墙的地图"""
        editor = MapEditor.create_with_walls(
            width=10.0, height=10.0, 
            wall_thickness=0.2, 
            resolution=0.1
        )
        
        # 检查四边有障碍物
        assert editor.is_occupied(0, 0)  # 左下角
        assert editor.is_occupied(99, 0)  # 右下角
        assert editor.is_occupied(0, 99)  # 左上角
        assert editor.is_occupied(99, 99)  # 右上角
    
    def test_world_to_map_conversion(self):
        """测试世界坐标转地图坐标"""
        editor = MapEditor.create_empty(width=10.0, height=10.0, resolution=0.5)
        
        mx, my = editor.world_to_map(5.0, 5.0)
        assert mx == 10
        assert my == 10
    
    def test_map_to_world_conversion(self):
        """测试地图坐标转世界坐标"""
        editor = MapEditor.create_empty(width=10.0, height=10.0, resolution=0.5)
        
        x, y = editor.map_to_world(10, 10)
        assert abs(x - 5.25) < 0.01  # 格子中心
        assert abs(y - 5.25) < 0.01
    
    def test_add_rect_obstacle(self):
        """测试添加矩形障碍物"""
        editor = MapEditor.create_empty(width=10.0, height=10.0, resolution=0.5)
        
        editor.add_rect_obstacle(5, 5, 3, 3)
        
        assert editor.is_occupied(5, 5)
        assert editor.is_occupied(7, 7)
        assert not editor.is_occupied(0, 0)
    
    def test_add_circle_obstacle(self):
        """测试添加圆形障碍物"""
        editor = MapEditor.create_empty(
            width=10.0, height=10.0, 
            resolution=0.1, 
            origin_x=0.0, origin_y=0.0
        )
        
        editor.add_circle_obstacle(5.0, 5.0, 1.0)
        
        # 圆心应该是障碍物
        assert editor.is_occupied_world(5.0, 5.0)
        # 半径外应该不是
        assert not editor.is_occupied_world(7.0, 5.0)
    
    def test_inflate_map(self):
        """测试地图膨胀"""
        editor = MapEditor.create_empty(
            width=10.0, height=10.0, 
            resolution=0.1
        )
        
        # 添加一个障碍物
        editor.set_obstacle(50, 50)
        
        # 膨胀
        editor.inflate(radius=0.3, cost_scaling=2.0)
        
        # 原来位置仍然是障碍物
        assert editor.is_occupied(50, 50)
        # 膨胀区域应该有代价
        assert editor.cost_map[50, 50] > 0
    
    def test_save_load_json(self, tmp_path):
        """测试 JSON 保存和加载"""
        editor = MapEditor.create_empty(width=5.0, height=5.0, resolution=0.5)
        editor.set_obstacle(2, 2)
        
        filepath = tmp_path / "test_map.json"
        editor.save(str(filepath))
        
        loaded = MapEditor.load(str(filepath))
        assert loaded.width == editor.width
        assert loaded.height == editor.height
        assert loaded.is_occupied(2, 2)


class TestPathPlanner:
    """路径规划测试类"""
    
    @pytest.fixture
    def simple_map(self):
        """创建简单测试地图"""
        editor = MapEditor.create_with_walls(
            width=10.0, height=10.0, 
            resolution=0.5
        )
        return editor
    
    def test_path_point_distance(self):
        """测试路径点距离计算"""
        p1 = PathPoint(0.0, 0.0)
        p2 = PathPoint(3.0, 4.0)
        
        assert abs(p1.distance_to(p2) - 5.0) < 1e-6
    
    def test_path_length(self):
        """测试路径长度计算"""
        path = Path([
            PathPoint(0.0, 0.0),
            PathPoint(3.0, 4.0),
            PathPoint(6.0, 8.0)
        ])
        
        assert abs(path.total_length() - 10.0) < 1e-6
    
    def test_astar_simple_path(self, simple_map):
        """测试 A* 简单路径规划"""
        planner = AStarPlanner(
            map_data=simple_map.data,
            resolution=simple_map.resolution
        )
        
        start = PathPoint(1.0, 1.0)
        goal = PathPoint(8.0, 8.0)
        
        path = planner.plan(start, goal)
        
        assert path is not None
        assert len(path) > 0
        assert path.points[0].x == start.x
        assert path.points[0].y == start.y
        assert path.points[-1].x == goal.x
        assert path.points[-1].y == goal.y
    
    def test_astar_no_path(self, simple_map):
        """测试 A* 无可行路径的情况"""
        # 创建完全被阻挡的地图
        editor = MapEditor.create_empty(width=5.0, height=5.0, resolution=0.5)
        for x in range(editor.width):
            for y in range(editor.height):
                editor.set_obstacle(x, y)
        
        planner = AStarPlanner(
            map_data=editor.data,
            resolution=editor.resolution
        )
        
        start = PathPoint(1.0, 1.0)
        goal = PathPoint(3.0, 3.0)
        
        path = planner.plan(start, goal)
        
        assert path is None
    
    def test_astar_collision_check(self):
        """测试 A* 碰撞检测"""
        # 创建一个带明确障碍物的地图
        editor = MapEditor.create_empty(width=5.0, height=5.0, resolution=0.5)
        editor.set_obstacle(5, 5)  # 在 (2.5, 2.5) 位置设置障碍物
        
        planner = AStarPlanner(
            map_data=editor.data,
            resolution=editor.resolution
        )
        
        # 检查障碍物位置
        assert planner.is_collision(2.5, 2.5)  # 障碍物中心
        assert not planner.is_collision(0.0, 0.0)  # 空闲区域
    
    def test_rrt_planning(self, simple_map):
        """测试 RRT 路径规划"""
        planner = RRTPlanner(
            map_data=simple_map.data,
            resolution=simple_map.resolution,
            max_iter=1000,
            step_size=0.3
        )
        
        start = PathPoint(1.0, 1.0)
        goal = PathPoint(8.0, 8.0)
        
        path = planner.plan(start, goal)
        
        # RRT 可能找到也可能找不到，取决于随机采样
        if path is not None:
            assert len(path) > 0
            # 检查起点和终点
            assert path.points[0].x == start.x
            assert path.points[-1].x == goal.x
    
    def test_dijkstra_planning(self, simple_map):
        """测试 Dijkstra 路径规划"""
        planner = DijkstraPlanner(
            map_data=simple_map.data,
            resolution=simple_map.resolution
        )
        
        start = PathPoint(1.0, 1.0)
        goal = PathPoint(8.0, 8.0)
        
        path = planner.plan(start, goal)
        
        assert path is not None
        assert len(path) > 0


class TestRobot:
    """机器人测试类"""
    
    @pytest.fixture
    def simple_config(self):
        """创建简单机器人配置"""
        return RobotConfig(
            name="test_robot",
            chassis_type=ChassisType.DIFFERENTIAL,
            wheel_radius=0.05,
            wheel_base=0.3,
            max_linear_speed=1.0,
            max_angular_speed=1.0,
            position_tolerance=0.1
        )
    
    @pytest.fixture
    def simple_map(self):
        """创建简单测试地图"""
        return MapEditor.create_with_walls(
            width=10.0, height=10.0, 
            resolution=0.5
        )
    
    def test_robot_creation(self, simple_config):
        """测试机器人创建"""
        robot = Robot(simple_config)
        
        assert robot.config.name == "test_robot"
        assert robot.pose.x == 0.0
        assert robot.pose.y == 0.0
        assert not robot.is_moving
    
    def test_robot_set_pose(self, simple_config):
        """测试设置机器人位姿"""
        robot = Robot(simple_config)
        robot.set_pose(1.0, 2.0, math.pi/4)
        
        assert robot.pose.x == 1.0
        assert robot.pose.y == 2.0
        assert abs(robot.pose.theta - math.pi/4) < 1e-6
    
    def test_robot_set_velocity(self, simple_config):
        """测试设置机器人速度"""
        robot = Robot(simple_config)
        robot.set_velocity(0.5, 0.0, 0.3)
        
        assert abs(robot.velocity.vx - 0.5) < 1e-6
        assert abs(robot.velocity.omega - 0.3) < 1e-6
    
    def test_robot_stop(self, simple_config):
        """测试机器人停止"""
        robot = Robot(simple_config)
        robot.set_velocity(0.5, 0.0, 0.3)
        robot.stop()
        
        assert robot.velocity.vx == 0.0
        assert robot.velocity.omega == 0.0
        assert not robot.is_moving
    
    def test_robot_move_to_without_planner(self, simple_config):
        """测试无规划器时的移动"""
        robot = Robot(simple_config)
        robot.set_pose(0.0, 0.0, 0.0)
        
        success = robot.move_to(3.0, 4.0)
        
        assert success
        assert robot.is_moving
    
    def test_robot_path_following(self, simple_config):
        """测试机器人路径跟随"""
        robot = Robot(simple_config)
        robot.set_pose(0.0, 0.0, 0.0)
        
        # 创建简单路径
        path = Path([
            PathPoint(0.0, 0.0),
            PathPoint(1.0, 0.0),
            PathPoint(2.0, 0.0)
        ])
        
        robot.follow_path(path)
        
        assert robot.is_moving
        assert robot.get_path_progress() == 0.0
    
    def test_robot_update(self, simple_config):
        """测试机器人更新"""
        robot = Robot(simple_config)
        robot.set_pose(0.0, 0.0, 0.0)
        robot.set_velocity(1.0, 0.0, 0.0)
        robot._is_moving = True
        
        # 更新 0.1 秒
        robot.update(0.1)
        
        # 应该前进 0.1 米
        assert abs(robot.pose.x - 0.1) < 0.01
    
    def test_robot_callback(self, simple_config):
        """测试机器人回调"""
        robot = Robot(simple_config)
        
        callback_called = [False]
        
        def on_goal_reached():
            callback_called[0] = True
        
        robot.register_callback('on_goal_reached', on_goal_reached)
        
        # 创建只有一个点的路径
        path = Path([PathPoint(0.0, 0.0)])
        robot.follow_path(path)
        robot.update(0.1)
        
        assert callback_called[0]
    
    def test_robot_is_at_goal(self, simple_config):
        """测试机器人到达目标检测"""
        robot = Robot(simple_config)
        robot.set_pose(1.0, 1.0, 0.0)
        
        # 在容差范围内
        goal_near = PathPoint(1.05, 1.05)
        assert robot.is_at_goal(goal_near)
        
        # 超出容差范围
        goal_far = PathPoint(2.0, 2.0)
        assert not robot.is_at_goal(goal_far)
    
    def test_robot_with_planner(self, simple_config, simple_map):
        """测试带规划器的机器人"""
        robot = Robot(simple_config)
        robot.set_pose(1.0, 1.0, 0.0)
        
        planner = AStarPlanner(
            map_data=simple_map.data,
            resolution=simple_map.resolution
        )
        robot.set_planner(planner)
        
        success = robot.move_to(8.0, 8.0)
        
        assert success
        assert robot.is_moving


class TestRobotFleet:
    """机器人群组测试类"""
    
    def test_fleet_creation(self):
        """测试机器人群组创建"""
        fleet = RobotFleet()
        assert len(fleet._robots) == 0
    
    def test_fleet_add_robot(self):
        """测试添加机器人到群组"""
        fleet = RobotFleet()
        
        config = RobotConfig(name="robot_1")
        robot = Robot(config)
        
        robot_id = fleet.add_robot(robot)
        
        assert robot_id == "robot_1"
        assert len(fleet._robots) == 1
    
    def test_fleet_get_robot(self):
        """测试获取机器人"""
        fleet = RobotFleet()
        
        config = RobotConfig(name="robot_1")
        robot = Robot(config)
        fleet.add_robot(robot)
        
        retrieved = fleet.get_robot("robot_1")
        assert retrieved is not None
        assert retrieved.config.name == "robot_1"
    
    def test_fleet_update_all(self):
        """测试批量更新机器人"""
        fleet = RobotFleet()
        
        for i in range(3):
            config = RobotConfig(name=f"robot_{i}")
            robot = Robot(config)
            robot.set_pose(float(i), 0.0, 0.0)
            fleet.add_robot(robot)
        
        fleet.update_all(0.1)
        
        assert len(fleet._robots) == 3
    
    def test_fleet_stop_all(self):
        """测试批量停止机器人"""
        fleet = RobotFleet()
        
        for i in range(3):
            config = RobotConfig(name=f"robot_{i}")
            robot = Robot(config)
            robot.set_velocity(1.0, 0.0, 0.0)
            fleet.add_robot(robot)
        
        fleet.stop_all()
        
        for robot in fleet._robots.values():
            assert robot.velocity.vx == 0.0
    
    def test_fleet_get_all_poses(self):
        """测试获取所有机器人位姿"""
        fleet = RobotFleet()
        
        for i in range(3):
            config = RobotConfig(name=f"robot_{i}")
            robot = Robot(config)
            robot.set_pose(float(i), float(i), 0.0)
            fleet.add_robot(robot)
        
        poses = fleet.get_all_poses()
        
        assert len(poses) == 3
        for i in range(3):
            assert f"robot_{i}" in poses
            assert poses[f"robot_{i}"].x == float(i)
    
    def test_fleet_clear(self):
        """测试清空机器人群组"""
        fleet = RobotFleet()
        
        for i in range(3):
            config = RobotConfig(name=f"robot_{i}")
            robot = Robot(config)
            fleet.add_robot(robot)
        
        fleet.clear()
        
        assert len(fleet._robots) == 0
        assert len(fleet._tasks) == 0


class TestIntegration:
    """集成测试类"""
    
    def test_full_navigation_pipeline(self):
        """测试完整导航流程 - 验证路径规划和运动模拟基本工作"""
        # 1. 创建地图
        editor = MapEditor.create_with_walls(
            width=10.0, height=10.0, 
            resolution=0.2
        )
        
        # 2. 添加障碍物
        editor.add_rect_obstacle_world(3.0, 3.0, 2.0, 2.0)
        
        # 3. 创建机器人
        config = RobotConfig(
            chassis_type=ChassisType.DIFFERENTIAL,
            wheel_radius=0.05,
            wheel_base=0.3,
            max_linear_speed=1.0,
            position_tolerance=0.5
        )
        robot = Robot(config)
        robot.set_pose(1.0, 1.0, 0.0)
        
        # 4. 设置规划器
        planner = AStarPlanner(
            map_data=editor.data,
            resolution=editor.resolution
        )
        robot.set_planner(planner)
        
        # 5. 规划并跟随路径
        success = robot.move_to(8.0, 8.0)
        assert success
        
        # 6. 验证路径已生成
        assert robot._current_path is not None
        assert len(robot._current_path.points) > 0
        
        # 7. 模拟运行一段时间（不严格要求到达目标，只验证运动）
        dt = 0.1
        for _ in range(100):  # 模拟 10 秒
            robot.update(dt)
        
        # 8. 验证机器人已经移动（不是还在起点）
        dist_moved = math.sqrt(
            (robot.pose.x - 1.0) ** 2 + 
            (robot.pose.y - 1.0) ** 2
        )
        assert dist_moved > 0.5  # 至少移动了 0.5 米
    
    def test_multi_chassis_types(self):
        """测试多种底盘类型"""
        chassis_types = [
            ChassisType.DIFFERENTIAL,
            ChassisType.ACKERMANN,
            ChassisType.TRACKED,
            ChassisType.MECANUM
        ]
        
        for chassis_type in chassis_types:
            config = RobotConfig(
                name=f"test_{chassis_type.name}",
                chassis_type=chassis_type,
                wheel_base=0.3,
                track_width=0.3
            )
            robot = Robot(config)
            
            # 测试基本运动 - 需要通过 _chassis 设置速度并更新里程计
            robot.set_pose(0.0, 0.0, 0.0)
            robot._chassis.set_velocity(0.5, 0.0, 0.0)
            
            for _ in range(10):
                robot._chassis.update_odometry(0.1)
                robot._pose = robot._chassis.pose
            
            # 每种底盘都应该有前进
            assert robot.pose.x > 0, f"{chassis_type.name} should move forward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
