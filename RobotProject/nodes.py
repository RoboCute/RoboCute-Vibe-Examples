# -*- coding: utf-8 -*-
"""
Robot Chassis RBC Nodes

机器人底盘仿真 RBC 节点实现

工作流:
ObstacleMapNode → PathPlanningNode → PathFollowingNode → UIPCSimNode → AnimationOutput
                                    ↓
                            RobotChassisNode
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import math

try:
    from robocute.node_base import RBCNode, NodeInput, NodeOutput
    from robocute.node_registry import register_node
    HAS_ROBOCUTE = True
except ImportError:
    # 在没有 robocute 时提供兼容的基类
    HAS_ROBOCUTE = False
    from .mock_nodes import RBCNode, NodeInput, NodeOutput, register_node

from .chassis import (
    ChassisType, ChassisConfig, ChassisState,
    DifferentialDrive, AckermannSteering, 
    TrackedVehicle, MecanumWheel, ChassisBase
)
from .path_planning import (
    Path, AStarPlanner, RRTPlanner, RRTStarPlanner,
    DWAPlanner, PurePursuitController
)
from .map_system import (
    OccupancyGrid, Costmap, 
    CircularObstacle, RectangularObstacle, DynamicObstacle,
    Obstacle
)
from .visualization import (
    create_chassis_mesh, create_path_mesh, create_trajectory_line,
    create_velocity_arrow, create_steering_indicator,
    get_chassis_type_color, Color
)


@register_node
class RobotChassisNode(RBCNode):
    """
    机器人底盘节点
    
    创建和配置机器人底盘模型
    """
    
    NODE_TYPE = "robot_chassis"
    DISPLAY_NAME = "机器人底盘"
    CATEGORY = "机器人仿真"
    DESCRIPTION = "创建机器人底盘模型"
    
    @classmethod
    def get_inputs(cls) -> List[NodeInput]:
        return [
            NodeInput(
                name="chassis_type",
                type="string",
                required=True,
                default="differential_drive",
                description="底盘类型: differential_drive, ackermann_steering, tracked_vehicle, mecanum_wheel"
            ),
            NodeInput(
                name="config",
                type="dict",
                required=False,
                default=None,
                description="底盘配置参数字典"
            ),
            NodeInput(
                name="initial_pose",
                type="list",
                required=False,
                default=[0.0, 0.0, 0.0],
                description="初始位姿 [x, y, theta]"
            ),
        ]
    
    @classmethod
    def get_outputs(cls) -> List[NodeOutput]:
        return [
            NodeOutput(name="chassis", type="object", description="底盘对象"),
            NodeOutput(name="chassis_type", type="string", description="底盘类型"),
            NodeOutput(name="config", type="dict", description="配置信息"),
        ]
    
    def execute(self) -> Dict[str, Any]:
        chassis_type_str = self.get_input("chassis_type", "differential_drive")
        config_dict = self.get_input("config", None)
        initial_pose = self.get_input("initial_pose", [0.0, 0.0, 0.0])
        
        # 创建配置
        if config_dict:
            config = ChassisConfig(**config_dict)
        else:
            config = ChassisConfig()
        
        # 创建底盘
        chassis_type_map = {
            "differential_drive": (ChassisType.DIFFERENTIAL_DRIVE, DifferentialDrive),
            "ackermann_steering": (ChassisType.ACKERMANN_STEERING, AckermannSteering),
            "tracked_vehicle": (ChassisType.TRACKED_VEHICLE, TrackedVehicle),
            "mecanum_wheel": (ChassisType.MECANUM_WHEEL, MecanumWheel),
        }
        
        if chassis_type_str not in chassis_type_map:
            raise ValueError(f"未知的底盘类型: {chassis_type_str}")
        
        chassis_type, chassis_class = chassis_type_map[chassis_type_str]
        chassis = chassis_class(config)
        
        # 设置初始位姿
        if len(initial_pose) >= 3:
            chassis.state.x = initial_pose[0]
            chassis.state.y = initial_pose[1]
            chassis.state.theta = initial_pose[2]
        
        return {
            "chassis": chassis,
            "chassis_type": chassis_type_str,
            "config": config.__dict__
        }


@register_node
class ObstacleMapNode(RBCNode):
    """
    障碍物地图节点
    
    创建栅格地图和障碍物
    """
    
    NODE_TYPE = "obstacle_map"
    DISPLAY_NAME = "障碍物地图"
    CATEGORY = "机器人仿真"
    DESCRIPTION = "创建障碍物地图"
    
    @classmethod
    def get_inputs(cls) -> List[NodeInput]:
        return [
            NodeInput(
                name="width",
                type="number",
                required=False,
                default=10.0,
                description="地图宽度 (m)"
            ),
            NodeInput(
                name="height",
                type="number",
                required=False,
                default=10.0,
                description="地图高度 (m)"
            ),
            NodeInput(
                name="resolution",
                type="number",
                required=False,
                default=0.1,
                description="栅格分辨率 (m)"
            ),
            NodeInput(
                name="origin",
                type="list",
                required=False,
                default=[-5.0, -5.0],
                description="地图原点 [x, y]"
            ),
            NodeInput(
                name="obstacles",
                type="list",
                required=False,
                default=[],
                description="障碍物列表 [{type, position, ...}, ...]"
            ),
        ]
    
    @classmethod
    def get_outputs(cls) -> List[NodeOutput]:
        return [
            NodeOutput(name="grid", type="object", description="栅格地图对象"),
            NodeOutput(name="costmap", type="object", description="成本地图对象"),
            NodeOutput(name="collision_fn", type="callable", description="碰撞检测函数"),
        ]
    
    def execute(self) -> Dict[str, Any]:
        width = self.get_input("width", 10.0)
        height = self.get_input("height", 10.0)
        resolution = self.get_input("resolution", 0.1)
        origin = tuple(self.get_input("origin", [-5.0, -5.0]))
        obstacles_data = self.get_input("obstacles", [])
        
        # 创建栅格地图
        grid = OccupancyGrid(width, height, resolution, origin)
        
        # 添加障碍物
        obstacle_id = 0
        for obs_data in obstacles_data:
            obs_type = obs_data.get("type", "circular")
            
            if obs_type == "circular":
                obstacle = CircularObstacle(
                    id=obstacle_id,
                    position=tuple(obs_data.get("position", [0, 0])),
                    radius=obs_data.get("radius", 0.5)
                )
            elif obs_type == "rectangular":
                obstacle = RectangularObstacle(
                    id=obstacle_id,
                    position=tuple(obs_data.get("position", [0, 0])),
                    width=obs_data.get("width", 1.0),
                    height=obs_data.get("height", 1.0),
                    angle=obs_data.get("angle", 0.0)
                )
            elif obs_type == "dynamic":
                obstacle = DynamicObstacle(
                    id=obstacle_id,
                    position=tuple(obs_data.get("position", [0, 0])),
                    radius=obs_data.get("radius", 0.5),
                    velocity=tuple(obs_data.get("velocity", [0, 0])),
                    trajectory=obs_data.get("trajectory", [])
                )
            else:
                continue
            
            grid.set_obstacle(obstacle)
            obstacle_id += 1
        
        # 创建成本地图
        costmap = Costmap(grid)
        
        # 碰撞检测函数
        def collision_fn(x: float, y: float) -> bool:
            return not grid.is_in_bounds_world(x, y) or grid.check_collision(x, y)
        
        return {
            "grid": grid,
            "costmap": costmap,
            "collision_fn": collision_fn
        }


@register_node
class PathPlanningNode(RBCNode):
    """
    路径规划节点
    
    使用各种算法规划路径
    """
    
    NODE_TYPE = "path_planning"
    DISPLAY_NAME = "路径规划"
    CATEGORY = "机器人仿真"
    DESCRIPTION = "规划从起点到终点的路径"
    
    @classmethod
    def get_inputs(cls) -> List[NodeInput]:
        return [
            NodeInput(
                name="start",
                type="list",
                required=True,
                default=[0.0, 0.0],
                description="起点 [x, y]"
            ),
            NodeInput(
                name="goal",
                type="list",
                required=True,
                default=[5.0, 5.0],
                description="终点 [x, y]"
            ),
            NodeInput(
                name="collision_fn",
                type="callable",
                required=True,
                description="碰撞检测函数"
            ),
            NodeInput(
                name="algorithm",
                type="string",
                required=False,
                default="astar",
                description="算法类型: astar, rrt, rrt_star, dwa"
            ),
            NodeInput(
                name="algorithm_params",
                type="dict",
                required=False,
                default={},
                description="算法参数字典"
            ),
        ]
    
    @classmethod
    def get_outputs(cls) -> List[NodeOutput]:
        return [
            NodeOutput(name="path", type="object", description="规划的路径"),
            NodeOutput(name="waypoints", type="list", description="路径点列表"),
            NodeOutput(name="path_length", type="number", description="路径长度"),
            NodeOutput(name="is_valid", type="boolean", description="路径是否有效"),
        ]
    
    def execute(self) -> Dict[str, Any]:
        start = tuple(self.get_input("start", [0.0, 0.0]))
        goal = tuple(self.get_input("goal", [5.0, 5.0]))
        collision_fn = self.get_input("collision_fn")
        algorithm = self.get_input("algorithm", "astar")
        params = self.get_input("algorithm_params", {})
        
        # 创建规划器
        if algorithm == "astar":
            resolution = params.get("resolution", 0.1)
            planner = AStarPlanner(collision_fn, resolution)
        elif algorithm == "rrt":
            planner = RRTPlanner(
                collision_fn,
                max_iter=params.get("max_iter", 5000),
                step_size=params.get("step_size", 0.2)
            )
        elif algorithm == "rrt_star":
            planner = RRTStarPlanner(
                collision_fn,
                max_iter=params.get("max_iter", 5000),
                step_size=params.get("step_size", 0.2),
                search_radius=params.get("search_radius", 0.5)
            )
        elif algorithm == "dwa":
            planner = DWAPlanner(
                collision_fn,
                max_linear_velocity=params.get("max_linear_velocity", 1.0),
                max_angular_velocity=params.get("max_angular_velocity", 2.0)
            )
        else:
            raise ValueError(f"未知的算法类型: {algorithm}")
        
        # 规划路径
        path = planner.plan(start, goal, **params)
        
        # 平滑路径（如果可能）
        if algorithm in ["astar", "rrt", "rrt_star"] and path.is_valid and len(path) > 2:
            path = path.smooth()
        
        # 计算路径长度
        path_length = 0.0
        if path.is_valid and len(path) > 1:
            for i in range(len(path) - 1):
                p1 = np.array(path.waypoints[i])
                p2 = np.array(path.waypoints[i + 1])
                path_length += np.linalg.norm(p2 - p1)
        
        return {
            "path": path,
            "waypoints": path.waypoints,
            "path_length": path_length,
            "is_valid": path.is_valid
        }


@register_node
class PathFollowingNode(RBCNode):
    """
    路径跟踪节点
    
    使用控制器跟踪路径
    """
    
    NODE_TYPE = "path_following"
    DISPLAY_NAME = "路径跟踪"
    CATEGORY = "机器人仿真"
    DESCRIPTION = "跟踪参考路径并生成控制命令"
    
    @classmethod
    def get_inputs(cls) -> List[NodeInput]:
        return [
            NodeInput(
                name="chassis",
                type="object",
                required=True,
                description="底盘对象"
            ),
            NodeInput(
                name="path",
                type="object",
                required=True,
                description="参考路径"
            ),
            NodeInput(
                name="controller_type",
                type="string",
                required=False,
                default="pure_pursuit",
                description="控制器类型: pure_pursuit, pid"
            ),
            NodeInput(
                name="dt",
                type="number",
                required=False,
                default=0.1,
                description="时间步长 (s)"
            ),
            NodeInput(
                name="simulation_time",
                type="number",
                required=False,
                default=30.0,
                description="仿真时间 (s)"
            ),
            NodeInput(
                name="controller_params",
                type="dict",
                required=False,
                default={},
                description="控制器参数字典"
            ),
        ]
    
    @classmethod
    def get_outputs(cls) -> List[NodeOutput]:
        return [
            NodeOutput(name="trajectory", type="list", description="实际轨迹 [(x, y, theta), ...]"),
            NodeOutput(name="control_commands", type="list", description="控制命令列表"),
            NodeOutput(name="final_pose", type="list", description="最终位姿 [x, y, theta]"),
            NodeOutput(name="tracking_error", type="number", description="跟踪误差"),
        ]
    
    def execute(self) -> Dict[str, Any]:
        chassis = self.get_input("chassis")
        path = self.get_input("path")
        controller_type = self.get_input("controller_type", "pure_pursuit")
        dt = self.get_input("dt", 0.1)
        simulation_time = self.get_input("simulation_time", 30.0)
        params = self.get_input("controller_params", {})
        
        if not isinstance(chassis, ChassisBase):
            raise ValueError("输入必须是底盘对象")
        
        if not isinstance(path, Path):
            raise ValueError("输入必须是路径对象")
        
        # 创建控制器
        if controller_type == "pure_pursuit":
            controller = PurePursuitController(
                lookahead_distance=params.get("lookahead_distance", 0.5),
                wheelbase=chassis.config.wheelbase,
                max_linear_velocity=chassis.config.max_linear_velocity
            )
        else:
            raise ValueError(f"未知的控制器类型: {controller_type}")
        
        # 仿真
        trajectory = []
        control_commands = []
        current_idx = 0
        num_steps = int(simulation_time / dt)
        
        for _ in range(num_steps):
            current_pose = (chassis.state.x, chassis.state.y, chassis.state.theta)
            trajectory.append(current_pose)
            
            # 检查是否到达终点
            if current_idx >= len(path) - 1:
                break
            
            # 计算控制命令
            v, omega, current_idx = controller.compute_control(
                current_pose, path, current_idx
            )
            
            # 根据底盘类型转换控制命令
            if chassis.chassis_type == ChassisType.DIFFERENTIAL_DRIVE:
                control = chassis.set_velocity(v, omega)
            elif chassis.chassis_type == ChassisType.ACKERMANN_STEERING:
                # 简化：使用 Pure Pursuit 的曲率计算转向角
                if abs(v) > 0.01:
                    R = v / (omega + 1e-6)
                    steering_angle = math.atan(chassis.config.wheelbase / R)
                else:
                    steering_angle = 0.0
                control = np.array([v, steering_angle])
            elif chassis.chassis_type == ChassisType.TRACKED_VEHICLE:
                W = chassis.config.track_width
                v_left = v - omega * W / 2
                v_right = v + omega * W / 2
                control = np.array([v_left, v_right])
            elif chassis.chassis_type == ChassisType.MECANUM_WHEEL:
                # 简化为前向运动
                control = np.array([v, 0, omega])
            else:
                control = np.array([v, omega])
            
            control_commands.append({
                'v': v,
                'omega': omega,
                'control': control.tolist()
            })
            
            # 应用控制
            chassis.apply_control(control, dt)
        
        # 计算跟踪误差
        tracking_error = 0.0
        if path.is_valid and len(path.waypoints) > 0:
            final_pos = np.array([chassis.state.x, chassis.state.y])
            goal_pos = np.array(path.waypoints[-1])
            tracking_error = np.linalg.norm(goal_pos - final_pos)
        
        return {
            "trajectory": trajectory,
            "control_commands": control_commands,
            "final_pose": [chassis.state.x, chassis.state.y, chassis.state.theta],
            "tracking_error": tracking_error
        }


@register_node
class TrajectoryVisualizerNode(RBCNode):
    """
    轨迹可视化节点
    
    生成轨迹可视化数据
    """
    
    NODE_TYPE = "trajectory_visualizer"
    DISPLAY_NAME = "轨迹可视化"
    CATEGORY = "机器人仿真"
    DESCRIPTION = "生成轨迹可视化数据"
    
    @classmethod
    def get_inputs(cls) -> List[NodeInput]:
        return [
            NodeInput(
                name="trajectory",
                type="list",
                required=True,
                description="轨迹点列表 [(x, y, theta), ...]"
            ),
            NodeInput(
                name="path",
                type="object",
                required=False,
                default=None,
                description="参考路径"
            ),
            NodeInput(
                name="chassis",
                type="object",
                required=False,
                default=None,
                description="底盘对象（用于显示当前状态）"
            ),
            NodeInput(
                name="show_velocity",
                type="boolean",
                required=False,
                default=True,
                description="是否显示速度矢量"
            ),
        ]
    
    @classmethod
    def get_outputs(cls) -> List[NodeOutput]:
        return [
            NodeOutput(name="visualization_data", type="dict", description="可视化数据字典"),
            NodeOutput(name="trajectory_mesh", type="dict", description="轨迹网格"),
            NodeOutput(name="path_mesh", type="dict", description="路径网格"),
        ]
    
    def execute(self) -> Dict[str, Any]:
        trajectory = self.get_input("trajectory", [])
        path = self.get_input("path", None)
        chassis = self.get_input("chassis", None)
        show_velocity = self.get_input("show_velocity", True)
        
        vis_data = {
            'trajectory': None,
            'path': None,
            'chassis': None,
            'velocity_arrow': None,
            'steering_indicator': None,
        }
        
        # 创建轨迹线
        if trajectory:
            vis_data['trajectory'] = create_trajectory_line(trajectory, Color.green())
        
        # 创建路径线
        if path and isinstance(path, Path):
            vis_data['path'] = create_path_mesh(path, Color.cyan())
        
        # 创建底盘网格
        if chassis and isinstance(chassis, ChassisBase):
            color = get_chassis_type_color(chassis.chassis_type)
            vis_data['chassis'] = create_chassis_mesh(chassis, color)
            
            # 速度矢量
            if show_velocity:
                vis_data['velocity_arrow'] = create_velocity_arrow(chassis.state)
            
            # 转向指示器（阿克曼底盘）
            steering_vis = create_steering_indicator(chassis)
            if steering_vis:
                vis_data['steering_indicator'] = steering_vis
        
        return {
            "visualization_data": vis_data,
            "trajectory_mesh": vis_data['trajectory'],
            "path_mesh": vis_data['path']
        }


@register_node
class ChassisSimulatorNode(RBCNode):
    """
    底盘仿真节点
    
    运行底盘物理仿真
    """
    
    NODE_TYPE = "chassis_simulator"
    DISPLAY_NAME = "底盘仿真器"
    CATEGORY = "机器人仿真"
    DESCRIPTION = "运行底盘物理仿真"
    
    @classmethod
    def get_inputs(cls) -> List[NodeInput]:
        return [
            NodeInput(
                name="chassis",
                type="object",
                required=True,
                description="底盘对象"
            ),
            NodeInput(
                name="control_sequence",
                type="list",
                required=True,
                description="控制命令序列 [{'v': ..., 'omega': ..., 'control': ...}, ...]"
            ),
            NodeInput(
                name="dt",
                type="number",
                required=False,
                default=0.1,
                description="时间步长 (s)"
            ),
        ]
    
    @classmethod
    def get_outputs(cls) -> List[NodeOutput]:
        return [
            NodeOutput(name="trajectory", type="list", description="仿真轨迹"),
            NodeOutput(name="final_state", type="object", description="最终状态"),
            NodeOutput(name="chassis", type="object", description="更新后的底盘对象"),
        ]
    
    def execute(self) -> Dict[str, Any]:
        chassis = self.get_input("chassis")
        control_sequence = self.get_input("control_sequence", [])
        dt = self.get_input("dt", 0.1)
        
        if not isinstance(chassis, ChassisBase):
            raise ValueError("输入必须是底盘对象")
        
        trajectory = []
        
        for cmd in control_sequence:
            control = np.array(cmd.get('control', [0, 0]))
            chassis.apply_control(control, dt)
            trajectory.append((chassis.state.x, chassis.state.y, chassis.state.theta))
        
        return {
            "trajectory": trajectory,
            "final_state": chassis.state,
            "chassis": chassis
        }


@register_node
class MultiChassisComparisonNode(RBCNode):
    """
    多底盘对比节点
    
    比较不同底盘类型的性能
    """
    
    NODE_TYPE = "multi_chassis_comparison"
    DISPLAY_NAME = "多底盘对比"
    CATEGORY = "机器人仿真"
    DESCRIPTION = "比较多底盘性能"
    
    @classmethod
    def get_inputs(cls) -> List[NodeInput]:
        return [
            NodeInput(
                name="chassis_list",
                type="list",
                required=True,
                description="底盘对象列表"
            ),
            NodeInput(
                name="trajectory_list",
                type="list",
                required=True,
                description="轨迹列表"
            ),
            NodeInput(
                name="path",
                type="object",
                required=True,
                description="参考路径"
            ),
        ]
    
    @classmethod
    def get_outputs(cls) -> List[NodeOutput]:
        return [
            NodeOutput(name="comparison_data", type="dict", description="对比数据"),
            NodeOutput(name="metrics", type="list", description="性能指标列表"),
        ]
    
    def execute(self) -> Dict[str, Any]:
        chassis_list = self.get_input("chassis_list", [])
        trajectory_list = self.get_input("trajectory_list", [])
        path = self.get_input("path")
        
        comparison = []
        
        for chassis, trajectory in zip(chassis_list, trajectory_list):
            if not isinstance(chassis, ChassisBase) or not trajectory:
                continue
            
            # 计算性能指标
            # 1. 路径长度
            path_length = 0.0
            for i in range(len(trajectory) - 1):
                p1 = np.array(trajectory[i][:2])
                p2 = np.array(trajectory[i + 1][:2])
                path_length += np.linalg.norm(p2 - p1)
            
            # 2. 跟踪误差
            if path and isinstance(path, Path) and path.is_valid:
                final_pos = np.array(trajectory[-1][:2])
                goal_pos = np.array(path.waypoints[-1])
                tracking_error = np.linalg.norm(goal_pos - final_pos)
            else:
                tracking_error = float('inf')
            
            # 3. 转向灵活性（累计转向角度）
            total_steering = 0.0
            for i in range(len(trajectory) - 1):
                theta_diff = abs(trajectory[i + 1][2] - trajectory[i][2])
                theta_diff = min(theta_diff, 2 * math.pi - theta_diff)
                total_steering += theta_diff
            
            metrics = {
                'chassis_type': chassis.chassis_type.value,
                'path_length': path_length,
                'tracking_error': tracking_error,
                'total_steering': total_steering,
                'final_position': [chassis.state.x, chassis.state.y],
            }
            
            comparison.append(metrics)
        
        return {
            "comparison_data": comparison,
            "metrics": comparison
        }
