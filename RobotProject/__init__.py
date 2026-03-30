# -*- coding: utf-8 -*-
"""
Robot Chassis Simulation Package

物理驱动的机器人底盘仿真
支持四种底盘类型：
- Differential Drive (差速底盘)
- Ackermann Steering (阿克曼底盘)
- Tracked Vehicle (履带底盘)
- Mecanum Wheel (麦克纳姆轮底盘)

提供路径规划、障碍物地图和可视化功能
"""

from .chassis import (
    ChassisType,
    ChassisConfig,
    ChassisState,
    DifferentialDrive,
    AckermannSteering,
    TrackedVehicle,
    MecanumWheel,
)

from .path_planning import (
    PathPlanner,
    AStarPlanner,
    RRTPlanner,
    RRTStarPlanner,
    DWAPlanner,
    PurePursuitController,
    Path,
)

from .rrt import (
    RRTPlanner as RRT,
    RRTStarPlanner as RRTStar,
    Path as RRTPath,
    Node as RRTNode,
)

from .map_system import (
    OccupancyGrid,
    Costmap,
    Obstacle,
    CircularObstacle,
    RectangularObstacle,
    PolygonObstacle,
    DynamicObstacle,
)

from .visualization import (
    Color,
    create_chassis_mesh,
    create_path_mesh,
    create_trajectory_line,
    create_velocity_arrow,
    get_chassis_type_color,
)

__all__ = [
    # Chassis
    'ChassisType',
    'ChassisConfig',
    'ChassisState',
    'DifferentialDrive',
    'AckermannSteering',
    'TrackedVehicle',
    'MecanumWheel',
    # Path Planning
    'PathPlanner',
    'AStarPlanner',
    'RRTPlanner',
    'RRTStarPlanner',
    'DWAPlanner',
    'PurePursuitController',
    'Path',
    # RRT Module
    'RRT',
    'RRTStar',
    'RRTPath',
    'RRTNode',
    # Map System
    'OccupancyGrid',
    'Costmap',
    'Obstacle',
    'CircularObstacle',
    'RectangularObstacle',
    'PolygonObstacle',
    'DynamicObstacle',
    # Visualization
    'Color',
    'create_chassis_mesh',
    'create_path_mesh',
    'create_trajectory_line',
    'create_velocity_arrow',
    'get_chassis_type_color',
]
