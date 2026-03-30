# -*- coding: utf-8 -*-
"""
RoboCute Robot Engineering - 机器人工程模块

提供完整的机器人仿真功能，包括:
- 多种底盘类型支持(差速/阿克曼/履带/麦轮)
- 路径规划算法集成
- 地图编辑器

使用 World API 进行实体管理，无需渲染接口。
"""

from .chassis import (
    ChassisType,
    ChassisBase,
    DifferentialChassis,
    AckermannChassis,
    TrackedChassis,
    MecanumChassis,
)
from .path_planner import (
    PathPlanner,
    AStarPlanner,
    RRTPlanner,
    PathPoint,
)
from .map_editor import (
    MapEditor,
    MapCell,
    CellType,
)
from .robot import (
    Robot,
    RobotConfig,
)

__all__ = [
    # Chassis
    'ChassisType',
    'ChassisBase',
    'DifferentialChassis',
    'AckermannChassis',
    'TrackedChassis',
    'MecanumChassis',
    # Path Planner
    'PathPlanner',
    'AStarPlanner',
    'RRTPlanner',
    'PathPoint',
    # Map Editor
    'MapEditor',
    'MapCell',
    'CellType',
    # Robot
    'Robot',
    'RobotConfig',
]

__version__ = "1.0.0"
