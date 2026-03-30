# -*- coding: utf-8 -*-
"""
Visualization Utilities for Robot Chassis Simulation

机器人底盘仿真的可视化工具
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import colorsys

from .chassis import ChassisBase, ChassisType, ChassisState
from .map_system import OccupancyGrid, Obstacle
from .path_planning import Path


@dataclass
class Color:
    """颜色类"""
    r: float
    g: float
    b: float
    a: float = 1.0
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)
    
    def to_rgb_tuple(self) -> Tuple[float, float, float]:
        return (self.r, self.g, self.b)
    
    @staticmethod
    def from_hsv(h: float, s: float, v: float, a: float = 1.0) -> 'Color':
        """从HSV创建颜色"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return Color(r, g, b, a)
    
    @staticmethod
    def red() -> 'Color':
        return Color(1.0, 0.2, 0.2)
    
    @staticmethod
    def green() -> 'Color':
        return Color(0.2, 1.0, 0.2)
    
    @staticmethod
    def blue() -> 'Color':
        return Color(0.2, 0.4, 1.0)
    
    @staticmethod
    def yellow() -> 'Color':
        return Color(1.0, 1.0, 0.2)
    
    @staticmethod
    def cyan() -> 'Color':
        return Color(0.2, 1.0, 1.0)
    
    @staticmethod
    def magenta() -> 'Color':
        return Color(1.0, 0.2, 1.0)
    
    @staticmethod
    def white() -> 'Color':
        return Color(1.0, 1.0, 1.0)
    
    @staticmethod
    def black() -> 'Color':
        return Color(0.0, 0.0, 0.0)
    
    @staticmethod
    def gray(level: float = 0.5) -> 'Color':
        return Color(level, level, level)


def create_grid_mesh(grid: OccupancyGrid) -> Dict[str, Any]:
    """
    创建栅格地图网格数据
    
    Args:
        grid: 栅格地图
        
    Returns:
        网格数据字典
    """
    occupied_cells = []
    
    for gy in range(grid.grid_height):
        for gx in range(grid.grid_width):
            if grid.data[gy, gx] > 0:  # 占用
                x, y = grid.grid_to_world(gx, gy)
                occupied_cells.append((x, y))
    
    return {
        'type': 'grid',
        'resolution': grid.resolution,
        'occupied_cells': occupied_cells,
        'origin': grid.origin,
        'width': grid.width,
        'height': grid.height,
    }


def create_obstacle_meshes(obstacles: List[Obstacle]) -> List[Dict[str, Any]]:
    """
    创建障碍物网格数据
    
    Args:
        obstacles: 障碍物列表
        
    Returns:
        网格数据列表
    """
    meshes = []
    
    for obstacle in obstacles:
        mesh_data = obstacle_to_mesh(obstacle)
        if mesh_data:
            meshes.append(mesh_data)
    
    return meshes


def obstacle_to_mesh(obstacle: Obstacle) -> Optional[Dict[str, Any]]:
    """将障碍物转换为网格数据"""
    from .map_system import CircularObstacle, RectangularObstacle, PolygonObstacle
    
    if isinstance(obstacle, CircularObstacle):
        return create_circle_mesh(
            obstacle.position[0], 
            obstacle.position[1], 
            obstacle.radius,
            Color.red()
        )
    
    elif isinstance(obstacle, RectangularObstacle):
        return create_rectangle_mesh(
            obstacle.position[0],
            obstacle.position[1],
            obstacle.width,
            obstacle.height,
            obstacle.angle,
            Color.red()
        )
    
    elif isinstance(obstacle, PolygonObstacle):
        return create_polygon_mesh(obstacle.vertices, Color.red())
    
    return None


def create_circle_mesh(cx: float, cy: float, radius: float, 
                       color: Color, segments: int = 32) -> Dict[str, Any]:
    """创建圆形网格"""
    vertices = []
    indices = []
    colors = []
    
    # 中心点
    vertices.append((cx, cy, 0.05))
    colors.append(color.to_rgb_tuple())
    
    # 圆周点
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        vertices.append((x, y, 0.05))
        colors.append(color.to_rgb_tuple())
    
    # 三角形索引
    for i in range(segments):
        indices.append(0)
        indices.append(1 + i)
        indices.append(1 + ((i + 1) % segments))
    
    return {
        'type': 'circle',
        'vertices': vertices,
        'indices': indices,
        'colors': colors,
        'center': (cx, cy),
        'radius': radius,
    }


def create_rectangle_mesh(cx: float, cy: float, width: float, height: float,
                          angle: float, color: Color) -> Dict[str, Any]:
    """创建矩形网格"""
    hw, hh = width / 2, height / 2
    
    # 局部坐标
    corners_local = [
        (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
    ]
    
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    vertices = []
    colors_list = []
    
    for lx, ly in corners_local:
        wx = cx + lx * cos_a - ly * sin_a
        wy = cy + lx * sin_a + ly * cos_a
        vertices.append((wx, wy, 0.05))
        colors_list.append(color.to_rgb_tuple())
    
    indices = [0, 1, 2, 0, 2, 3]
    
    return {
        'type': 'rectangle',
        'vertices': vertices,
        'indices': indices,
        'colors': colors_list,
        'center': (cx, cy),
        'width': width,
        'height': height,
        'angle': angle,
    }


def create_polygon_mesh(vertices_2d: List[Tuple[float, float]], 
                       color: Color) -> Dict[str, Any]:
    """创建多边形网格"""
    vertices = [(v[0], v[1], 0.05) for v in vertices_2d]
    colors_list = [color.to_rgb_tuple() for _ in vertices]
    
    # 简单的三角剖分（假设是凸多边形）
    indices = []
    for i in range(1, len(vertices) - 1):
        indices.extend([0, i, i + 1])
    
    return {
        'type': 'polygon',
        'vertices': vertices,
        'indices': indices,
        'colors': colors_list,
    }


def create_chassis_mesh(chassis: ChassisBase, color: Optional[Color] = None) -> Dict[str, Any]:
    """
    创建底盘网格数据
    
    Args:
        chassis: 底盘对象
        color: 颜色
        
    Returns:
        网格数据字典
    """
    if color is None:
        color = Color.blue()
    
    state = chassis.state
    config = chassis.config
    
    # 车体尺寸
    L, W = config.body_length, config.body_width
    
    # 创建车体矩形
    chassis_color = color
    body_mesh = create_robot_body_mesh(state, L, W, chassis_color)
    
    # 创建轮子
    wheel_meshes = create_wheel_meshes(chassis)
    
    # 创建方向指示器
    direction_mesh = create_direction_indicator(state, L, color)
    
    return {
        'type': 'chassis',
        'body': body_mesh,
        'wheels': wheel_meshes,
        'direction': direction_mesh,
        'state': state,
        'chassis_type': chassis.chassis_type,
    }


def create_robot_body_mesh(state: ChassisState, length: float, width: float,
                           color: Color) -> Dict[str, Any]:
    """创建机器人车体网格"""
    x, y, theta = state.x, state.y, state.theta
    
    # 车体中心偏移（车头朝前）
    center_offset = length / 4
    
    hw, hl = width / 2, length / 2
    
    # 局部坐标（车体坐标系）
    corners_local = [
        (hl + center_offset, -hw),   # 右前
        (hl + center_offset, hw),    # 左前
        (-hl + center_offset, hw),   # 左后
        (-hl + center_offset, -hw),  # 右后
    ]
    
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    vertices = []
    colors_list = []
    
    for lx, ly in corners_local:
        # 旋转并平移到世界坐标
        wx = x + lx * cos_t - ly * sin_t
        wy = y + lx * sin_t + ly * cos_t
        vertices.append((wx, wy, 0.1))
        colors_list.append(color.to_rgb_tuple())
    
    indices = [0, 1, 2, 0, 2, 3]
    
    return {
        'vertices': vertices,
        'indices': indices,
        'colors': colors_list,
    }


def create_wheel_meshes(chassis: ChassisBase) -> List[Dict[str, Any]]:
    """创建轮子网格"""
    state = chassis.state
    wheel_positions = chassis.get_wheel_positions()
    wheel_radius = chassis.config.wheel_radius
    wheel_width = 0.05
    
    wheel_meshes = []
    
    for i, wheel_pos_local in enumerate(wheel_positions):
        # 转换到世界坐标
        wheel_pos_world = chassis.local_to_world(wheel_pos_local)
        
        # 轮子颜色
        wheel_color = Color.gray(0.3)
        
        # 如果是阿克曼底盘且是前轮，添加转向
        if (chassis.chassis_type == ChassisType.ACKERMANN_STEERING and 
            i < 2 and state.wheel_angles is not None):
            wheel_angle = state.theta + state.wheel_angles[i]
        else:
            wheel_angle = state.theta
        
        mesh = create_wheel_mesh(
            wheel_pos_world[0], wheel_pos_world[1],
            wheel_radius, wheel_width, wheel_angle,
            wheel_color
        )
        wheel_meshes.append(mesh)
    
    return wheel_meshes


def create_wheel_mesh(cx: float, cy: float, radius: float, width: float,
                      angle: float, color: Color) -> Dict[str, Any]:
    """创建单个轮子网格"""
    # 简化为矩形
    hw = width / 2
    hr = radius
    
    corners_local = [
        (hr, -hw), (hr, hw), (-hr, hw), (-hr, -hw)
    ]
    
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    vertices = []
    colors_list = []
    
    for lx, ly in corners_local:
        wx = cx + lx * cos_a - ly * sin_a
        wy = cy + lx * sin_a + ly * cos_a
        vertices.append((wx, wy, 0.05))
        colors_list.append(color.to_rgb_tuple())
    
    indices = [0, 1, 2, 0, 2, 3]
    
    return {
        'vertices': vertices,
        'indices': indices,
        'colors': colors_list,
    }


def create_direction_indicator(state: ChassisState, length: float,
                                color: Color) -> Dict[str, Any]:
    """创建方向指示器（箭头）"""
    x, y, theta = state.x, state.y, state.theta
    
    # 箭头起点（车体中心）
    start_x = x
    start_y = y
    
    # 箭头终点
    arrow_len = length * 0.8
    end_x = x + arrow_len * math.cos(theta)
    end_y = y + arrow_len * math.sin(theta)
    
    # 箭头翼
    arrow_wing_len = 0.15
    wing_angle = math.pi / 6
    
    left_wing_x = end_x - arrow_wing_len * math.cos(theta - wing_angle)
    left_wing_y = end_y - arrow_wing_len * math.sin(theta - wing_angle)
    
    right_wing_x = end_x - arrow_wing_len * math.cos(theta + wing_angle)
    right_wing_y = end_y - arrow_wing_len * math.sin(theta + wing_angle)
    
    vertices = [
        (start_x, start_y, 0.15),
        (end_x, end_y, 0.15),
        (left_wing_x, left_wing_y, 0.15),
        (right_wing_x, right_wing_y, 0.15),
    ]
    
    colors_list = [
        color.to_rgb_tuple(),
        color.to_rgb_tuple(),
        color.to_rgb_tuple(),
        color.to_rgb_tuple(),
    ]
    
    indices = [0, 1, 2, 0, 1, 3]
    
    return {
        'vertices': vertices,
        'indices': indices,
        'colors': colors_list,
    }


def create_trajectory_line(trajectory: List[Tuple[float, float, float]],
                           color: Color = None) -> Dict[str, Any]:
    """
    创建轨迹线
    
    Args:
        trajectory: 轨迹点列表 [(x, y, theta), ...]
        color: 颜色
        
    Returns:
        轨迹线数据
    """
    if color is None:
        color = Color.green()
    
    if len(trajectory) < 2:
        return None
    
    vertices = []
    colors_list = []
    
    for x, y, theta in trajectory:
        vertices.append((x, y, 0.02))
        colors_list.append(color.to_rgb_tuple())
    
    indices = list(range(len(vertices)))
    
    return {
        'type': 'line_strip',
        'vertices': vertices,
        'indices': indices,
        'colors': colors_list,
        'width': 2.0,
    }


def create_path_mesh(path: Path, color: Color = None) -> Dict[str, Any]:
    """
    创建路径网格
    
    Args:
        path: 路径对象
        color: 颜色
        
    Returns:
        路径网格数据
    """
    if color is None:
        color = Color.cyan()
    
    if not path.is_valid or len(path) < 2:
        return None
    
    vertices = []
    colors_list = []
    
    for x, y in path.waypoints:
        vertices.append((x, y, 0.03))
        colors_list.append(color.to_rgb_tuple())
    
    return {
        'type': 'line_strip',
        'vertices': vertices,
        'indices': list(range(len(vertices))),
        'colors': colors_list,
        'width': 3.0,
    }


def create_velocity_arrow(state: ChassisState, scale: float = 1.0) -> Dict[str, Any]:
    """
    创建速度矢量箭头
    
    Args:
        state: 底盘状态
        scale: 缩放因子
        
    Returns:
        箭头网格数据
    """
    x, y = state.x, state.y
    vx, vy = state.vx, state.vy
    
    if abs(vx) < 0.01 and abs(vy) < 0.01:
        return None
    
    # 箭头终点
    end_x = x + vx * scale
    end_y = y + vy * scale
    
    # 箭头翼
    v_len = math.sqrt(vx**2 + vy**2)
    angle = math.atan2(vy, vx)
    
    arrow_wing_len = min(0.2, v_len * scale * 0.3)
    wing_angle = math.pi / 6
    
    left_wing_x = end_x - arrow_wing_len * math.cos(angle - wing_angle)
    left_wing_y = end_y - arrow_wing_len * math.sin(angle - wing_angle)
    
    right_wing_x = end_x - arrow_wing_len * math.cos(angle + wing_angle)
    right_wing_y = end_y - arrow_wing_len * math.sin(angle + wing_angle)
    
    vertices = [
        (x, y, 0.2),
        (end_x, end_y, 0.2),
        (left_wing_x, left_wing_y, 0.2),
        (right_wing_x, right_wing_y, 0.2),
    ]
    
    color = Color.yellow()
    colors_list = [color.to_rgb_tuple()] * 4
    
    return {
        'type': 'arrow',
        'vertices': vertices,
        'indices': [0, 1, 0, 2, 0, 3],
        'colors': colors_list,
        'width': 2.0,
    }


def create_steering_indicator(chassis: ChassisBase) -> Optional[Dict[str, Any]]:
    """
    创建转向角指示器（仅阿克曼底盘）
    
    Returns:
        指示器网格数据
    """
    if chassis.chassis_type != ChassisType.ACKERMANN_STEERING:
        return None
    
    state = chassis.state
    if state.wheel_angles is None:
        return None
    
    # 绘制转向弧线
    turning_radius = chassis.get_turning_radius()
    if math.isinf(turning_radius):
        return None
    
    x, y, theta = state.x, state.y, state.theta
    
    # 转弯中心
    center_x = x - turning_radius * math.sin(theta)
    center_y = y + turning_radius * math.cos(theta)
    
    # 绘制弧线
    vertices = []
    colors_list = []
    
    num_segments = 32
    arc_angle = math.pi / 3  # 60度弧线
    
    for i in range(num_segments + 1):
        angle = theta - arc_angle/2 + arc_angle * i / num_segments
        px = center_x + abs(turning_radius) * math.cos(angle)
        py = center_y + abs(turning_radius) * math.sin(angle)
        vertices.append((px, py, 0.05))
        colors_list.append(Color.magenta().to_rgb_tuple())
    
    return {
        'type': 'line_strip',
        'vertices': vertices,
        'indices': list(range(len(vertices))),
        'colors': colors_list,
        'width': 2.0,
    }


def generate_heatmap_data(grid: OccupancyGrid, 
                          value_fn: callable) -> np.ndarray:
    """
    生成热力图数据
    
    Args:
        grid: 栅格地图
        value_fn: 值函数，输入 (x, y) 返回值
        
    Returns:
        热力图数据数组
    """
    heatmap = np.zeros((grid.grid_height, grid.grid_width))
    
    for gy in range(grid.grid_height):
        for gx in range(grid.grid_width):
            x, y = grid.grid_to_world(gx, gy)
            heatmap[gy, gx] = value_fn(x, y)
    
    return heatmap


def get_chassis_type_color(chassis_type: ChassisType) -> Color:
    """根据底盘类型获取颜色"""
    colors = {
        ChassisType.DIFFERENTIAL_DRIVE: Color.blue(),
        ChassisType.ACKERMANN_STEERING: Color.green(),
        ChassisType.TRACKED_VEHICLE: Color.yellow(),
        ChassisType.MECANUM_WHEEL: Color.magenta(),
    }
    return colors.get(chassis_type, Color.white())
