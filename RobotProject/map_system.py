# -*- coding: utf-8 -*-
"""
Map and Obstacle System

地图和障碍物系统
支持栅格地图、成本地图和各种障碍物类型
"""

import numpy as np
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class CellState(Enum):
    """栅格单元状态"""
    FREE = 0      # 空闲
    OCCUPIED = 1  # 占用
    UNKNOWN = -1  # 未知


@dataclass
class Obstacle:
    """障碍物基类"""
    id: int = 0
    position: Tuple[float, float] = (0.0, 0.0)
    
    @abstractmethod
    def contains(self, x: float, y: float) -> bool:
        """检查点是否在障碍物内"""
        pass
    
    @abstractmethod
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """获取包围盒 (min_x, max_x, min_y, max_y)"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        pass


@dataclass
class CircularObstacle(Obstacle):
    """圆形障碍物"""
    radius: float = 0.5
    
    def contains(self, x: float, y: float) -> bool:
        dx = x - self.position[0]
        dy = y - self.position[1]
        return dx**2 + dy**2 <= self.radius**2
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        x, y = self.position
        r = self.radius
        return (x - r, x + r, y - r, y + r)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'circular',
            'id': self.id,
            'position': self.position,
            'radius': self.radius
        }


@dataclass
class RectangularObstacle(Obstacle):
    """矩形障碍物"""
    width: float = 1.0
    height: float = 1.0
    angle: float = 0.0  # 旋转角度 (rad)
    
    def contains(self, x: float, y: float) -> bool:
        # 将点转换到障碍物的局部坐标系
        dx = x - self.position[0]
        dy = y - self.position[1]
        
        cos_a = math.cos(-self.angle)
        sin_a = math.sin(-self.angle)
        
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        half_w = self.width / 2
        half_h = self.height / 2
        
        return -half_w <= local_x <= half_w and -half_h <= local_y <= half_h
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        # 简化为包含旋转矩形的包围盒
        x, y = self.position
        max_dim = max(self.width, self.height) / 2
        diagonal = max_dim * math.sqrt(2)
        return (x - diagonal, x + diagonal, y - diagonal, y + diagonal)
    
    def get_corners(self) -> List[Tuple[float, float]]:
        """获取四个角点（世界坐标）"""
        x, y = self.position
        hw, hh = self.width / 2, self.height / 2
        
        corners_local = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
        ]
        
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        
        corners_world = []
        for cx, cy in corners_local:
            wx = x + cx * cos_a - cy * sin_a
            wy = y + cx * sin_a + cy * cos_a
            corners_world.append((wx, wy))
        
        return corners_world
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'rectangular',
            'id': self.id,
            'position': self.position,
            'width': self.width,
            'height': self.height,
            'angle': self.angle
        }


@dataclass
class PolygonObstacle(Obstacle):
    """多边形障碍物"""
    vertices: List[Tuple[float, float]] = field(default_factory=list)
    
    def contains(self, x: float, y: float) -> bool:
        """使用射线法判断点是否在多边形内"""
        n = len(self.vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        return (min(xs), max(xs), min(ys), max(ys))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'polygon',
            'id': self.id,
            'vertices': self.vertices
        }


@dataclass
class DynamicObstacle(CircularObstacle):
    """动态障碍物"""
    velocity: Tuple[float, float] = (0.0, 0.0)
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    
    def update_position(self, t: float):
        """根据时间更新位置"""
        if len(self.trajectory) >= 2:
            # 查找当前时间所在的轨迹段
            for i in range(len(self.trajectory) - 1):
                t1, x1, y1 = self.trajectory[i]
                t2, x2, y2 = self.trajectory[i + 1]
                if t1 <= t <= t2:
                    # 线性插值
                    if t2 - t1 > 0:
                        ratio = (t - t1) / (t2 - t1)
                        self.position = (
                            x1 + ratio * (x2 - x1),
                            y1 + ratio * (y2 - y1)
                        )
                    break
        else:
            # 匀速运动
            self.position = (
                self.position[0] + self.velocity[0] * t,
                self.position[1] + self.velocity[1] * t
            )
    
    def predict_position(self, t: float) -> Tuple[float, float]:
        """预测未来位置"""
        if len(self.trajectory) >= 2:
            # 使用轨迹预测
            for i in range(len(self.trajectory) - 1):
                t1, x1, y1 = self.trajectory[i]
                t2, x2, y2 = self.trajectory[i + 1]
                if t1 <= t <= t2:
                    ratio = (t - t1) / (t2 - t1)
                    return (
                        x1 + ratio * (x2 - x1),
                        y1 + ratio * (y2 - y1)
                    )
            # 如果超出轨迹范围，使用最后一个点
            return (self.trajectory[-1][1], self.trajectory[-1][2])
        else:
            # 匀速预测
            return (
                self.position[0] + self.velocity[0] * t,
                self.position[1] + self.velocity[1] * t
            )
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data['type'] = 'dynamic'
        data['velocity'] = self.velocity
        data['trajectory'] = self.trajectory
        return data


class OccupancyGrid:
    """
    占用栅格地图
    
    2D 栅格地图，表示环境的占用情况
    """
    
    def __init__(self, width: float = 10.0, height: float = 10.0,
                 resolution: float = 0.1, origin: Tuple[float, float] = (-5, -5)):
        """
        初始化栅格地图
        
        Args:
            width: 地图宽度 (m)
            height: 地图高度 (m)
            resolution: 栅格分辨率 (m/cell)
            origin: 地图原点 (左下角)
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # 初始化栅格数据 (0 = free, 1 = occupied, -1 = unknown)
        self.data = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        
        # 障碍物列表
        self.obstacles: List[Obstacle] = []
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return (gx, gy)
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标（栅格中心）"""
        x = self.origin[0] + (gx + 0.5) * self.resolution
        y = self.origin[1] + (gy + 0.5) * self.resolution
        return (x, y)
    
    def is_in_bounds(self, gx: int, gy: int) -> bool:
        """检查栅格坐标是否在范围内"""
        return 0 <= gx < self.grid_width and 0 <= gy < self.grid_height
    
    def is_in_bounds_world(self, x: float, y: float) -> bool:
        """检查世界坐标是否在范围内"""
        return (self.origin[0] <= x < self.origin[0] + self.width and
                self.origin[1] <= y < self.origin[1] + self.height)
    
    def get_cell(self, gx: int, gy: int) -> CellState:
        """获取栅格状态"""
        if not self.is_in_bounds(gx, gy):
            return CellState.UNKNOWN
        value = self.data[gy, gx]
        if value == 0:
            return CellState.FREE
        elif value >= 1:
            return CellState.OCCUPIED
        else:
            return CellState.UNKNOWN
    
    def set_cell(self, gx: int, gy: int, state: CellState):
        """设置栅格状态"""
        if self.is_in_bounds(gx, gy):
            self.data[gy, gx] = state.value
    
    def set_obstacle(self, obstacle: Obstacle):
        """添加障碍物"""
        self.obstacles.append(obstacle)
        self._rasterize_obstacle(obstacle)
    
    def remove_obstacle(self, obstacle_id: int):
        """移除障碍物"""
        self.obstacles = [o for o in self.obstacles if o.id != obstacle_id]
        self._rebuild_grid()
    
    def _rasterize_obstacle(self, obstacle: Obstacle):
        """栅格化障碍物"""
        bbox = obstacle.get_bounding_box()
        min_gx, max_gx, min_gy, max_gy = self._bbox_to_grid(bbox)
        
        for gx in range(max(0, min_gx), min(self.grid_width, max_gx + 1)):
            for gy in range(max(0, min_gy), min(self.grid_height, max_gy + 1)):
                x, y = self.grid_to_world(gx, gy)
                if obstacle.contains(x, y):
                    self.data[gy, gx] = CellState.OCCUPIED.value
    
    def _bbox_to_grid(self, bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """包围盒转栅格坐标"""
        min_x, max_x, min_y, max_y = bbox
        min_gx, min_gy = self.world_to_grid(min_x, min_y)
        max_gx, max_gy = self.world_to_grid(max_x, max_y)
        return (min_gx, max_gx, min_gy, max_gy)
    
    def _rebuild_grid(self):
        """重建栅格地图"""
        self.data.fill(0)
        for obstacle in self.obstacles:
            self._rasterize_obstacle(obstacle)
    
    def check_collision(self, x: float, y: float, radius: float = 0.0) -> bool:
        """
        检查点是否与障碍物碰撞
        
        Args:
            x, y: 点坐标
            radius: 碰撞半径
            
        Returns:
            是否碰撞
        """
        if radius == 0:
            # 简单点检查
            gx, gy = self.world_to_grid(x, y)
            if not self.is_in_bounds(gx, gy):
                return True  # 超出边界视为碰撞
            return self.data[gy, gx] == CellState.OCCUPIED.value
        else:
            # 圆形碰撞检查
            for obstacle in self.obstacles:
                if isinstance(obstacle, CircularObstacle):
                    dx = x - obstacle.position[0]
                    dy = y - obstacle.position[1]
                    if dx**2 + dy**2 <= (obstacle.radius + radius)**2:
                        return True
                elif isinstance(obstacle, (RectangularObstacle, PolygonObstacle)):
                    # 采样检查
                    num_samples = max(8, int(2 * math.pi * radius / self.resolution))
                    for i in range(num_samples):
                        angle = 2 * math.pi * i / num_samples
                        cx = x + radius * math.cos(angle)
                        cy = y + radius * math.sin(angle)
                        if obstacle.contains(cx, cy):
                            return True
            return False
    
    def get_distance_to_obstacle(self, x: float, y: float) -> float:
        """
        获取到最近障碍物的距离
        
        Returns:
            距离（如果没有障碍物则返回 inf）
        """
        min_dist = float('inf')
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircularObstacle):
                dx = x - obstacle.position[0]
                dy = y - obstacle.position[1]
                dist = max(0, math.sqrt(dx**2 + dy**2) - obstacle.radius)
            elif isinstance(obstacle, RectangularObstacle):
                dist = self._dist_to_rect(x, y, obstacle)
            else:
                # 默认使用中心点距离
                dx = x - obstacle.position[0]
                dy = y - obstacle.position[1]
                dist = math.sqrt(dx**2 + dy**2)
            
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _dist_to_rect(self, x: float, y: float, rect: RectangularObstacle) -> float:
        """计算点到矩形的距离"""
        # 简化为到中心点的距离减去半对角线长度
        dx = x - rect.position[0]
        dy = y - rect.position[1]
        center_dist = math.sqrt(dx**2 + dy**2)
        half_diagonal = math.sqrt(rect.width**2 + rect.height**2) / 2
        return max(0, center_dist - half_diagonal)
    
    def update_dynamic_obstacles(self, t: float):
        """更新动态障碍物位置"""
        for obstacle in self.obstacles:
            if isinstance(obstacle, DynamicObstacle):
                obstacle.update_position(t)
        self._rebuild_grid()
    
    def to_numpy(self) -> np.ndarray:
        """导出为 numpy 数组"""
        return self.data.copy()
    
    @staticmethod
    def from_numpy(data: np.ndarray, resolution: float = 0.1,
                   origin: Tuple[float, float] = (0, 0)) -> 'OccupancyGrid':
        """从 numpy 数组创建栅格地图"""
        grid_height, grid_width = data.shape
        width = grid_width * resolution
        height = grid_height * resolution
        
        grid = OccupancyGrid(width, height, resolution, origin)
        grid.data = data.astype(np.int8)
        return grid


class Costmap:
    """
    成本地图
    
    多层成本地图，用于路径规划
    """
    
    def __init__(self, occupancy_grid: OccupancyGrid):
        """
        初始化成本地图
        
        Args:
            occupancy_grid: 基础栅格地图
        """
        self.grid = occupancy_grid
        
        # 各层成本
        self.static_layer = np.zeros((occupancy_grid.grid_height, 
                                       occupancy_grid.grid_width), dtype=np.float32)
        self.dynamic_layer = np.zeros((occupancy_grid.grid_height, 
                                        occupancy_grid.grid_width), dtype=np.float32)
        self.inflation_layer = np.zeros((occupancy_grid.grid_height, 
                                          occupancy_grid.grid_width), dtype=np.float32)
        
        # 最终成本
        self.cost = np.zeros((occupancy_grid.grid_height, 
                              occupancy_grid.grid_width), dtype=np.float32)
        
        # 膨胀参数
        self.inflation_radius = 0.5  # 膨胀半径 (m)
        self.inflation_cost = 50.0   # 膨胀成本
        
        self._update_layers()
    
    def _update_layers(self):
        """更新各层成本"""
        # 静态层：障碍物
        self.static_layer = (self.grid.data > 0).astype(np.float32) * 100.0
        
        # 动态层：初始化为0
        self.dynamic_layer.fill(0)
        
        # 计算膨胀层
        self._compute_inflation_layer()
        
        # 合并各层
        self.cost = self.static_layer + self.dynamic_layer + self.inflation_layer
    
    def _compute_inflation_layer(self):
        """计算膨胀层"""
        self.inflation_layer.fill(0)
        
        inflation_cells = int(self.inflation_radius / self.grid.resolution)
        
        for gy in range(self.grid.grid_height):
            for gx in range(self.grid.grid_width):
                if self.grid.data[gy, gx] > 0:  # 障碍物
                    # 膨胀
                    for dy in range(-inflation_cells, inflation_cells + 1):
                        for dx in range(-inflation_cells, inflation_cells + 1):
                            ngx, ngy = gx + dx, gy + dy
                            if self.grid.is_in_bounds(ngx, ngy):
                                dist = math.sqrt(dx**2 + dy**2) * self.grid.resolution
                                if dist <= self.inflation_radius:
                                    # 成本随距离衰减
                                    cost = self.inflation_cost * (1 - dist / self.inflation_radius)
                                    self.inflation_layer[ngy, ngx] = max(
                                        self.inflation_layer[ngy, ngx], cost)
    
    def update_dynamic_obstacles(self, obstacles: List[DynamicObstacle], t: float):
        """更新动态障碍物层"""
        self.dynamic_layer.fill(0)
        
        for obstacle in obstacles:
            # 预测未来位置
            for dt in [0, 0.5, 1.0]:  # 多个时间点
                future_pos = obstacle.predict_position(t + dt)
                gx, gy = self.grid.world_to_grid(future_pos[0], future_pos[1])
                
                if self.grid.is_in_bounds(gx, gy):
                    # 添加成本（时间越远，成本越低）
                    cost = 80.0 / (1 + dt)
                    radius_cells = int((obstacle.radius + 0.2) / self.grid.resolution)
                    
                    for dy in range(-radius_cells, radius_cells + 1):
                        for dx in range(-radius_cells, radius_cells + 1):
                            ngx, ngy = gx + dx, gy + dy
                            if self.grid.is_in_bounds(ngx, ngy):
                                dist = math.sqrt(dx**2 + dy**2) * self.grid.resolution
                                if dist <= obstacle.radius + 0.2:
                                    self.dynamic_layer[ngy, ngx] = max(
                                        self.dynamic_layer[ngy, ngx], cost)
        
        # 更新总成本
        self.cost = self.static_layer + self.dynamic_layer + self.inflation_layer
    
    def get_cost(self, x: float, y: float) -> float:
        """获取指定位置的成本"""
        gx, gy = self.grid.world_to_grid(x, y)
        if not self.grid.is_in_bounds(gx, gy):
            return float('inf')
        return self.cost[gy, gx]
    
    def set_inflation_params(self, radius: float, cost: float):
        """设置膨胀参数"""
        self.inflation_radius = radius
        self.inflation_cost = cost
        self._compute_inflation_layer()
        self.cost = self.static_layer + self.dynamic_layer + self.inflation_layer
