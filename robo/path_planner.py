# -*- coding: utf-8 -*-
"""
路径规划模块 - 提供多种路径规划算法

支持的算法:
- A* (A-Star): 最优路径搜索
- RRT (Rapidly-exploring Random Tree): 快速随机树
- Dijkstra: 最短路径
"""

from __future__ import annotations
import math
import random
import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Callable
import numpy as np


@dataclass
class PathPoint:
    """路径点"""
    x: float
    y: float
    theta: float = 0.0  # 朝向 (可选)
    
    def distance_to(self, other: PathPoint) -> float:
        """计算到另一个点的欧氏距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def to_tuple(self) -> Tuple[float, float]:
        """转换为元组"""
        return (self.x, self.y)
    
    @staticmethod
    def from_tuple(t: Tuple[float, float]) -> PathPoint:
        """从元组创建"""
        return PathPoint(t[0], t[1])


@dataclass
class Path:
    """路径"""
    points: List[PathPoint] = field(default_factory=list)
    cost: float = 0.0
    
    def __len__(self) -> int:
        return len(self.points)
    
    def is_empty(self) -> bool:
        return len(self.points) == 0
    
    def total_length(self) -> float:
        """计算路径总长度"""
        length = 0.0
        for i in range(len(self.points) - 1):
            length += self.points[i].distance_to(self.points[i + 1])
        return length
    
    def smooth(self, weight_data: float = 0.5, weight_smooth: float = 0.1, 
               tolerance: float = 0.00001) -> Path:
        """
        路径平滑处理 (梯度下降法)
        
        Args:
            weight_data: 数据权重
            weight_smooth: 平滑权重
            tolerance: 收敛容差
            
        Returns:
            平滑后的路径
        """
        if len(self.points) < 3:
            return Path(self.points.copy(), self.cost)
        
        new_points = [PathPoint(p.x, p.y, p.theta) for p in self.points]
        
        change = tolerance
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(new_points) - 1):
                for coord in ['x', 'y']:
                    old_val = getattr(new_points[i], coord)
                    
                    # 梯度下降更新
                    data_term = weight_data * (getattr(self.points[i], coord) - old_val)
                    smooth_term = weight_smooth * (
                        getattr(new_points[i - 1], coord) + 
                        getattr(new_points[i + 1], coord) - 2 * old_val
                    )
                    
                    new_val = old_val + data_term + smooth_term
                    setattr(new_points[i], coord, new_val)
                    
                    change += abs(new_val - old_val)
        
        return Path(new_points, self.cost)


class PathPlanner(ABC):
    """路径规划器基类"""
    
    def __init__(self, map_data: Optional[np.ndarray] = None, 
                 resolution: float = 0.1,
                 origin: Tuple[float, float] = (0.0, 0.0)):
        """
        初始化路径规划器
        
        Args:
            map_data: 地图数据 (2D numpy 数组，0=空闲，1=障碍)
            resolution: 地图分辨率 (m/cell)
            origin: 地图原点 (m)
        """
        self._map = map_data
        self._resolution = resolution
        self._origin = origin
        self._collision_checker: Optional[Callable[[float, float], bool]] = None
    
    def set_map(self, map_data: np.ndarray, resolution: float = 0.1,
                origin: Tuple[float, float] = (0.0, 0.0)):
        """设置地图"""
        self._map = map_data
        self._resolution = resolution
        self._origin = origin
    
    def set_collision_checker(self, checker: Callable[[float, float], bool]):
        """
        设置碰撞检测函数
        
        Args:
            checker: 函数 (x, y) -> bool，True 表示碰撞
        """
        self._collision_checker = checker
    
    def is_collision(self, x: float, y: float) -> bool:
        """
        检查点是否在碰撞状态
        
        Args:
            x, y: 世界坐标 (m)
            
        Returns:
            True 表示碰撞
        """
        if self._collision_checker:
            return self._collision_checker(x, y)
        
        if self._map is None:
            return False
        
        # 转换到地图坐标
        mx = int((x - self._origin[0]) / self._resolution)
        my = int((y - self._origin[1]) / self._resolution)
        
        # 检查边界
        if mx < 0 or mx >= self._map.shape[1] or my < 0 or my >= self._map.shape[0]:
            return True  # 地图外视为碰撞
        
        return self._map[my, mx] > 0
    
    def is_line_collision(self, p1: PathPoint, p2: PathPoint, 
                          step_size: float = 0.05) -> bool:
        """
        检查线段是否与障碍物碰撞 (Bresenham 算法简化版)
        
        Args:
            p1, p2: 线段端点
            step_size: 检查步长
            
        Returns:
            True 表示碰撞
        """
        dist = p1.distance_to(p2)
        if dist < 1e-6:
            return self.is_collision(p1.x, p1.y)
        
        steps = max(1, int(dist / step_size))
        for i in range(steps + 1):
            t = i / steps
            x = p1.x + t * (p2.x - p1.x)
            y = p1.y + t * (p2.y - p1.y)
            if self.is_collision(x, y):
                return True
        
        return False
    
    @abstractmethod
    def plan(self, start: PathPoint, goal: PathPoint) -> Optional[Path]:
        """
        规划路径
        
        Args:
            start: 起点
            goal: 终点
            
        Returns:
            规划的路径，失败返回 None
        """
        pass


class AStarPlanner(PathPlanner):
    """
    A* 路径规划器
    
    使用启发式搜索找到最优路径。
    """
    
    def __init__(self, map_data: Optional[np.ndarray] = None,
                 resolution: float = 0.1,
                 origin: Tuple[float, float] = (0.0, 0.0),
                 diagonal_movement: bool = True):
        super().__init__(map_data, resolution, origin)
        self._diagonal = diagonal_movement
    
    def _heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """启发式函数 (欧氏距离)"""
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        
        if self._diagonal:
            # 对角距离
            return math.sqrt(dx * dx + dy * dy) * self._resolution
        else:
            # 曼哈顿距离
            return (dx + dy) * self._resolution
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], float]]:
        """获取邻居节点"""
        neighbors = []
        x, y = pos
        
        # 四方向
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        # 八方向 (对角)
        if self._diagonal:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # 检查边界
            if self._map is not None:
                if nx < 0 or nx >= self._map.shape[1] or ny < 0 or ny >= self._map.shape[0]:
                    continue
                if self._map[ny, nx] > 0:
                    continue
            
            # 计算代价
            cost = math.sqrt(dx * dx + dy * dy) * self._resolution
            neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def plan(self, start: PathPoint, goal: PathPoint) -> Optional[Path]:
        """A* 路径规划"""
        if self._map is None:
            # 无地图时直接返回直线路径
            return Path([start, goal], start.distance_to(goal))
        
        # 转换到地图坐标
        start_grid = (int((start.x - self._origin[0]) / self._resolution),
                      int((start.y - self._origin[1]) / self._resolution))
        goal_grid = (int((goal.x - self._origin[0]) / self._resolution),
                     int((goal.y - self._origin[1]) / self._resolution))
        
        # 检查起点和终点
        if self.is_collision(start.x, start.y) or self.is_collision(goal.x, goal.y):
            return None
        
        # A* 算法
        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start_grid))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_grid: 0.0}
        f_score: Dict[Tuple[int, int], float] = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # 重建路径
                return self._reconstruct_path(came_from, current, start, goal)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            for neighbor, cost in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 无可行路径
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]],
                         current: Tuple[int, int], 
                         start_world: PathPoint, 
                         goal_world: PathPoint) -> Path:
        """重建路径"""
        path_points = [current]
        
        while current in came_from:
            current = came_from[current]
            path_points.append(current)
        
        path_points.reverse()
        
        # 转换到世界坐标
        world_points = [start_world]
        for p in path_points[1:-1]:
            x = p[0] * self._resolution + self._origin[0] + self._resolution / 2
            y = p[1] * self._resolution + self._origin[1] + self._resolution / 2
            world_points.append(PathPoint(x, y))
        world_points.append(goal_world)
        
        # 计算路径代价
        cost = sum(world_points[i].distance_to(world_points[i + 1]) 
                   for i in range(len(world_points) - 1))
        
        return Path(world_points, cost)


class RRTPlanner(PathPlanner):
    """
    RRT (Rapidly-exploring Random Tree) 路径规划器
    
    适用于高维空间和复杂障碍物环境。
    """
    
    def __init__(self, map_data: Optional[np.ndarray] = None,
                 resolution: float = 0.1,
                 origin: Tuple[float, float] = (0.0, 0.0),
                 max_iter: int = 1000,
                 step_size: float = 0.1,
                 goal_sample_rate: float = 0.1,
                 search_radius: float = 0.5):
        super().__init__(map_data, resolution, origin)
        self._max_iter = max_iter
        self._step_size = step_size
        self._goal_sample_rate = goal_sample_rate
        self._search_radius = search_radius
        self._nodes: List[PathPoint] = []
        self._parent_map: Dict[int, int] = {}
        
        # 边界
        if map_data is not None:
            self._x_min = origin[0]
            self._x_max = origin[0] + map_data.shape[1] * resolution
            self._y_min = origin[1]
            self._y_max = origin[1] + map_data.shape[0] * resolution
        else:
            self._x_min, self._x_max = -10.0, 10.0
            self._y_min, self._y_max = -10.0, 10.0
    
    def set_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """设置采样边界"""
        self._x_min, self._x_max = x_min, x_max
        self._y_min, self._y_max = y_min, y_max
    
    def _random_point(self, goal: PathPoint) -> PathPoint:
        """随机采样点"""
        if random.random() < self._goal_sample_rate:
            return goal
        
        x = random.uniform(self._x_min, self._x_max)
        y = random.uniform(self._y_min, self._y_max)
        return PathPoint(x, y)
    
    def _nearest_node(self, point: PathPoint) -> int:
        """找到最近的节点索引"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, node in enumerate(self._nodes):
            dist = node.distance_to(point)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def _steer(self, from_point: PathPoint, to_point: PathPoint) -> PathPoint:
        """朝目标方向前进一步"""
        dist = from_point.distance_to(to_point)
        
        if dist < self._step_size:
            return to_point
        
        theta = math.atan2(to_point.y - from_point.y, to_point.x - from_point.x)
        new_x = from_point.x + self._step_size * math.cos(theta)
        new_y = from_point.y + self._step_size * math.sin(theta)
        
        return PathPoint(new_x, new_y)
    
    def _is_collision_free(self, p1: PathPoint, p2: PathPoint) -> bool:
        """检查两点之间是否无碰撞"""
        return not self.is_line_collision(p1, p2, step_size=self._step_size * 0.5)
    
    def plan(self, start: PathPoint, goal: PathPoint) -> Optional[Path]:
        """RRT 路径规划"""
        self._nodes = [start]
        self._parent_map = {}
        
        for _ in range(self._max_iter):
            # 随机采样
            random_point = self._random_point(goal)
            
            # 找到最近节点
            nearest_idx = self._nearest_node(random_point)
            nearest_node = self._nodes[nearest_idx]
            
            # 扩展
            new_node = self._steer(nearest_node, random_point)
            
            # 检查碰撞
            if not self._is_collision_free(nearest_node, new_node):
                continue
            
            # 添加到树
            new_idx = len(self._nodes)
            self._nodes.append(new_node)
            self._parent_map[new_idx] = nearest_idx
            
            # 检查是否到达目标
            if new_node.distance_to(goal) < self._step_size:
                if self._is_collision_free(new_node, goal):
                    # 连接目标点
                    goal_idx = len(self._nodes)
                    self._nodes.append(goal)
                    self._parent_map[goal_idx] = new_idx
                    return self._reconstruct_path(goal_idx)
        
        return None  # 规划失败
    
    def _reconstruct_path(self, goal_idx: int) -> Path:
        """重建路径"""
        path_indices = [goal_idx]
        current = goal_idx
        
        while current in self._parent_map:
            current = self._parent_map[current]
            path_indices.append(current)
        
        path_indices.reverse()
        
        points = [self._nodes[i] for i in path_indices]
        
        # 简化路径
        points = self._shortcut_path(points)
        
        # 计算代价
        cost = sum(points[i].distance_to(points[i + 1]) 
                   for i in range(len(points) - 1))
        
        return Path(points, cost)
    
    def _shortcut_path(self, points: List[PathPoint]) -> List[PathPoint]:
        """路径简化"""
        if len(points) <= 2:
            return points
        
        simplified = [points[0]]
        i = 0
        
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1:
                if self._is_collision_free(points[i], points[j]):
                    break
                j -= 1
            
            simplified.append(points[j])
            i = j
        
        return simplified


class DijkstraPlanner(AStarPlanner):
    """
    Dijkstra 路径规划器
    
    A* 的特例，启发式函数为 0，保证找到最短路径。
    """
    
    def _heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Dijkstra 不使用启发式"""
        return 0.0
