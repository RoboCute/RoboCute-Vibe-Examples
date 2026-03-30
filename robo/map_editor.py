# -*- coding: utf-8 -*-
"""
地图编辑器模块 - 提供地图创建和编辑功能

支持的功能:
- 创建/加载/保存地图
- 设置障碍物
- 地图膨胀 (Inflation)
- 导入/导出多种格式
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np


class CellType(Enum):
    """地图单元格类型"""
    FREE = 0       # 空闲
    OCCUPIED = 1   # 障碍物
    UNKNOWN = 2    # 未知
    INFLATED = 3   # 膨胀区域


@dataclass
class MapCell:
    """地图单元格"""
    x: int
    y: int
    cell_type: CellType = CellType.FREE
    cost: float = 0.0  # 通行代价 (0-1)
    
    def is_free(self) -> bool:
        """是否可通行"""
        return self.cell_type == CellType.FREE
    
    def is_occupied(self) -> bool:
        """是否是障碍物"""
        return self.cell_type == CellType.OCCUPIED


@dataclass
class MapMetadata:
    """地图元数据"""
    resolution: float = 0.1  # 米/格子
    width: int = 100         # 格子数
    height: int = 100        # 格子数
    origin_x: float = 0.0    # 原点 x (米)
    origin_y: float = 0.0    # 原点 y (米)
    frame_id: str = "map"


class MapEditor:
    """
    地图编辑器
    
    提供地图的创建、编辑和管理功能。
    """
    
    def __init__(self, metadata: Optional[MapMetadata] = None):
        """
        初始化地图编辑器
        
        Args:
            metadata: 地图元数据，默认创建 10m x 10m 地图
        """
        self._metadata = metadata or MapMetadata()
        self._data = np.zeros(
            (self._metadata.height, self._metadata.width),
            dtype=np.uint8
        )
        self._cost_map = np.zeros(
            (self._metadata.height, self._metadata.width),
            dtype=np.float32
        )
        self._modified = False
    
    @property
    def metadata(self) -> MapMetadata:
        """获取地图元数据"""
        return self._metadata
    
    @property
    def data(self) -> np.ndarray:
        """获取地图数据"""
        return self._data
    
    @property
    def cost_map(self) -> np.ndarray:
        """获取代价地图"""
        return self._cost_map
    
    @property
    def resolution(self) -> float:
        """获取地图分辨率"""
        return self._metadata.resolution
    
    @property
    def width(self) -> int:
        """获取地图宽度 (格子数)"""
        return self._metadata.width
    
    @property
    def height(self) -> int:
        """获取地图高度 (格子数)"""
        return self._metadata.height
    
    def world_to_map(self, x: float, y: float) -> Tuple[int, int]:
        """
        世界坐标转地图坐标
        
        Args:
            x, y: 世界坐标 (米)
            
        Returns:
            地图坐标 (格子索引)
        """
        mx = int((x - self._metadata.origin_x) / self._metadata.resolution)
        my = int((y - self._metadata.origin_y) / self._metadata.resolution)
        return (mx, my)
    
    def map_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """
        地图坐标转世界坐标
        
        Args:
            mx, my: 地图坐标 (格子索引)
            
        Returns:
            世界坐标 (米，格子中心)
        """
        x = mx * self._metadata.resolution + self._metadata.origin_x + self._metadata.resolution / 2
        y = my * self._metadata.resolution + self._metadata.origin_y + self._metadata.resolution / 2
        return (x, y)
    
    def is_in_bounds(self, mx: int, my: int) -> bool:
        """检查地图坐标是否在范围内"""
        return 0 <= mx < self._metadata.width and 0 <= my < self._metadata.height
    
    def get_cell(self, mx: int, my: int) -> Optional[MapCell]:
        """
        获取指定位置的单元格
        
        Args:
            mx, my: 地图坐标
            
        Returns:
            单元格信息，越界返回 None
        """
        if not self.is_in_bounds(mx, my):
            return None
        
        cell_type = CellType(self._data[my, mx])
        cost = self._cost_map[my, mx]
        return MapCell(mx, my, cell_type, cost)
    
    def set_cell(self, mx: int, my: int, cell_type: CellType, cost: float = 0.0):
        """
        设置指定位置的单元格
        
        Args:
            mx, my: 地图坐标
            cell_type: 单元格类型
            cost: 通行代价
        """
        if not self.is_in_bounds(mx, my):
            return
        
        self._data[my, mx] = cell_type.value
        self._cost_map[my, mx] = cost
        self._modified = True
    
    def set_obstacle(self, mx: int, my: int):
        """设置障碍物"""
        self.set_cell(mx, my, CellType.OCCUPIED, cost=1.0)
    
    def set_free(self, mx: int, my: int):
        """设置空闲"""
        self.set_cell(mx, my, CellType.FREE, cost=0.0)
    
    def set_obstacle_world(self, x: float, y: float):
        """在世界坐标设置障碍物"""
        mx, my = self.world_to_map(x, y)
        self.set_obstacle(mx, my)
    
    def set_free_world(self, x: float, y: float):
        """在世界坐标设置空闲"""
        mx, my = self.world_to_map(x, y)
        self.set_free(mx, my)
    
    def is_occupied(self, mx: int, my: int) -> bool:
        """检查是否障碍物"""
        cell = self.get_cell(mx, my)
        return cell is not None and cell.is_occupied()
    
    def is_occupied_world(self, x: float, y: float) -> bool:
        """检查世界坐标是否障碍物"""
        mx, my = self.world_to_map(x, y)
        return self.is_occupied(mx, my)
    
    def add_rect_obstacle(self, mx: int, my: int, width: int, height: int):
        """
        添加矩形障碍物
        
        Args:
            mx, my: 左下角坐标
            width, height: 宽度和高度 (格子数)
        """
        for dx in range(width):
            for dy in range(height):
                self.set_obstacle(mx + dx, my + dy)
    
    def add_rect_obstacle_world(self, x: float, y: float, width: float, height: float):
        """
        添加矩形障碍物 (世界坐标)
        
        Args:
            x, y: 左下角坐标 (米)
            width, height: 宽度和高度 (米)
        """
        mx, my = self.world_to_map(x, y)
        w_cells = int(width / self._metadata.resolution)
        h_cells = int(height / self._metadata.resolution)
        self.add_rect_obstacle(mx, my, w_cells, h_cells)
    
    def add_circle_obstacle(self, center_x: float, center_y: float, radius: float):
        """
        添加圆形障碍物
        
        Args:
            center_x, center_y: 圆心 (世界坐标)
            radius: 半径 (米)
        """
        cx, cy = self.world_to_map(center_x, center_y)
        r_cells = int(radius / self._metadata.resolution)
        
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy <= r_cells * r_cells:
                    self.set_obstacle(cx + dx, cy + dy)
    
    def inflate(self, radius: float, cost_scaling: float = 1.0):
        """
        地图膨胀 (生成代价地图)
        
        对障碍物进行膨胀，生成距离代价。
        
        Args:
            radius: 膨胀半径 (米)
            cost_scaling: 代价缩放因子
        """
        r_cells = int(radius / self._metadata.resolution)
        
        # 找到所有障碍物
        obstacles = []
        for y in range(self._metadata.height):
            for x in range(self._metadata.width):
                if self._data[y, x] == CellType.OCCUPIED.value:
                    obstacles.append((x, y))
        
        # 对每个障碍物进行膨胀
        for ox, oy in obstacles:
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    nx, ny = ox + dx, oy + dy
                    
                    if not self.is_in_bounds(nx, ny):
                        continue
                    
                    # 计算距离
                    dist = math.sqrt(dx * dx + dy * dy) * self._metadata.resolution
                    
                    if dist > radius:
                        continue
                    
                    # 计算代价 (指数衰减)
                    cost = 1.0 - math.exp(-cost_scaling * (radius - dist))
                    
                    # 更新代价 (取最大值)
                    current_cost = self._cost_map[ny, nx]
                    if cost > current_cost:
                        self._cost_map[ny, nx] = cost
                        
                        # 如果不是障碍物，标记为膨胀区域
                        if self._data[ny, nx] == CellType.FREE.value:
                            self._data[ny, nx] = CellType.INFLATED.value
        
        self._modified = True
    
    def clear(self):
        """清空地图"""
        self._data.fill(CellType.FREE.value)
        self._cost_map.fill(0.0)
        self._modified = True
    
    def resize(self, width: int, height: int, resolution: Optional[float] = None):
        """
        调整地图大小
        
        Args:
            width, height: 新大小 (格子数)
            resolution: 新分辨率，None 表示保持不变
        """
        new_data = np.zeros((height, width), dtype=np.uint8)
        new_cost = np.zeros((height, width), dtype=np.float32)
        
        # 复制旧数据
        h = min(self._metadata.height, height)
        w = min(self._metadata.width, width)
        new_data[:h, :w] = self._data[:h, :w]
        new_cost[:h, :w] = self._cost_map[:h, :w]
        
        self._data = new_data
        self._cost_map = new_cost
        self._metadata.width = width
        self._metadata.height = height
        
        if resolution is not None:
            self._metadata.resolution = resolution
        
        self._modified = True
    
    def to_occupancy_grid(self) -> np.ndarray:
        """
        转换为 ROS OccupancyGrid 格式
        
        Returns:
            0-100 的占用栅格地图
        """
        grid = np.zeros_like(self._data, dtype=np.int8)
        
        for y in range(self._metadata.height):
            for x in range(self._metadata.width):
                if self._data[y, x] == CellType.OCCUPIED.value:
                    grid[y, x] = 100
                elif self._data[y, x] == CellType.UNKNOWN.value:
                    grid[y, x] = -1
                else:
                    # 根据代价设置
                    cost = self._cost_map[y, x]
                    if cost > 0:
                        grid[y, x] = int(cost * 100)
        
        return grid
    
    def save(self, filepath: str):
        """
        保存地图到文件
        
        Args:
            filepath: 文件路径 (.json 或 .npy)
        """
        path = Path(filepath)
        
        if path.suffix == '.json':
            self._save_json(filepath)
        elif path.suffix == '.npy':
            self._save_npy(filepath)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        self._modified = False
    
    def _save_json(self, filepath: str):
        """保存为 JSON 格式"""
        data = {
            'metadata': asdict(self._metadata),
            'data': self._data.tolist(),
            'cost_map': self._cost_map.tolist(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _save_npy(self, filepath: str):
        """保存为 NumPy 格式"""
        np.savez(
            filepath,
            metadata=np.array([
                self._metadata.resolution,
                self._metadata.width,
                self._metadata.height,
                self._metadata.origin_x,
                self._metadata.origin_y,
            ]),
            data=self._data,
            cost_map=self._cost_map
        )
    
    @staticmethod
    def load(filepath: str) -> MapEditor:
        """
        从文件加载地图
        
        Args:
            filepath: 文件路径
            
        Returns:
            MapEditor 实例
        """
        path = Path(filepath)
        
        if path.suffix == '.json':
            return MapEditor._load_json(filepath)
        elif path.suffix == '.npy' or path.suffix == '.npz':
            return MapEditor._load_npy(filepath)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @staticmethod
    def _load_json(filepath: str) -> MapEditor:
        """从 JSON 加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = MapMetadata(**data['metadata'])
        editor = MapEditor(metadata)
        editor._data = np.array(data['data'], dtype=np.uint8)
        editor._cost_map = np.array(data['cost_map'], dtype=np.float32)
        return editor
    
    @staticmethod
    def _load_npy(filepath: str) -> MapEditor:
        """从 NumPy 加载"""
        npz = np.load(filepath)
        
        meta_arr = npz['metadata']
        metadata = MapMetadata(
            resolution=float(meta_arr[0]),
            width=int(meta_arr[1]),
            height=int(meta_arr[2]),
            origin_x=float(meta_arr[3]),
            origin_y=float(meta_arr[4]),
        )
        
        editor = MapEditor(metadata)
        editor._data = npz['data']
        editor._cost_map = npz['cost_map']
        return editor
    
    @staticmethod
    def create_empty(width: float = 10.0, height: float = 10.0,
                     resolution: float = 0.1,
                     origin_x: float = 0.0, origin_y: float = 0.0) -> MapEditor:
        """
        创建空地图
        
        Args:
            width, height: 地图尺寸 (米)
            resolution: 分辨率 (米/格子)
            origin_x, origin_y: 原点位置
            
        Returns:
            MapEditor 实例
        """
        w_cells = int(width / resolution)
        h_cells = int(height / resolution)
        
        metadata = MapMetadata(
            resolution=resolution,
            width=w_cells,
            height=h_cells,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        
        return MapEditor(metadata)
    
    @staticmethod
    def create_with_walls(width: float = 10.0, height: float = 10.0,
                          wall_thickness: float = 0.2,
                          resolution: float = 0.1) -> MapEditor:
        """
        创建带围墙的地图
        
        Args:
            width, height: 地图尺寸 (米)
            wall_thickness: 围墙厚度 (米)
            resolution: 分辨率
            
        Returns:
            MapEditor 实例
        """
        editor = MapEditor.create_empty(width, height, resolution)
        
        t_cells = int(wall_thickness / resolution)
        w_cells = editor.width
        h_cells = editor.height
        
        # 四边围墙
        for x in range(w_cells):
            for i in range(t_cells):
                editor.set_obstacle(x, i)  # 底边
                editor.set_obstacle(x, h_cells - 1 - i)  # 顶边
        
        for y in range(h_cells):
            for i in range(t_cells):
                editor.set_obstacle(i, y)  # 左边
                editor.set_obstacle(w_cells - 1 - i, y)  # 右边
        
        return editor
    
    def get_neighbors(self, mx: int, my: int, allow_diagonal: bool = True) -> List[Tuple[int, int]]:
        """
        获取邻居格子
        
        Args:
            mx, my: 当前格子
            allow_diagonal: 是否允许对角移动
            
        Returns:
            邻居坐标列表
        """
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        if allow_diagonal:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        
        for dx, dy in directions:
            nx, ny = mx + dx, my + dy
            if self.is_in_bounds(nx, ny) and not self.is_occupied(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def find_frontier_cells(self) -> List[Tuple[int, int]]:
        """
        找到边界单元格 (前沿探索用)
        
        Returns:
            边界格子坐标列表
        """
        frontiers = []
        
        for y in range(1, self._metadata.height - 1):
            for x in range(1, self._metadata.width - 1):
                if self._data[y, x] == CellType.FREE.value:
                    # 检查是否有邻居是未知区域
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if self._data[y + dy, x + dx] == CellType.UNKNOWN.value:
                            frontiers.append((x, y))
                            break
        
        return frontiers
