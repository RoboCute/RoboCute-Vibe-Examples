# -*- coding: utf-8 -*-
"""
A* Path Planning Algorithm

A* (A-Star) 路径规划算法实现
用于在栅格地图上寻找从起点到终点的最优路径

Features:
    - 8-directional movement (4 cardinal + 4 diagonal)
    - Euclidean distance heuristic
    - Binary heap for efficient priority queue
    - Supports customizable collision detection
    - Grid-based and continuous world coordinate support
"""

import heapq
import math
from typing import List, Tuple, Optional, Set, Dict, Callable, Union
from dataclasses import dataclass, field


@dataclass
class Path:
    """路径结果数据结构"""
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    total_cost: float = 0.0
    is_valid: bool = False
    nodes_explored: int = 0  # 探索的节点数
    
    def __len__(self) -> int:
        return len(self.waypoints)
    
    def __repr__(self) -> str:
        status = "Valid" if self.is_valid else "Invalid"
        return f"Path({status}, {len(self)} waypoints, cost={self.total_cost:.3f})"


class Node:
    """A* 搜索节点"""
    
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.g: float = float('inf')  # 从起点到当前节点的实际代价
        self.h: float = 0.0           # 启发式估计代价
        self.parent: Optional['Node'] = None
    
    @property
    def f(self) -> float:
        """总代价 f = g + h"""
        return self.g + self.h
    
    def get_position(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __lt__(self, other: 'Node') -> bool:
        """用于优先队列比较"""
        return self.f < other.f
    
    def __repr__(self) -> str:
        return f"Node({self.x}, {self.y}, g={self.g:.2f}, h={self.h:.2f})"


class AStarPlanner:
    """
    A* 路径规划算法实现
    
    A* 算法结合了 Dijkstra 算法的完备性和贪心搜索的效率，
    通过启发式函数引导搜索方向，快速找到最优路径。
    
    Attributes:
        collision_check: 碰撞检测函数，输入 (x, y) 返回是否碰撞
        resolution: 栅格分辨率（世界坐标单位/栅格）
        diagonal_allowed: 是否允许对角移动
    
    Example:
        >>> # 创建简单的碰撞检测函数
        >>> obstacles = {(2, 2), (2, 3), (3, 2)}
        >>> collision_fn = lambda x, y: (int(x), int(y)) in obstacles
        >>> 
        >>> # 创建规划器
        >>> planner = AStarPlanner(collision_fn, resolution=1.0)
        >>> 
        >>> # 规划路径
        >>> path = planner.plan((0, 0), (5, 5))
        >>> print(path)
        Path(Valid, 8 waypoints, cost=7.071)
    """
    
    def __init__(
        self,
        collision_check: Callable[[float, float], bool],
        resolution: float = 0.1,
        diagonal_allowed: bool = True
    ):
        """
        初始化 A* 规划器
        
        Args:
            collision_check: 碰撞检测函数，输入 (x, y)，返回 True 表示有碰撞
            resolution: 栅格分辨率（米/栅格），默认 0.1
            diagonal_allowed: 是否允许对角移动，默认 True
        """
        self.collision_check = collision_check
        self.resolution = resolution
        self.diagonal_allowed = diagonal_allowed
        
        # 定义移动方向 (dx, dy, cost)
        self.directions: List[Tuple[int, int, float]] = [
            (0, 1, 1.0),    # 上
            (0, -1, 1.0),   # 下
            (1, 0, 1.0),    # 右
            (-1, 0, 1.0),   # 左
        ]
        
        if diagonal_allowed:
            self.directions.extend([
                (1, 1, math.sqrt(2)),     # 右上
                (1, -1, math.sqrt(2)),    # 右下
                (-1, 1, math.sqrt(2)),    # 左上
                (-1, -1, math.sqrt(2)),   # 左下
            ])
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        世界坐标转换为栅格坐标
        
        Args:
            x: 世界坐标 X
            y: 世界坐标 Y
            
        Returns:
            (gx, gy) 栅格坐标
        """
        return (int(round(x / self.resolution)), int(round(y / self.resolution)))
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """
        栅格坐标转换为世界坐标
        
        Args:
            gx: 栅格坐标 X
            gy: 栅格坐标 Y
            
        Returns:
            (x, y) 世界坐标
        """
        return (gx * self.resolution, gy * self.resolution)
    
    def heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """
        启发式函数 - 欧几里得距离
        
        使用欧几里得距离作为启发式，保证可采纳性（admissible）
        对于允许对角移动的网格，这是最优的启发式函数
        
        Args:
            p1: 点1 (x1, y1)
            p2: 点2 (x2, y2)
            
        Returns:
            启发式估计代价
        """
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def is_valid(self, x: float, y: float) -> bool:
        """
        检查位置是否有效（无碰撞）
        
        Args:
            x: 世界坐标 X
            y: 世界坐标 Y
            
        Returns:
            True 如果位置有效
        """
        return not self.collision_check(x, y)
    
    def get_neighbors(self, node: Node) -> List[Tuple[int, int, float]]:
        """
        获取节点的所有有效邻居
        
        Args:
            node: 当前节点
            
        Returns:
            邻居列表 [(nx, ny, cost), ...]
        """
        neighbors = []
        
        for dx, dy, cost in self.directions:
            nx, ny = node.x + dx, node.y + dy
            wx, wy = self.grid_to_world(nx, ny)
            
            if self.is_valid(wx, wy):
                neighbors.append((nx, ny, cost))
        
        return neighbors
    
    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        max_iterations: int = 100000
    ) -> Path:
        """
        执行 A* 路径规划
        
        Args:
            start: 起点 (x, y) 世界坐标
            goal: 终点 (x, y) 世界坐标
            max_iterations: 最大迭代次数，防止无限循环
            
        Returns:
            Path 对象，包含路径结果
        """
        # 转换为栅格坐标
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        # 检查起点和终点有效性
        if not self.is_valid(start[0], start[1]):
            return Path(is_valid=False)
        
        if not self.is_valid(goal[0], goal[1]):
            return Path(is_valid=False)
        
        # 起点就是终点
        if start_grid == goal_grid:
            return Path(
                waypoints=[start, goal],
                total_cost=0.0,
                is_valid=True,
                nodes_explored=1
            )
        
        # 初始化起点节点
        start_node = Node(start_grid[0], start_grid[1])
        start_node.g = 0.0
        start_node.h = self.heuristic(start_grid, goal_grid)
        
        # 优先队列 (f值, 节点)
        open_list: List[Tuple[float, Node]] = [(start_node.f, start_node)]
        
        # 已访问集合 - 存储最优 g 值
        closed_set: Set[Tuple[int, int]] = set()
        g_scores: Dict[Tuple[int, int], float] = {start_grid: 0.0}
        
        # 父节点映射，用于重建路径
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        nodes_explored = 0
        
        while open_list and nodes_explored < max_iterations:
            # 获取 f 值最小的节点
            _, current = heapq.heappop(open_list)
            current_pos = current.get_position()
            
            # 跳过已处理过的节点（处理堆中过期条目）
            if current_pos in closed_set:
                continue
            
            nodes_explored += 1
            
            # 到达目标
            if current_pos == goal_grid:
                path = self._reconstruct_path(came_from, current_pos, start_grid)
                world_path = [self.grid_to_world(x, y) for x, y in path]
                
                # 确保起点和终点精确
                world_path[0] = start
                world_path[-1] = goal
                
                return Path(
                    waypoints=world_path,
                    total_cost=g_scores[current_pos],
                    is_valid=True,
                    nodes_explored=nodes_explored
                )
            
            closed_set.add(current_pos)
            
            # 扩展邻居
            for nx, ny, move_cost in self.get_neighbors(current):
                neighbor_pos = (nx, ny)
                
                if neighbor_pos in closed_set:
                    continue
                
                # 计算 tentative g 值
                tentative_g = g_scores[current_pos] + move_cost
                
                # 如果发现更优路径
                if neighbor_pos not in g_scores or tentative_g < g_scores[neighbor_pos]:
                    came_from[neighbor_pos] = current_pos
                    g_scores[neighbor_pos] = tentative_g
                    
                    neighbor_node = Node(nx, ny)
                    neighbor_node.g = tentative_g
                    neighbor_node.h = self.heuristic(neighbor_pos, goal_grid)
                    
                    heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
        
        # 未找到路径
        return Path(is_valid=False, nodes_explored=nodes_explored)
    
    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
        start: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        从终点回溯重建路径
        
        Args:
            came_from: 父节点映射
            current: 当前节点（终点）
            start: 起点
            
        Returns:
            从起点到终点的栅格路径
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def plan_smooth(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        **kwargs
    ) -> Path:
        """
        规划路径并平滑处理
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            **kwargs: 传递给 plan 的额外参数
            
        Returns:
            平滑后的路径
        """
        path = self.plan(start, goal, **kwargs)
        if not path.is_valid or len(path) < 3:
            return path
        
        # 简单的路径平滑：移除冗余点
        smoothed = self._simplify_path(path.waypoints)
        
        return Path(
            waypoints=smoothed,
            total_cost=path.total_cost,
            is_valid=True,
            nodes_explored=path.nodes_explored
        )
    
    def _simplify_path(
        self,
        waypoints: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        简化路径，移除共线点
        
        Args:
            waypoints: 原始路径点
            
        Returns:
            简化后的路径
        """
        if len(waypoints) <= 2:
            return waypoints.copy()
        
        result = [waypoints[0]]
        
        for i in range(1, len(waypoints) - 1):
            p0 = waypoints[i - 1]
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            
            # 检查三点是否共线
            dx1 = p1[0] - p0[0]
            dy1 = p1[1] - p0[1]
            dx2 = p2[0] - p1[0]
            dy2 = p2[1] - p1[1]
            
            # 计算叉积
            cross = dx1 * dy2 - dy1 * dx2
            
            # 如果不共线，保留该点
            if abs(cross) > 1e-6:
                result.append(p1)
        
        result.append(waypoints[-1])
        return result


# ==================== 示例和测试 ====================

def example_basic():
    """基础使用示例"""
    print("=" * 50)
    print("A* 路径规划 - 基础示例")
    print("=" * 50)
    
    # 定义障碍物（简单迷宫）
    obstacles = {
        (2, 0), (2, 1), (2, 2), (2, 3),
        (3, 3), (4, 3), (5, 3),
        (5, 4), (5, 5)
    }
    
    def collision_fn(x: float, y: float) -> bool:
        gx, gy = int(round(x)), int(round(y))
        return (gx, gy) in obstacles
    
    # 创建规划器
    planner = AStarPlanner(collision_fn, resolution=1.0)
    
    # 规划路径
    start = (0, 0)
    goal = (6, 6)
    
    print(f"起点: {start}")
    print(f"终点: {goal}")
    print(f"障碍物: {obstacles}")
    print()
    
    path = planner.plan(start, goal)
    
    if path.is_valid:
        print(f"[OK] 找到路径!")
        print(f"  路径点数量: {len(path)}")
        print(f"  总代价: {path.total_cost:.3f}")
        print(f"  探索节点数: {path.nodes_explored}")
        print(f"  路径点: {path.waypoints}")
    else:
        print("[FAIL] 未找到路径")
    
    return path


def example_with_visualization():
    """带可视化的示例"""
    print("\n" + "=" * 50)
    print("A* 路径规划 - 可视化示例")
    print("=" * 50)
    
    # 创建 20x20 的地图
    width, height = 20, 20
    obstacles = set()
    
    # 添加一些随机障碍物
    import random
    random.seed(42)
    for _ in range(60):
        ox = random.randint(2, width - 3)
        oy = random.randint(2, height - 3)
        obstacles.add((ox, oy))
    
    def collision_fn(x: float, y: float) -> bool:
        gx, gy = int(round(x)), int(round(y))
        # 边界检查
        if gx < 0 or gx >= width or gy < 0 or gy >= height:
            return True
        return (gx, gy) in obstacles
    
    planner = AStarPlanner(collision_fn, resolution=1.0)
    
    start = (1, 1)
    goal = (18, 18)
    
    print(f"地图大小: {width}x{height}")
    print(f"起点: {start}")
    print(f"终点: {goal}")
    print(f"障碍物数量: {len(obstacles)}")
    print()
    
    path = planner.plan(start, goal)
    
    if path.is_valid:
        print(f"[OK] 找到路径!")
        print(f"  路径点数量: {len(path)}")
        print(f"  总代价: {path.total_cost:.3f}")
        print(f"  探索节点数: {path.nodes_explored}")
        
        # 可视化 ASCII 地图
        print("\n地图可视化:")
        print("S = 起点, G = 终点, # = 障碍物, * = 路径, . = 空")
        print("-" * (width + 2))
        
        path_set = {(int(round(p[0])), int(round(p[1]))) for p in path.waypoints}
        
        for y in range(height - 1, -1, -1):
            row = ["|"]
            for x in range(width):
                if (x, y) == (int(start[0]), int(start[1])):
                    row.append("S")
                elif (x, y) == (int(goal[0]), int(goal[1])):
                    row.append("G")
                elif (x, y) in obstacles:
                    row.append("#")
                elif (x, y) in path_set:
                    row.append("*")
                else:
                    row.append(".")
            row.append("|")
            print("".join(row))
        print("-" * (width + 2))
    else:
        print("[FAIL] 未找到路径")
    
    return path


def example_comparison():
    """对比允许和禁止对角移动的效果"""
    print("\n" + "=" * 50)
    print("A* 路径规划 - 对角移动对比")
    print("=" * 50)
    
    # 简单地图
    obstacles = {(2, 2), (2, 3), (3, 2)}
    
    def collision_fn(x: float, y: float) -> bool:
        gx, gy = int(round(x)), int(round(y))
        return (gx, gy) in obstacles
    
    start = (0, 0)
    goal = (5, 5)
    
    # 允许对角移动
    planner_diag = AStarPlanner(collision_fn, resolution=1.0, diagonal_allowed=True)
    path_diag = planner_diag.plan(start, goal)
    
    # 禁止对角移动
    planner_straight = AStarPlanner(collision_fn, resolution=1.0, diagonal_allowed=False)
    path_straight = planner_straight.plan(start, goal)
    
    print(f"起点: {start}, 终点: {goal}")
    print()
    print("允许对角移动:")
    print(f"  路径长度: {len(path_diag)} 点")
    print(f"  总代价: {path_diag.total_cost:.3f}")
    print(f"  路径: {path_diag.waypoints}")
    print()
    print("禁止对角移动:")
    print(f"  路径长度: {len(path_straight)} 点")
    print(f"  总代价: {path_straight.total_cost:.3f}")
    print(f"  路径: {path_straight.waypoints}")


def example_smooth_path():
    """路径平滑示例"""
    print("\n" + "=" * 50)
    print("A* 路径规划 - 路径平滑")
    print("=" * 50)
    
    # L 形障碍物
    obstacles = set()
    for i in range(8):
        obstacles.add((5, i))
        obstacles.add((i, 5))
    
    def collision_fn(x: float, y: float) -> bool:
        gx, gy = int(round(x)), int(round(y))
        return (gx, gy) in obstacles
    
    planner = AStarPlanner(collision_fn, resolution=1.0)
    
    start = (0, 0)
    goal = (8, 8)
    
    # 原始路径
    path = planner.plan(start, goal)
    # 平滑路径
    smooth_path = planner.plan_smooth(start, goal)
    
    print(f"起点: {start}, 终点: {goal}")
    print()
    print("原始路径:")
    print(f"  点数: {len(path)}")
    print(f"  代价: {path.total_cost:.3f}")
    print()
    print("平滑后路径:")
    print(f"  点数: {len(smooth_path)}")
    print(f"  代价: {smooth_path.total_cost:.3f}")


if __name__ == "__main__":
    # 运行所有示例
    example_basic()
    example_with_visualization()
    example_comparison()
    example_smooth_path()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成!")
    print("=" * 50)
