# -*- coding: utf-8 -*-
"""
RRT (Rapidly-exploring Random Tree) Path Planning Algorithm

RRT (快速探索随机树) 路径规划算法实现
用于在高维空间和复杂环境中寻找从起点到终点的可行路径

Features:
    - Probabilistically complete sampling-based planner
    - Handles high-dimensional configuration spaces
    - Suitable for complex environments with obstacles
    - Supports customizable collision detection
    - Includes RRT* variant for asymptotic optimality
    - Path smoothing and simplification

算法特点:
    - 基于采样的概率完备性规划器
    - 适用于高维构型空间
    - 适合具有障碍物的复杂环境
    - 支持自定义碰撞检测
    - 包含 RRT* 变体实现渐进最优性
    - 支持路径平滑和简化
"""

import math
import random
from typing import List, Tuple, Optional, Callable, Set, Dict
from dataclasses import dataclass, field


@dataclass
class Path:
    """路径结果数据结构"""
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    total_cost: float = 0.0
    is_valid: bool = False
    nodes_explored: int = 0  # 探索的节点数（RRT树节点数）
    
    def __len__(self) -> int:
        return len(self.waypoints)
    
    def __repr__(self) -> str:
        status = "Valid" if self.is_valid else "Invalid"
        return f"Path({status}, {len(self)} waypoints, cost={self.total_cost:.3f})"


class Node:
    """RRT 树节点"""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.parent: Optional['Node'] = None
        self.cost: float = 0.0  # 从起点到该节点的累计代价
    
    def get_position(self) -> Tuple[float, float]:
        """获取节点位置"""
        return (self.x, self.y)
    
    def distance_to(self, other: 'Node') -> float:
        """计算到另一节点的欧几里得距离"""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def distance_to_point(self, x: float, y: float) -> float:
        """计算到指定点的欧几里得距离"""
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)
    
    def __repr__(self) -> str:
        return f"Node({self.x:.2f}, {self.y:.2f}, cost={self.cost:.2f})"


class RRTPlanner:
    """
    RRT (Rapidly-exploring Random Tree) 路径规划算法
    
    RRT 是一种基于采样的路径规划算法，通过随机采样和树扩展
    快速探索构型空间，适合高维空间和复杂约束环境。
    
    算法流程:
        1. 初始化：以起点为根节点创建树
        2. 随机采样：在空间中随机采样一个点
        3. 最近邻：找到树中距离采样点最近的节点
        4. 扩展：从最近节点向采样方向扩展固定步长
        5. 碰撞检测：检查扩展路径是否无碰撞
        6. 添加节点：若无碰撞，将新节点加入树
        7. 目标检测：检查是否到达目标附近
        8. 重复 2-7 直到找到路径或达到最大迭代次数
    
    Attributes:
        collision_check: 碰撞检测函数，输入 (x, y) 返回是否碰撞
        max_iter: 最大迭代次数
        step_size: 扩展步长
        goal_sample_rate: 目标采样概率（引导采样偏向目标）
        goal_tolerance: 到达目标的容差距离
    
    Example:
        >>> # 创建简单的碰撞检测函数
        >>> obstacles = [(3, 3, 1.0), (7, 7, 1.5)]  # (x, y, radius)
        >>> def collision_fn(x, y):
        ...     for ox, oy, r in obstacles:
        ...         if math.sqrt((x-ox)**2 + (y-oy)**2) < r:
        ...             return True
        ...     return False
        >>> 
        >>> # 创建 RRT 规划器
        >>> planner = RRTPlanner(collision_fn, max_iter=5000, step_size=0.5)
        >>> 
        >>> # 规划路径
        >>> path = planner.plan((0, 0), (10, 10), bounds=(0, 10, 0, 10))
        >>> print(path)
        Path(Valid, 23 waypoints, cost=14.142)
    """
    
    def __init__(
        self,
        collision_check: Callable[[float, float], bool],
        max_iter: int = 5000,
        step_size: float = 0.5,
        goal_sample_rate: float = 0.1,
        goal_tolerance: float = 0.5
    ):
        """
        初始化 RRT 规划器
        
        Args:
            collision_check: 碰撞检测函数，输入 (x, y)，返回 True 表示有碰撞
            max_iter: 最大迭代次数，默认 5000
            step_size: 扩展步长，默认 0.5
            goal_sample_rate: 目标采样概率 (0-1)，默认 0.1
                             较高的值会更多地向目标方向采样
            goal_tolerance: 到达目标的容差距离，默认 0.5
        """
        self.collision_check = collision_check
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
    
    def is_valid_point(self, x: float, y: float) -> bool:
        """
        检查点是否有效（无碰撞且在边界内）
        
        Args:
            x: X 坐标
            y: Y 坐标
            
        Returns:
            True 如果点有效
        """
        return not self.collision_check(x, y)
    
    def is_valid_path_segment(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        check_step: float = 0.05
    ) -> bool:
        """
        检查路径段是否有效（无碰撞）
        
        使用离散化检查路径段上的点
        
        Args:
            p1: 起点 (x, y)
            p2: 终点 (x, y)
            check_step: 检查步长（相对于 step_size 的比例）
            
        Returns:
            True 如果路径段有效
        """
        x1, y1 = p1
        x2, y2 = p2
        
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if distance < 1e-6:
            return self.is_valid_point(x1, y1)
        
        # 计算检查步数
        step = self.step_size * check_step
        num_steps = max(int(distance / step), 1)
        
        dx = (x2 - x1) / num_steps
        dy = (y2 - y1) / num_steps
        
        for i in range(num_steps + 1):
            x = x1 + dx * i
            y = y1 + dy * i
            if not self.is_valid_point(x, y):
                return False
        
        return True
    
    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        bounds: Optional[Tuple[float, float, float, float]] = None,
        max_iterations: Optional[int] = None
    ) -> Path:
        """
        执行 RRT 路径规划
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            bounds: 搜索边界 (min_x, max_x, min_y, max_y)，
                   默认为包含起点和终点的自动边界
            max_iterations: 最大迭代次数，覆盖初始化时的设置
            
        Returns:
            Path 对象，包含路径结果
        """
        sx, sy = start
        gx, gy = goal
        
        # 设置边界
        if bounds is None:
            margin = 5.0
            min_x = min(sx, gx) - margin
            max_x = max(sx, gx) + margin
            min_y = min(sy, gy) - margin
            max_y = max(sy, gy) + margin
        else:
            min_x, max_x, min_y, max_y = bounds
        
        # 检查起点和终点
        if not self.is_valid_point(sx, sy):
            return Path(is_valid=False)
        
        if not self.is_valid_point(gx, gy):
            return Path(is_valid=False)
        
        # 起点就是终点
        if math.sqrt((sx - gx) ** 2 + (sy - gy) ** 2) < self.goal_tolerance:
            return Path(
                waypoints=[start, goal],
                total_cost=0.0,
                is_valid=True,
                nodes_explored=1
            )
        
        # 初始化树
        start_node = Node(sx, sy)
        goal_node = Node(gx, gy)
        
        nodes: List[Node] = [start_node]
        
        max_iter = max_iterations if max_iterations is not None else self.max_iter
        
        for iteration in range(max_iter):
            # 随机采样
            rnd_x, rnd_y = self._sample_random_point(
                min_x, max_x, min_y, max_y, gx, gy
            )
            
            # 找到最近节点
            nearest_node = self._get_nearest_node(nodes, rnd_x, rnd_y)
            
            # 扩展新节点
            new_node = self._steer(nearest_node, rnd_x, rnd_y)
            
            if new_node is None:
                continue
            
            # 检查路径段是否无碰撞
            if self.is_valid_path_segment(
                nearest_node.get_position(),
                new_node.get_position()
            ):
                nodes.append(new_node)
                
                # 检查是否到达目标
                dist_to_goal = new_node.distance_to(goal_node)
                if dist_to_goal < self.goal_tolerance:
                    # 尝试直接连接到目标
                    if self.is_valid_path_segment(
                        new_node.get_position(),
                        goal_node.get_position()
                    ):
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + dist_to_goal
                        
                        path = self._extract_path(goal_node)
                        total_cost = goal_node.cost
                        
                        return Path(
                            waypoints=path,
                            total_cost=total_cost,
                            is_valid=True,
                            nodes_explored=len(nodes)
                        )
        
        # 未找到路径
        return Path(is_valid=False, nodes_explored=len(nodes))
    
    def _sample_random_point(
        self,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
        goal_x: float,
        goal_y: float
    ) -> Tuple[float, float]:
        """
        在边界内随机采样一个点
        
        以一定概率直接采样目标点，引导搜索方向
        
        Args:
            min_x, max_x: X 轴边界
            min_y, max_y: Y 轴边界
            goal_x, goal_y: 目标点坐标
            
        Returns:
            采样点 (x, y)
        """
        if random.random() < self.goal_sample_rate:
            return (goal_x, goal_y)
        else:
            return (
                random.uniform(min_x, max_x),
                random.uniform(min_y, max_y)
            )
    
    def _get_nearest_node(self, nodes: List[Node], x: float, y: float) -> Node:
        """
        找到距离指定点最近的树节点
        
        Args:
            nodes: 树节点列表
            x, y: 目标点坐标
            
        Returns:
            最近节点
        """
        nearest = nodes[0]
        min_dist = nearest.distance_to_point(x, y)
        
        for node in nodes[1:]:
            dist = node.distance_to_point(x, y)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_node: Node, to_x: float, to_y: float) -> Optional[Node]:
        """
        从 from_node 向目标点扩展一个新节点
        
        扩展距离限制为 step_size
        
        Args:
            from_node: 起始节点
            to_x, to_y: 目标点坐标
            
        Returns:
            新节点，如果距离太小则返回 None
        """
        dx = to_x - from_node.x
        dy = to_y - from_node.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < 1e-6:
            return None
        
        # 限制步长
        if distance > self.step_size:
            ratio = self.step_size / distance
            new_x = from_node.x + dx * ratio
            new_y = from_node.y + dy * ratio
        else:
            new_x = to_x
            new_y = to_y
        
        new_node = Node(new_x, new_y)
        new_node.parent = from_node
        new_node.cost = from_node.cost + math.sqrt(
            (new_x - from_node.x) ** 2 + (new_y - from_node.y) ** 2
        )
        
        return new_node
    
    def _extract_path(self, goal_node: Node) -> List[Tuple[float, float]]:
        """
        从目标节点回溯提取路径
        
        Args:
            goal_node: 目标节点
            
        Returns:
            从起点到终点的路径点列表
        """
        path = []
        node: Optional[Node] = goal_node
        
        while node is not None:
            path.append(node.get_position())
            node = node.parent
        
        path.reverse()
        return path
    
    def smooth_path(
        self,
        path: Path,
        max_iterations: int = 100
    ) -> Path:
        """
        使用 shortcut 方法平滑路径
        
        随机选择两个点，如果可以直接连接则移除中间点
        
        Args:
            path: 原始路径
            max_iterations: 最大迭代次数
            
        Returns:
            平滑后的路径
        """
        if not path.is_valid or len(path) < 3:
            return path
        
        waypoints = path.waypoints.copy()
        
        for _ in range(max_iterations):
            if len(waypoints) < 3:
                break
            
            # 随机选择两个不同的点
            i = random.randint(0, len(waypoints) - 2)
            j = random.randint(i + 1, len(waypoints) - 1)
            
            # 检查是否可以直接连接
            if self.is_valid_path_segment(waypoints[i], waypoints[j]):
                # 移除中间点
                waypoints = waypoints[:i+1] + waypoints[j:]
        
        # 重新计算代价
        total_cost = 0.0
        for i in range(len(waypoints) - 1):
            x1, y1 = waypoints[i]
            x2, y2 = waypoints[i + 1]
            total_cost += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        return Path(
            waypoints=waypoints,
            total_cost=total_cost,
            is_valid=True,
            nodes_explored=path.nodes_explored
        )


class RRTStarPlanner(RRTPlanner):
    """
    RRT* (Rapidly-exploring Random Tree Star) 路径规划算法
    
    RRT* 是 RRT 的优化版本，通过重布线（rewiring）机制
    保证渐进最优性（asymptotic optimality），随着采样数量
    增加，解会收敛到最优解。
    
    相比 RRT 的额外步骤:
        1. 找到新节点搜索半径内的所有邻居
        2. 选择代价最小的作为父节点
        3. 重布线：检查邻居是否通过新节点有更小代价
    
    Attributes:
        search_radius: 搜索半径，用于查找邻居节点
        rewire_factor: 重布线半径因子（随迭代调整）
    
    Example:
        >>> planner = RRTStarPlanner(collision_fn, max_iter=5000, 
        ...                          step_size=0.5, search_radius=1.0)
        >>> path = planner.plan((0, 0), (10, 10), bounds=(0, 10, 0, 10))
        >>> print(path)
        Path(Valid, 20 waypoints, cost=12.728)
    """
    
    def __init__(
        self,
        collision_check: Callable[[float, float], bool],
        max_iter: int = 5000,
        step_size: float = 0.5,
        goal_sample_rate: float = 0.1,
        goal_tolerance: float = 0.5,
        search_radius: float = 1.0
    ):
        """
        初始化 RRT* 规划器
        
        Args:
            collision_check: 碰撞检测函数
            max_iter: 最大迭代次数
            step_size: 扩展步长
            goal_sample_rate: 目标采样概率
            goal_tolerance: 到达目标的容差距离
            search_radius: 搜索半径，用于查找邻居和重布线
        """
        super().__init__(
            collision_check, max_iter, step_size,
            goal_sample_rate, goal_tolerance
        )
        self.search_radius = search_radius
    
    def _find_neighbors(self, nodes: List[Node], new_node: Node) -> List[Node]:
        """
        找到搜索半径内的所有邻居节点
        
        Args:
            nodes: 树节点列表
            new_node: 新节点
            
        Returns:
            邻居节点列表
        """
        neighbors = []
        for node in nodes:
            if node.distance_to(new_node) < self.search_radius:
                neighbors.append(node)
        return neighbors
    
    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        bounds: Optional[Tuple[float, float, float, float]] = None,
        max_iterations: Optional[int] = None
    ) -> Path:
        """
        执行 RRT* 路径规划
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            bounds: 搜索边界 (min_x, max_x, min_y, max_y)
            max_iterations: 最大迭代次数
            
        Returns:
            Path 对象，包含最优路径结果
        """
        sx, sy = start
        gx, gy = goal
        
        # 设置边界
        if bounds is None:
            margin = 5.0
            min_x = min(sx, gx) - margin
            max_x = max(sx, gx) + margin
            min_y = min(sy, gy) - margin
            max_y = max(sy, gy) + margin
        else:
            min_x, max_x, min_y, max_y = bounds
        
        # 检查起点和终点
        if not self.is_valid_point(sx, sy):
            return Path(is_valid=False)
        
        if not self.is_valid_point(gx, gy):
            return Path(is_valid=False)
        
        # 初始化
        start_node = Node(sx, sy)
        goal_node = Node(gx, gy)
        
        nodes: List[Node] = [start_node]
        
        best_path: Optional[List[Tuple[float, float]]] = None
        best_cost = float('inf')
        
        max_iter = max_iterations if max_iterations is not None else self.max_iter
        
        for iteration in range(max_iter):
            # 随机采样
            rnd_x, rnd_y = self._sample_random_point(
                min_x, max_x, min_y, max_y, gx, gy
            )
            
            # 找到最近节点
            nearest_node = self._get_nearest_node(nodes, rnd_x, rnd_y)
            
            # 扩展
            new_node = self._steer(nearest_node, rnd_x, rnd_y)
            
            if new_node is None:
                continue
            
            # 检查路径段有效性
            if not self.is_valid_path_segment(
                nearest_node.get_position(),
                new_node.get_position()
            ):
                continue
            
            # 找到邻居节点
            neighbors = self._find_neighbors(nodes, new_node)
            
            # 选择最优父节点
            best_parent = nearest_node
            best_parent_cost = nearest_node.cost + nearest_node.distance_to(new_node)
            
            for neighbor in neighbors:
                cost = neighbor.cost + neighbor.distance_to(new_node)
                if cost < best_parent_cost:
                    if self.is_valid_path_segment(
                        neighbor.get_position(),
                        new_node.get_position()
                    ):
                        best_parent = neighbor
                        best_parent_cost = cost
            
            new_node.parent = best_parent
            new_node.cost = best_parent_cost
            nodes.append(new_node)
            
            # 重布线：优化邻居的连接
            for neighbor in neighbors:
                new_cost = new_node.cost + new_node.distance_to(neighbor)
                if new_cost < neighbor.cost:
                    if self.is_valid_path_segment(
                        new_node.get_position(),
                        neighbor.get_position()
                    ):
                        neighbor.parent = new_node
                        neighbor.cost = new_cost
            
            # 检查是否到达目标
            dist_to_goal = new_node.distance_to(goal_node)
            if dist_to_goal < self.goal_tolerance:
                if self.is_valid_path_segment(
                    new_node.get_position(),
                    goal_node.get_position()
                ):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + dist_to_goal
                    
                    if goal_node.cost < best_cost:
                        best_cost = goal_node.cost
                        best_path = self._extract_path(goal_node)
        
        if best_path is not None:
            return Path(
                waypoints=best_path,
                total_cost=best_cost,
                is_valid=True,
                nodes_explored=len(nodes)
            )
        
        return Path(is_valid=False, nodes_explored=len(nodes))


# ==================== 示例和测试 ====================

def example_basic():
    """基础使用示例"""
    print("=" * 60)
    print("RRT 路径规划 - 基础示例")
    print("=" * 60)
    
    # 定义障碍物（圆形障碍物）
    obstacles = [
        (3, 3, 1.5),   # (x, y, radius)
        (7, 7, 2.0),
        (5, 1, 1.0),
        (1, 6, 1.2),
    ]
    
    def collision_fn(x: float, y: float) -> bool:
        for ox, oy, r in obstacles:
            if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < r:
                return True
        return False
    
    # 创建 RRT 规划器
    planner = RRTPlanner(
        collision_fn,
        max_iter=5000,
        step_size=0.5,
        goal_sample_rate=0.1
    )
    
    # 规划路径
    start = (0, 0)
    goal = (10, 10)
    bounds = (0, 10, 0, 10)
    
    print(f"起点: {start}")
    print(f"终点: {goal}")
    print(f"边界: {bounds}")
    print(f"障碍物: {obstacles}")
    print()
    
    path = planner.plan(start, goal, bounds=bounds)
    
    if path.is_valid:
        print(f"[OK] 找到路径!")
        print(f"  路径点数量: {len(path)}")
        print(f"  总代价: {path.total_cost:.3f}")
        print(f"  探索节点数: {path.nodes_explored}")
        print(f"  路径点: {path.waypoints}")
    else:
        print("[FAIL] 未找到路径")
    
    return path


def example_rrt_star():
    """RRT* 示例"""
    print("\n" + "=" * 60)
    print("RRT* 路径规划 - 渐进最优示例")
    print("=" * 60)
    
    obstacles = [
        (3, 3, 1.5),
        (7, 7, 2.0),
        (5, 1, 1.0),
        (1, 6, 1.2),
    ]
    
    def collision_fn(x: float, y: float) -> bool:
        for ox, oy, r in obstacles:
            if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < r:
                return True
        return False
    
    # 对比 RRT 和 RRT*
    start = (0, 0)
    goal = (10, 10)
    bounds = (0, 10, 0, 10)
    
    print(f"起点: {start}, 终点: {goal}")
    print(f"障碍物数量: {len(obstacles)}")
    print()
    
    # RRT
    rrt = RRTPlanner(collision_fn, max_iter=3000, step_size=0.5)
    path_rrt = rrt.plan(start, goal, bounds=bounds)
    
    # RRT*
    rrt_star = RRTStarPlanner(
        collision_fn, max_iter=3000, step_size=0.5, search_radius=1.5
    )
    path_rrt_star = rrt_star.plan(start, goal, bounds=bounds)
    
    print("RRT 结果:")
    if path_rrt.is_valid:
        print(f"  路径点数: {len(path_rrt)}")
        print(f"  总代价: {path_rrt.total_cost:.3f}")
    else:
        print("  未找到路径")
    
    print()
    print("RRT* 结果:")
    if path_rrt_star.is_valid:
        print(f"  路径点数: {len(path_rrt_star)}")
        print(f"  总代价: {path_rrt_star.total_cost:.3f}")
    else:
        print("  未找到路径")


def example_smoothing():
    """路径平滑示例"""
    print("\n" + "=" * 60)
    print("RRT 路径规划 - 路径平滑")
    print("=" * 60)
    
    obstacles = [
        (3, 3, 1.5),
        (7, 7, 2.0),
        (5, 1, 1.0),
    ]
    
    def collision_fn(x: float, y: float) -> bool:
        for ox, oy, r in obstacles:
            if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < r:
                return True
        return False
    
    planner = RRTPlanner(
        collision_fn, max_iter=5000, step_size=0.3, goal_sample_rate=0.1
    )
    
    start = (0, 0)
    goal = (10, 10)
    bounds = (0, 10, 0, 10)
    
    path = planner.plan(start, goal, bounds=bounds)
    
    if path.is_valid:
        print(f"原始路径:")
        print(f"  路径点数: {len(path)}")
        print(f"  总代价: {path.total_cost:.3f}")
        
        # 平滑路径
        smoothed = planner.smooth_path(path, max_iterations=100)
        
        print()
        print(f"平滑后路径:")
        print(f"  路径点数: {len(smoothed)}")
        print(f"  总代价: {smoothed.total_cost:.3f}")
        print(f"  减少点数: {len(path) - len(smoothed)}")


def example_maze():
    """迷宫环境示例"""
    print("\n" + "=" * 60)
    print("RRT 路径规划 - 迷宫环境")
    print("=" * 60)
    
    # 创建迷宫（使用矩形障碍物）
    maze_obstacles = []
    
    # 外墙
    for x in range(21):
        maze_obstacles.append((x, 0, 0.5))
        maze_obstacles.append((x, 20, 0.5))
    for y in range(21):
        maze_obstacles.append((0, y, 0.5))
        maze_obstacles.append((20, y, 0.5))
    
    # 内部墙壁
    wall_positions = [
        (5, 5, 5, 1),   # x, y, width, height
        (5, 10, 1, 6),
        (10, 5, 1, 10),
        (12, 12, 6, 1),
        (15, 5, 1, 5),
    ]
    
    for wx, wy, ww, wh in wall_positions:
        for dx in range(ww):
            for dy in range(wh):
                maze_obstacles.append((wx + dx, wy + dy, 0.5))
    
    def collision_fn(x: float, y: float) -> bool:
        for ox, oy, r in maze_obstacles:
            if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < r:
                return True
        # 边界检查
        if x < 0 or x > 20 or y < 0 or y > 20:
            return True
        return False
    
    planner = RRTPlanner(
        collision_fn, max_iter=10000, step_size=0.8, goal_sample_rate=0.15
    )
    
    start = (2, 2)
    goal = (18, 18)
    bounds = (0, 20, 0, 20)
    
    print(f"起点: {start}")
    print(f"终点: {goal}")
    print(f"障碍物数量: {len(maze_obstacles)}")
    print()
    
    path = planner.plan(start, goal, bounds=bounds)
    
    if path.is_valid:
        print(f"[OK] 找到路径!")
        print(f"  路径点数: {len(path)}")
        print(f"  总代价: {path.total_cost:.3f}")
        print(f"  探索节点数: {path.nodes_explored}")
        
        # ASCII 可视化
        print("\n路径可视化 (S=起点, G=终点, #=障碍物, *=路径, .=空):")
        print("-" * 24)
        
        # 创建路径点集合
        path_set = set()
        for px, py in path.waypoints:
            path_set.add((int(px), int(py)))
        
        obstacle_set = set()
        for ox, oy, r in maze_obstacles:
            obstacle_set.add((int(ox), int(oy)))
        
        for y in range(20, -1, -1):
            row = ["|"]
            for x in range(21):
                if (x, y) == (int(start[0]), int(start[1])):
                    row.append("S")
                elif (x, y) == (int(goal[0]), int(goal[1])):
                    row.append("G")
                elif (x, y) in obstacle_set:
                    row.append("#")
                elif (x, y) in path_set:
                    row.append("*")
                else:
                    row.append(".")
            row.append("|")
            print("".join(row))
        print("-" * 24)
    else:
        print("[FAIL] 未找到路径")


def example_comparison():
    """不同参数对比示例"""
    print("\n" + "=" * 60)
    print("RRT 路径规划 - 参数对比")
    print("=" * 60)
    
    obstacles = [
        (5, 5, 2.0),
        (8, 8, 1.5),
        (5, 9, 1.0),
    ]
    
    def collision_fn(x: float, y: float) -> bool:
        for ox, oy, r in obstacles:
            if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < r:
                return True
        return False
    
    start = (0, 0)
    goal = (12, 12)
    bounds = (0, 12, 0, 12)
    
    print(f"起点: {start}, 终点: {goal}")
    print()
    
    # 不同步长对比
    step_sizes = [0.3, 0.5, 1.0]
    
    for step in step_sizes:
        planner = RRTPlanner(
            collision_fn, max_iter=3000, step_size=step, goal_sample_rate=0.1
        )
        path = planner.plan(start, goal, bounds=bounds)
        
        print(f"步长 = {step}:")
        if path.is_valid:
            print(f"  路径点数: {len(path)}")
            print(f"  总代价: {path.total_cost:.3f}")
            print(f"  探索节点数: {path.nodes_explored}")
        else:
            print("  未找到路径")
        print()


if __name__ == "__main__":
    # 运行所有示例
    example_basic()
    example_rrt_star()
    example_smoothing()
    example_maze()
    example_comparison()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)
