# -*- coding: utf-8 -*-
"""
Path Planning Algorithms

路径规划算法实现
包括 A*, RRT, DWA 和 Pure Pursuit
"""

import numpy as np
import heapq
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set, Dict, Callable
from dataclasses import dataclass, field


@dataclass
class Path:
    """路径数据结构"""
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    total_cost: float = 0.0
    is_valid: bool = False
    
    def __len__(self):
        return len(self.waypoints)
    
    def get_waypoint(self, index: int) -> Optional[Tuple[float, float]]:
        """获取指定索引的路径点"""
        if 0 <= index < len(self.waypoints):
            return self.waypoints[index]
        return None
    
    def get_remaining_distance(self, current_index: int) -> float:
        """获取从当前索引到终点的剩余距离"""
        if current_index >= len(self.waypoints) - 1:
            return 0.0
        
        distance = 0.0
        for i in range(current_index, len(self.waypoints) - 1):
            p1 = np.array(self.waypoints[i])
            p2 = np.array(self.waypoints[i + 1])
            distance += np.linalg.norm(p2 - p1)
        return distance
    
    def smooth(self, weight_data: float = 0.5, weight_smooth: float = 0.3, 
               tolerance: float = 0.00001) -> 'Path':
        """
        使用梯度下降法平滑路径
        
        Args:
            weight_data: 数据权重
            weight_smooth: 平滑权重
            tolerance: 收敛容差
            
        Returns:
            平滑后的路径
        """
        if len(self.waypoints) < 3:
            return Path(waypoints=self.waypoints.copy(), is_valid=self.is_valid)
        
        new_waypoints = [np.array(p) for p in self.waypoints]
        change = tolerance
        
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(new_waypoints) - 1):
                old_pos = new_waypoints[i].copy()
                
                # 梯度下降更新
                new_waypoints[i] += weight_data * (np.array(self.waypoints[i]) - new_waypoints[i])
                new_waypoints[i] += weight_smooth * (new_waypoints[i-1] + new_waypoints[i+1] 
                                                      - 2 * new_waypoints[i])
                
                change += np.linalg.norm(new_waypoints[i] - old_pos)
        
        smoothed_waypoints = [(p[0], p[1]) for p in new_waypoints]
        return Path(waypoints=smoothed_waypoints, is_valid=self.is_valid)


class PathPlanner(ABC):
    """路径规划器基类"""
    
    def __init__(self, collision_check_fn: Callable[[float, float], bool]):
        """
        初始化路径规划器
        
        Args:
            collision_check_fn: 碰撞检测函数，输入 (x, y) 返回是否碰撞
        """
        self.collision_check = collision_check_fn
        
    @abstractmethod
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], 
             **kwargs) -> Path:
        """
        规划路径
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            **kwargs: 额外参数
            
        Returns:
            规划的路径
        """
        pass
    
    def is_valid_point(self, x: float, y: float) -> bool:
        """检查点是否有效（无碰撞）"""
        return not self.collision_check(x, y)
    
    def is_valid_path_segment(self, p1: Tuple[float, float], 
                               p2: Tuple[float, float],
                               step_size: float = 0.1) -> bool:
        """
        检查路径段是否有效（无碰撞）
        
        Args:
            p1: 起点
            p2: 终点
            step_size: 检查步长
            
        Returns:
            是否有效
        """
        p1_arr = np.array(p1)
        p2_arr = np.array(p2)
        distance = np.linalg.norm(p2_arr - p1_arr)
        
        if distance < 1e-6:
            return self.is_valid_point(p1[0], p1[1])
        
        num_steps = max(int(distance / step_size), 1)
        direction = (p2_arr - p1_arr) / distance
        
        for i in range(num_steps + 1):
            t = i / num_steps
            point = p1_arr + t * distance * direction
            if not self.is_valid_point(point[0], point[1]):
                return False
        
        return True


class AStarPlanner(PathPlanner):
    """
    A* 路径规划算法
    
    使用栅格地图进行路径规划
    """
    
    def __init__(self, collision_check_fn: Callable[[float, float], bool],
                 resolution: float = 0.1):
        """
        初始化 A* 规划器
        
        Args:
            collision_check_fn: 碰撞检测函数
            resolution: 栅格分辨率 (m)
        """
        super().__init__(collision_check_fn)
        self.resolution = resolution
        
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        return (int(round(x / self.resolution)), 
                int(round(y / self.resolution)))
    
    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        return (gx * self.resolution, gy * self.resolution)
    
    def _heuristic(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """
        启发式函数（欧几里得距离）
        """
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        """
        获取邻居节点
        
        Returns:
            邻居列表 [(gx, gy, cost), ...]
        """
        gx, gy = pos
        neighbors = []
        
        # 8个方向
        directions = [
            (0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),  # 4-邻接
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), 
            (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))  # 对角
        ]
        
        for dx, dy, cost in directions:
            ng_x, ng_y = gx + dx, gy + dy
            world_x, world_y = self._grid_to_world(ng_x, ng_y)
            if self.is_valid_point(world_x, world_y):
                neighbors.append((ng_x, ng_y, cost))
        
        return neighbors
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             **kwargs) -> Path:
        """
        A* 路径规划
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            
        Returns:
            规划的路径
        """
        start_grid = self._world_to_grid(start[0], start[1])
        goal_grid = self._world_to_grid(goal[0], goal[1])
        
        # 检查起点和终点
        if not self.is_valid_point(start[0], start[1]):
            return Path(is_valid=False)
        if not self.is_valid_point(goal[0], goal[1]):
            return Path(is_valid=False)
        
        # A* 算法
        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start_grid))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_grid: 0.0}
        f_score: Dict[Tuple[int, int], float] = {start_grid: self._heuristic(start_grid, goal_grid)}
        
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            # 到达目标
            if current == goal_grid:
                path = self._reconstruct_path(came_from, current)
                return Path(waypoints=path, is_valid=True, total_cost=g_score[current])
            
            closed_set.add(current)
            
            for neighbor, cost in [(n[:2], n[2]) for n in self._get_neighbors(current)]:
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # 未找到路径
        return Path(is_valid=False)
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                          current: Tuple[int, int]) -> List[Tuple[float, float]]:
        """重建路径"""
        path_grid = [current]
        while current in came_from:
            current = came_from[current]
            path_grid.append(current)
        path_grid.reverse()
        
        # 转换为世界坐标
        return [self._grid_to_world(gx, gy) for gx, gy in path_grid]


class RRTPlanner(PathPlanner):
    """
    RRT (Rapidly-exploring Random Tree) 路径规划算法
    
    适合高维空间和复杂约束
    """
    
    def __init__(self, collision_check_fn: Callable[[float, float], bool],
                 max_iter: int = 5000,
                 step_size: float = 0.2,
                 goal_sample_rate: float = 0.1):
        """
        初始化 RRT 规划器
        
        Args:
            collision_check_fn: 碰撞检测函数
            max_iter: 最大迭代次数
            step_size: 扩展步长
            goal_sample_rate: 目标采样概率
        """
        super().__init__(collision_check_fn)
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        
    class Node:
        """RRT 树节点"""
        def __init__(self, x: float, y: float):
            self.x = x
            self.y = y
            self.parent: Optional['RRTPlanner.Node'] = None
            self.cost: float = 0.0
            
        def get_position(self) -> Tuple[float, float]:
            return (self.x, self.y)
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             bounds: Optional[Tuple[float, float, float, float]] = None,
             **kwargs) -> Path:
        """
        RRT 路径规划
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            bounds: 搜索范围 (min_x, max_x, min_y, max_y)
            
        Returns:
            规划的路径
        """
        if bounds is None:
            # 默认范围
            min_x = min(start[0], goal[0]) - 5
            max_x = max(start[0], goal[0]) + 5
            min_y = min(start[1], goal[1]) - 5
            max_y = max(start[1], goal[1]) + 5
        else:
            min_x, max_x, min_y, max_y = bounds
        
        # 检查起点和终点
        if not self.is_valid_point(start[0], start[1]):
            return Path(is_valid=False)
        if not self.is_valid_point(goal[0], goal[1]):
            return Path(is_valid=False)
        
        start_node = self.Node(start[0], start[1])
        goal_node = self.Node(goal[0], goal[1])
        
        nodes = [start_node]
        
        for _ in range(self.max_iter):
            # 随机采样
            if np.random.random() < self.goal_sample_rate:
                rnd_x, rnd_y = goal[0], goal[1]
            else:
                rnd_x = np.random.uniform(min_x, max_x)
                rnd_y = np.random.uniform(min_y, max_y)
            
            # 找到最近的节点
            nearest_node = self._get_nearest_node(nodes, rnd_x, rnd_y)
            
            # 扩展
            new_node = self._steer(nearest_node, rnd_x, rnd_y)
            
            if new_node and self.is_valid_path_segment(
                nearest_node.get_position(), new_node.get_position()):
                nodes.append(new_node)
                
                # 检查是否到达目标
                dist_to_goal = math.sqrt((new_node.x - goal[0])**2 + 
                                          (new_node.y - goal[1])**2)
                if dist_to_goal < self.step_size:
                    # 尝试连接到目标
                    if self.is_valid_path_segment(new_node.get_position(), goal):
                        goal_node.parent = new_node
                        path = self._extract_path(goal_node)
                        return Path(waypoints=path, is_valid=True)
        
        # 未找到路径
        return Path(is_valid=False)
    
    def _get_nearest_node(self, nodes: List['RRTPlanner.Node'], 
                          x: float, y: float) -> 'RRTPlanner.Node':
        """找到最近的节点"""
        distances = [(n.x - x)**2 + (n.y - y)**2 for n in nodes]
        return nodes[np.argmin(distances)]
    
    def _steer(self, from_node: 'RRTPlanner.Node', to_x: float, 
               to_y: float) -> Optional['RRTPlanner.Node']:
        """向目标方向扩展"""
        dx = to_x - from_node.x
        dy = to_y - from_node.y
        distance = math.sqrt(dx**2 + dy**2)
        
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
        
        new_node = self.Node(new_x, new_y)
        new_node.parent = from_node
        new_node.cost = from_node.cost + math.sqrt((new_x - from_node.x)**2 + 
                                                    (new_y - from_node.y)**2)
        return new_node
    
    def _extract_path(self, goal_node: 'RRTPlanner.Node') -> List[Tuple[float, float]]:
        """提取路径"""
        path = []
        node = goal_node
        while node is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()
        return path


class RRTStarPlanner(RRTPlanner):
    """
    RRT* 算法（渐进最优的 RRT）
    """
    
    def __init__(self, collision_check_fn: Callable[[float, float], bool],
                 max_iter: int = 5000,
                 step_size: float = 0.2,
                 goal_sample_rate: float = 0.1,
                 search_radius: float = 0.5):
        super().__init__(collision_check_fn, max_iter, step_size, goal_sample_rate)
        self.search_radius = search_radius
        
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             bounds: Optional[Tuple[float, float, float, float]] = None,
             **kwargs) -> Path:
        """RRT* 路径规划"""
        if bounds is None:
            min_x = min(start[0], goal[0]) - 5
            max_x = max(start[0], goal[0]) + 5
            min_y = min(start[1], goal[1]) - 5
            max_y = max(start[1], goal[1]) + 5
        else:
            min_x, max_x, min_y, max_y = bounds
        
        if not self.is_valid_point(start[0], start[1]):
            return Path(is_valid=False)
        if not self.is_valid_point(goal[0], goal[1]):
            return Path(is_valid=False)
        
        start_node = self.Node(start[0], start[1])
        goal_node = self.Node(goal[0], goal[1])
        
        nodes = [start_node]
        best_path = None
        best_cost = float('inf')
        
        for _ in range(self.max_iter):
            # 随机采样
            if np.random.random() < self.goal_sample_rate:
                rnd_x, rnd_y = goal[0], goal[1]
            else:
                rnd_x = np.random.uniform(min_x, max_x)
                rnd_y = np.random.uniform(min_y, max_y)
            
            # 找到最近的节点
            nearest_node = self._get_nearest_node(nodes, rnd_x, rnd_y)
            
            # 扩展
            new_node = self._steer(nearest_node, rnd_x, rnd_y)
            
            if new_node and self.is_valid_path_segment(
                nearest_node.get_position(), new_node.get_position()):
                
                # 找到搜索半径内的邻居
                neighbors = self._find_neighbors(nodes, new_node)
                
                # 选择最优父节点
                best_parent = nearest_node
                best_parent_cost = nearest_node.cost + self._distance(nearest_node, new_node)
                
                for neighbor in neighbors:
                    cost = neighbor.cost + self._distance(neighbor, new_node)
                    if cost < best_parent_cost and self.is_valid_path_segment(
                        neighbor.get_position(), new_node.get_position()):
                        best_parent = neighbor
                        best_parent_cost = cost
                
                new_node.parent = best_parent
                new_node.cost = best_parent_cost
                nodes.append(new_node)
                
                # 重布线
                for neighbor in neighbors:
                    new_cost = new_node.cost + self._distance(new_node, neighbor)
                    if new_cost < neighbor.cost and self.is_valid_path_segment(
                        new_node.get_position(), neighbor.get_position()):
                        neighbor.parent = new_node
                        neighbor.cost = new_cost
                
                # 检查是否到达目标
                dist_to_goal = self._distance(new_node, goal_node)
                if dist_to_goal < self.step_size:
                    if self.is_valid_path_segment(new_node.get_position(), goal):
                        goal_node.parent = new_node
                        goal_node.cost = new_node.cost + dist_to_goal
                        if goal_node.cost < best_cost:
                            best_cost = goal_node.cost
                            best_path = self._extract_path(goal_node)
        
        if best_path:
            return Path(waypoints=best_path, is_valid=True, total_cost=best_cost)
        return Path(is_valid=False)
    
    def _find_neighbors(self, nodes: List['RRTPlanner.Node'], 
                        new_node: 'RRTPlanner.Node') -> List['RRTPlanner.Node']:
        """找到搜索半径内的邻居"""
        neighbors = []
        for node in nodes:
            if self._distance(node, new_node) < self.search_radius:
                neighbors.append(node)
        return neighbors
    
    def _distance(self, n1: 'RRTPlanner.Node', n2: 'RRTPlanner.Node') -> float:
        """计算两节点距离"""
        return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)


class DWAPlanner(PathPlanner):
    """
    DWA (Dynamic Window Approach) 动态窗口法
    
    适用于动态避障的局部路径规划
    """
    
    def __init__(self, collision_check_fn: Callable[[float, float], bool],
                 max_linear_velocity: float = 1.0,
                 max_angular_velocity: float = 2.0,
                 max_linear_accel: float = 2.0,
                 max_angular_accel: float = 4.0,
                 dt: float = 0.1,
                 predict_time: float = 2.0,
                 velocity_resolution: int = 20,
                 omega_resolution: int = 20):
        """
        初始化 DWA 规划器
        
        Args:
            collision_check_fn: 碰撞检测函数
            max_linear_velocity: 最大线速度
            max_angular_velocity: 最大角速度
            max_linear_accel: 最大线加速度
            max_angular_accel: 最大角加速度
            dt: 时间步长
            predict_time: 预测时间
            velocity_resolution: 速度采样分辨率
            omega_resolution: 角速度采样分辨率
        """
        super().__init__(collision_check_fn)
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.max_linear_accel = max_linear_accel
        self.max_angular_accel = max_angular_accel
        self.dt = dt
        self.predict_time = predict_time
        self.velocity_resolution = velocity_resolution
        self.omega_resolution = omega_resolution
        
        # 评价函数权重
        self.weight_goal = 1.0
        self.weight_speed = 0.5
        self.weight_obstacle = 2.0
    
    def plan(self, start: Tuple[float, float, float],  # (x, y, theta)
             goal: Tuple[float, float],
             current_velocity: Tuple[float, float] = (0, 0),  # (v, omega)
             **kwargs) -> Path:
        """
        DWA 路径规划
        
        Args:
            start: 当前位姿 (x, y, theta)
            goal: 目标位置 (x, y)
            current_velocity: 当前速度 (v, omega)
            
        Returns:
            规划的路径
        """
        x, y, theta = start
        v, omega = current_velocity
        
        # 计算动态窗口
        dw = self._calculate_dynamic_window(v, omega)
        
        # 采样并评估
        best_score = -float('inf')
        best_trajectory = None
        best_control = (0, 0)
        
        for tv in np.linspace(dw[0], dw[1], self.velocity_resolution):
            for to in np.linspace(dw[2], dw[3], self.omega_resolution):
                trajectory = self._predict_trajectory(x, y, theta, tv, to)
                score = self._evaluate_trajectory(trajectory, goal)
                
                if score > best_score:
                    best_score = score
                    best_trajectory = trajectory
                    best_control = (tv, to)
        
        if best_trajectory:
            waypoints = [(p[0], p[1]) for p in best_trajectory]
            return Path(waypoints=waypoints, is_valid=True)
        
        return Path(is_valid=False)
    
    def _calculate_dynamic_window(self, v: float, omega: float) -> Tuple[float, float, float, float]:
        """
        计算动态窗口
        
        Returns:
            (min_v, max_v, min_omega, max_omega)
        """
        # 速度限制
        vs = [0, self.max_linear_velocity, -self.max_angular_velocity, 
              self.max_angular_velocity]
        
        # 加速度限制
        vd = [
            v - self.max_linear_accel * self.dt,
            v + self.max_linear_accel * self.dt,
            omega - self.max_angular_accel * self.dt,
            omega + self.max_angular_accel * self.dt
        ]
        
        # 动态窗口为交集
        dw = [
            max(vs[0], vd[0]),
            min(vs[1], vd[1]),
            max(vs[2], vd[2]),
            min(vs[3], vd[3])
        ]
        
        return tuple(dw)
    
    def _predict_trajectory(self, x: float, y: float, theta: float,
                           v: float, omega: float) -> List[Tuple[float, float, float]]:
        """预测轨迹"""
        trajectory = [(x, y, theta)]
        
        for _ in range(int(self.predict_time / self.dt)):
            x += v * math.cos(theta) * self.dt
            y += v * math.sin(theta) * self.dt
            theta += omega * self.dt
            trajectory.append((x, y, theta))
            
            # 检查碰撞
            if self.collision_check(x, y):
                break
        
        return trajectory
    
    def _evaluate_trajectory(self, trajectory: List[Tuple[float, float, float]],
                            goal: Tuple[float, float]) -> float:
        """评估轨迹"""
        if not trajectory:
            return -float('inf')
        
        final_x, final_y, final_theta = trajectory[-1]
        
        # 目标接近度
        dist_to_goal = math.sqrt((final_x - goal[0])**2 + (final_y - goal[1])**2)
        goal_score = 1.0 / (1.0 + dist_to_goal)
        
        # 速度评分
        # 假设轨迹由 (x, y, theta) 组成，无法直接获取速度
        # 简化处理：轨迹越长越好（表示走得远）
        speed_score = len(trajectory) * self.dt / self.predict_time
        
        # 障碍物评分
        obstacle_score = float('inf')
        for x, y, _ in trajectory:
            if self.collision_check(x, y):
                obstacle_score = 0
                break
            # 可以添加到障碍物的距离计算
        
        if obstacle_score == float('inf'):
            obstacle_score = 1.0
        
        # 综合评分
        total_score = (self.weight_goal * goal_score + 
                      self.weight_speed * speed_score + 
                      self.weight_obstacle * obstacle_score)
        
        return total_score


class PurePursuitController:
    """
    Pure Pursuit 路径跟踪控制器
    
    用于跟踪参考路径
    """
    
    def __init__(self, lookahead_distance: float = 0.5,
                 wheelbase: float = 0.5,
                 max_linear_velocity: float = 1.0):
        """
        初始化 Pure Pursuit 控制器
        
        Args:
            lookahead_distance: 前瞻距离
            wheelbase: 轴距
            max_linear_velocity: 最大线速度
        """
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_linear_velocity = max_linear_velocity
    
    def compute_control(self, current_pose: Tuple[float, float, float],
                       path: Path,
                       current_idx: int = 0,
                       target_velocity: float = None) -> Tuple[float, float, int]:
        """
        计算控制指令
        
        Args:
            current_pose: 当前位姿 (x, y, theta)
            path: 参考路径
            current_idx: 当前在路径上的索引
            target_velocity: 目标速度
            
        Returns:
            (v, omega, target_idx) 线速度、角速度、目标索引
        """
        if not path.is_valid or len(path) == 0:
            return 0.0, 0.0, current_idx
        
        x, y, theta = current_pose
        
        # 找到前瞻点
        target_idx = self._find_lookahead_point(path, x, y, current_idx)
        target_point = path.get_waypoint(target_idx)
        
        if target_point is None:
            return 0.0, 0.0, current_idx
        
        # 计算到目标点的距离和角度
        dx = target_point[0] - x
        dy = target_point[1] - y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < 0.01:  # 到达目标点
            return 0.0, 0.0, target_idx
        
        # 计算目标角度
        target_angle = math.atan2(dy, dx)
        alpha = target_angle - theta
        
        # 归一化角度
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        
        # 计算曲率
        if abs(alpha) < 1e-6:
            curvature = 0
        else:
            curvature = 2 * math.sin(alpha) / distance
        
        # 计算控制量
        if target_velocity is None:
            v = self.max_linear_velocity
        else:
            v = target_velocity
        
        omega = v * curvature
        
        return v, omega, target_idx
    
    def _find_lookahead_point(self, path: Path, x: float, y: float,
                              start_idx: int) -> int:
        """找到前瞻点"""
        for i in range(start_idx, len(path)):
            waypoint = path.get_waypoint(i)
            if waypoint is None:
                continue
            dist = math.sqrt((waypoint[0] - x)**2 + (waypoint[1] - y)**2)
            if dist >= self.lookahead_distance:
                return i
        
        # 如果没有找到，返回最后一个点
        return len(path) - 1
