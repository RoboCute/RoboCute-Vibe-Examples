# Robot Chassis Simulation (机器人底盘仿真)

基于物理驱动的机器人底盘仿真系统，支持四种常见底盘类型和多种路径规划算法。

## 功能特性

### 底盘类型

1. **差速底盘 (Differential Drive)**
   - 双驱动轮 + 万向轮
   - 速度控制接口
   - 原地转向能力

2. **阿克曼底盘 (Ackermann Steering)**
   - 真实车辆转向几何
   - 转向角限制
   - 最小转弯半径计算

3. **履带底盘 (Tracked Vehicle)**
   - 双履带控制
   - 地形适应性模型
   - 地面压力分布

4. **麦克纳姆轮底盘 (Mecanum Wheel)**
   - 全向移动控制
   - 速度解算
   - 侧向移动能力

### 路径规划算法

- **A\*算法**: 最优栅格路径规划
- **RRT/RRT\***: 快速随机树（支持渐进最优）
- **DWA**: 动态窗口法（适用于动态避障）
- **Pure Pursuit**: 纯跟踪控制器

### 地图系统

- **栅格地图 (Occupancy Grid)**: 静态障碍物表示
- **成本地图 (Costmap)**: 多层成本计算（静态、动态、膨胀层）
- **障碍物类型**: 圆形、矩形、多边形、动态障碍物

### RBC 节点

```
ObstacleMapNode → PathPlanningNode → PathFollowingNode → UIPCSimNode → AnimationOutput
                                    ↓
                            RobotChassisNode
```

1. `RobotChassisNode`: 创建和配置底盘模型
2. `ObstacleMapNode`: 创建栅格地图和障碍物
3. `PathPlanningNode`: 路径规划
4. `PathFollowingNode`: 路径跟踪控制
5. `TrajectoryVisualizerNode`: 轨迹可视化
6. `ChassisSimulatorNode`: 底盘物理仿真
7. `MultiChassisComparisonNode`: 多底盘性能对比

## 安装

确保已安装 RoboCute 和相关依赖：

```bash
# 安装依赖
pip install numpy

# 确保 RoboCute 在 Python 路径中
export PYTHONPATH="${PYTHONPATH}:D:/RoboCute/src"
```

## 快速开始

### 命令行演示

```bash
# 运行所有演示
python -m samples.robot.demo

# 运行特定演示
python -m samples.robot.demo --demo single --chassis differential_drive
python -m samples.robot.demo --demo planning
python -m samples.robot.demo --demo chassis
python -m samples.robot.demo --demo nodes

# 运行示例场景
python -m samples.robot.demo --demo warehouse
python -m samples.robot.demo --demo offroad
python -m samples.robot.demo --demo multi
```

### 编程使用

```python
from robot import (
    ChassisType, ChassisConfig, DifferentialDrive,
    OccupancyGrid, CircularObstacle,
    AStarPlanner, PurePursuitController
)

# 创建底盘
config = ChassisConfig()
chassis = DifferentialDrive(config)

# 创建地图
grid = OccupancyGrid(10, 10, 0.1, (-5, -5))
obstacle = CircularObstacle(id=0, position=(2, 2), radius=0.5)
grid.set_obstacle(obstacle)

# 路径规划
collision_fn = lambda x, y: grid.check_collision(x, y)
planner = AStarPlanner(collision_fn, resolution=0.1)
path = planner.plan((-3, -3), (3, 3))

# 路径跟踪
controller = PurePursuitController(lookahead_distance=0.5, 
                                   wheelbase=config.wheelbase)
```

## 示例场景

### 场景1: 仓库导航 (`scene_warehouse_navigation.py`)

- 货架障碍物布局
- 多个拣货点
- 路径优化
- 多底盘对比

### 场景2: 越野地形 (`scene_offroad_terrain.py`)

- 不平整地面
- 斜坡和台阶
- 地形适应性模型
- 底盘性能对比

### 场景3: 多机器人协同 (`scene_multi_robot.py`)

- 多底盘同时仿真
- 碰撞避免
- 任务分配
- 优先级协调

## 文件结构

```
samples/robot/
├── __init__.py              # 包初始化
├── chassis.py               # 底盘运动学和物理模型
├── path_planning.py         # 路径规划算法
├── map_system.py            # 地图和障碍物系统
├── visualization.py         # 可视化工具
├── nodes.py                 # RBC 节点实现
├── mock_nodes.py            # 兼容模块（无 robocute 时使用）
├── demo.py                  # 主演示程序
├── scene_warehouse_navigation.py   # 仓库导航场景
├── scene_offroad_terrain.py        # 越野地形场景
├── scene_multi_robot.py            # 多机器人协同场景
└── README.md                # 本文件
```

## API 参考

### ChassisBase (底盘基类)

```python
class ChassisBase:
    def apply_control(self, control_input: np.ndarray, dt: float)
    def get_transform_matrix(self) -> np.ndarray
    def local_to_world(self, local_pos: np.ndarray) -> np.ndarray
    def get_wheel_positions(self) -> List[np.ndarray]
```

### PathPlanner (路径规划器)

```python
class PathPlanner:
    def plan(self, start: Tuple[float, float], 
             goal: Tuple[float, float], **kwargs) -> Path
```

### OccupancyGrid (栅格地图)

```python
class OccupancyGrid:
    def set_obstacle(self, obstacle: Obstacle)
    def check_collision(self, x: float, y: float, radius: float = 0) -> bool
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]
```

## 开发计划

根据 v03.md 的开发计划，本实现已完成以下功能：

- [x] 差速底盘实现
- [x] 阿克曼底盘实现
- [x] 履带底盘实现
- [x] 麦克纳姆轮底盘实现
- [x] A* 路径规划
- [x] RRT/RRT* 路径规划
- [x] DWA 动态窗口法
- [x] Pure Pursuit 跟踪
- [x] 栅格地图系统
- [x] 成本地图系统
- [x] 障碍物系统
- [x] RBC 节点系统
- [x] 可视化工具
- [x] 示例场景

## 许可

本项目遵循 RoboCute 项目的许可证。
