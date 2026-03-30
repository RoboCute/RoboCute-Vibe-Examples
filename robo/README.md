# RoboCute 机器人工程 (samples/robo)

基于 RoboCute World API 的完整机器人仿真工程，提供多种底盘类型、路径规划和地图编辑功能。

## 特性

- **多种底盘类型**: 差速、阿克曼、履带、麦轮
- **路径规划**: A*、RRT、Dijkstra 算法
- **地图编辑器**: 创建、编辑、保存/加载地图
- **World API 集成**: 与 RoboCute 实体系统无缝集成
- **无渲染模式**: 纯仿真控制，不依赖渲染窗口

## 模块结构

```
samples/robo/
├── __init__.py       # 模块导出
├── chassis.py        # 底盘运动学模型
├── path_planner.py   # 路径规划算法
├── map_editor.py     # 地图编辑器
├── robot.py          # 机器人主类
├── main.py           # 示例演示
└── README.md         # 本文档
```

## 快速开始

### 基础使用

```python
from samples.robo import (
    Robot, RobotConfig, ChassisType,
    MapEditor, AStarPlanner
)

# 1. 创建地图
map_editor = MapEditor.create_with_walls(
    width=10.0, height=10.0, resolution=0.2
)

# 添加障碍物
map_editor.add_rect_obstacle_world(2.0, 2.0, 1.0, 3.0)
map_editor.add_circle_obstacle(5.0, 5.0, 1.0)

# 2. 创建机器人
config = RobotConfig(
    chassis_type=ChassisType.DIFFERENTIAL,
    wheel_radius=0.05,
    wheel_base=0.3
)
robot = Robot(config)

# 3. 设置规划器
planner = AStarPlanner(map_editor.data, map_editor.resolution)
robot.set_planner(planner)

# 4. 规划并跟随路径
robot.set_pose(1.0, 1.0, 0.0)
start = PathPoint(1.0, 1.0)
goal = PathPoint(8.0, 8.0)
path = planner.plan(start, goal)
robot.follow_path(path)

# 5. 更新循环
dt = 0.1
while not robot.is_at_goal():
    robot.update(dt)
    time.sleep(dt)
```

### 运行示例

```bash
# 运行演示
cd samples/robo
python main.py

# 与 World API 集成 (需要项目路径)
python main.py --project /path/to/project --backend dx
```

## 底盘类型

### 差速底盘 (Differential)

两轮差速驱动，广泛用于移动机器人。

```python
from samples.robo import DifferentialChassis

chassis = DifferentialChassis(
    wheel_radius=0.05,  # 轮子半径 (m)
    wheel_base=0.3      # 轮距 (m)
)

# 设置速度: vx=前进速度, omega=角速度
chassis.set_velocity(vx=0.5, vy=0.0, omega=0.3)

# 更新里程计
chassis.update_odometry(dt=0.1)
print(f"位置: ({chassis.pose.x}, {chassis.pose.y})")
```

### 阿克曼底盘 (Ackermann)

汽车转向模型，前轮转向后轮驱动。

```python
from samples.robo import AckermannChassis

chassis = AckermannChassis(
    wheel_radius=0.05,
    wheel_base=0.3,
    track_width=0.25,
    max_steering_angle=math.pi/4
)

# 设置转向角
chassis.set_steering(math.radians(20))
chassis.set_velocity(vx=0.5, vy=0.0, omega=0.0)
```

### 履带底盘 (Tracked)

坦克模型，两侧履带独立驱动，支持滑动系数。

```python
from samples.robo import TrackedChassis

chassis = TrackedChassis(
    wheel_radius=0.05,
    wheel_base=0.3,
    slip_factor=0.9  # 滑动系数
)
```

### 麦轮底盘 (Mecanum)

麦克纳姆轮全向移动，可实现任意方向平移。

```python
from samples.robo import MecanumChassis

chassis = MecanumChassis(
    wheel_radius=0.05,
    wheel_base=0.3,
    track_width=0.3
)

# 斜向移动
chassis.set_velocity(vx=0.3, vy=0.3, omega=0.0)
```

## 路径规划

### A* 算法

最优路径搜索，适用于已知地图。

```python
from samples.robo import AStarPlanner, PathPoint

planner = AStarPlanner(
    map_data=map_editor.data,
    resolution=0.2,
    diagonal_movement=True  # 允许对角移动
)

start = PathPoint(1.0, 1.0)
goal = PathPoint(8.0, 8.0)

path = planner.plan(start, goal)
if path:
    print(f"路径长度: {path.total_length():.2f} m")
    print(f"路径点数量: {len(path)}")
    
    # 路径平滑
    smooth_path = path.smooth(weight_data=0.5, weight_smooth=0.1)
```

### RRT 算法

快速随机树，适用于复杂障碍物环境。

```python
from samples.robo import RRTPlanner

planner = RRTPlanner(
    map_data=map_editor.data,
    resolution=0.2,
    max_iter=2000,
    step_size=0.3,
    goal_sample_rate=0.1
)

path = planner.plan(start, goal)
```

### Dijkstra 算法

最短路径搜索，保证全局最优。

```python
from samples.robo import DijkstraPlanner

planner = DijkstraPlanner(map_editor.data, resolution=0.2)
path = planner.plan(start, goal)
```

## 地图编辑器

### 创建地图

```python
from samples.robo import MapEditor

# 创建空地图
map_editor = MapEditor.create_empty(
    width=20.0, height=20.0, resolution=0.1
)

# 创建带围墙的地图
map_editor = MapEditor.create_with_walls(
    width=10.0, height=10.0, wall_thickness=0.2
)
```

### 编辑地图

```python
# 添加矩形障碍物
map_editor.add_rect_obstacle_world(x=2.0, y=2.0, width=1.0, height=3.0)

# 添加圆形障碍物
map_editor.add_circle_obstacle(center_x=5.0, center_y=5.0, radius=1.0)

# 单个格子操作
map_editor.set_obstacle(mx=10, my=10)
map_editor.set_free(mx=10, my=10)

# 检查格子
is_obs = map_editor.is_occupied(mx=10, my=10)
is_obs_world = map_editor.is_occupied_world(x=1.0, y=1.0)
```

### 地图膨胀

```python
# 对障碍物进行膨胀，生成代价地图
map_editor.inflate(radius=0.3, cost_scaling=2.0)

# 获取代价地图
cost_map = map_editor.cost_map  # 0.0 - 1.0，越高越难通行
```

### 保存/加载

```python
# 保存为 JSON
map_editor.save("my_map.json")

# 保存为 NumPy
map_editor.save("my_map.npy")

# 加载
map_editor = MapEditor.load("my_map.json")
```

### 坐标转换

```python
# 世界坐标 -> 地图坐标
mx, my = map_editor.world_to_map(x=1.0, y=1.0)

# 地图坐标 -> 世界坐标 (格子中心)
x, y = map_editor.map_to_world(mx=10, my=10)
```

## 机器人控制

### 基本控制

```python
from samples.robo import Robot, RobotConfig, ChassisType

config = RobotConfig(
    chassis_type=ChassisType.DIFFERENTIAL,
    wheel_radius=0.05,
    wheel_base=0.3,
    max_linear_speed=1.0,
    max_angular_speed=1.0
)

robot = Robot(config)

# 设置位姿
robot.set_pose(x=1.0, y=1.0, theta=0.0)

# 设置速度
robot.set_velocity(vx=0.5, vy=0.0, omega=0.0)

# 停止
robot.stop()
```

### 路径跟随

```python
# 规划并跟随路径
start = PathPoint(1.0, 1.0)
goal = PathPoint(8.0, 8.0)
path = planner.plan(start, goal)
robot.follow_path(path)

# 跟随指定路径
robot.follow_path(path)

# 更新循环
dt = 0.1
while not robot.is_at_goal():
    robot.update(dt)
    time.sleep(dt)

# 检查状态
progress = robot.get_path_progress()  # 0.0 - 1.0
remaining = robot.get_remaining_distance()  # 剩余距离 (m)
is_reached = robot.is_at_goal()
```

### 事件回调

```python
def on_goal_reached():
    print("到达目标!")

def on_path_complete():
    print("路径完成!")

robot.register_callback('on_goal_reached', on_goal_reached)
robot.register_callback('on_path_complete', on_path_complete)
```

### World 集成

```python
import robocute as rbc
import robocute.rbc_ext as re

# 初始化应用
app = rbc.app.App()
app.init(project_path=Path("..."), backend_name="dx")

# 创建机器人在 World 中
robot.create_in_world(
    scene=app.scene,
    x=1.0, y=1.0, theta=0.0
)

# 此时机器人的 TransformComponent 会自动更新
```

## 机器人群组

```python
from samples.robo import RobotFleet

fleet = RobotFleet()

# 添加机器人
robot1 = Robot(RobotConfig(chassis_type=ChassisType.DIFFERENTIAL))
robot2 = Robot(RobotConfig(chassis_type=ChassisType.MECANUM))

fleet.add_robot(robot1)
fleet.add_robot(robot2)

# 批量控制
fleet.update_all(dt=0.1)
fleet.stop_all()

# 分配任务
fleet.assign_task("r1", target_x=5.0, target_y=5.0)
fleet.assign_task("r2", target_x=3.0, target_y=7.0)

# 获取所有位姿
poses = fleet.get_all_poses()
```

## API 参考

### 底盘模块 (chassis.py)

| 类 | 说明 |
|----|------|
| `ChassisBase` | 底盘基类 |
| `DifferentialChassis` | 差速底盘 |
| `AckermannChassis` | 阿克曼底盘 |
| `TrackedChassis` | 履带底盘 |
| `MecanumChassis` | 麦轮底盘 |
| `Pose2D` | 2D位姿 (x, y, theta) |
| `Velocity` | 速度 (vx, vy, omega) |

### 路径规划模块 (path_planner.py)

| 类 | 说明 |
|----|------|
| `PathPlanner` | 规划器基类 |
| `AStarPlanner` | A* 规划器 |
| `RRTPlanner` | RRT 规划器 |
| `DijkstraPlanner` | Dijkstra 规划器 |
| `Path` | 路径类 |
| `PathPoint` | 路径点 |

### 地图编辑器模块 (map_editor.py)

| 类 | 说明 |
|----|------|
| `MapEditor` | 地图编辑器 |
| `MapCell` | 地图单元格 |
| `CellType` | 格子类型枚举 |
| `MapMetadata` | 地图元数据 |

### 机器人模块 (robot.py)

| 类 | 说明 |
|----|------|
| `Robot` | 机器人主类 |
| `RobotConfig` | 机器人配置 |
| `RobotFleet` | 机器人群组 |

## 依赖

- Python >= 3.8
- NumPy
- RoboCute (可选，用于 World API 集成)

## 注意事项

1. **无渲染模式**: 本工程设计为不依赖渲染窗口，适合后台仿真
2. **World API**: 与实体系统集成时会自动同步 TransformComponent
3. **碰撞检测**: 路径规划器支持自定义碰撞检测函数
4. **坐标系**: 使用右手坐标系，theta 为绕 Z 轴旋转角度
