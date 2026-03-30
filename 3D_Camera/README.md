# 3D Camera Follow Project

A comprehensive implementation of 3D camera following algorithms for the RoboCute rendering engine, demonstrating various camera control techniques commonly used in games and 3D applications.

## Project Structure

```
3D_Camera/
├── README.md              # This file
├── __init__.py            # Package initialization
├── camera_math.py         # Core math utilities (Vector3, interpolation)
├── camera_controller.py   # Camera controller with follow algorithms
├── app_3d_camera.py       # Full-featured demo application
├── simple_demo.py         # Minimal example for quick start
└── test_camera.py         # Unit tests
```

## Features

### 1. Camera Follow Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Smooth Follow** | Critically damped spring system | General 3D games |
| **Spring Follow** | Configurable spring-damped system | Racing/flying games |
| **Predictive Follow** | Look-ahead based on target velocity | Fast-paced action games |
| **Orbital Follow** | Camera orbits around target at fixed distance | Action/inspection cameras |

### 2. Core Algorithms

#### Smooth Damping (Critically Damped Spring)
```python
# Uses a critically damped spring system for smooth, non-oscillating motion
omega = 2.0 / smooth_time
spring_accel = displacement * (omega * omega)
damping_accel = velocity * (-2.0 * omega)
acceleration = spring_accel + damping_accel

# Semi-implicit Euler integration
new_velocity = velocity + acceleration * delta_time
new_position = current + new_velocity * delta_time
```

#### Linear Interpolation
```python
# Simple lerp for quick transitions
t = clamp(smooth_speed * delta_time, 0, 1)
new_pos = current + (target - current) * t
```

#### Spring Physics
```python
# Full spring system for bouncy, physical motion
F_spring = -stiffness * displacement
F_damping = -damping * velocity
F_total = F_spring + F_damping
acceleration = F_total / mass
```

## Quick Start

### Running the Simple Demo

```bash
cd samples/3D_Camera
python simple_demo.py
```

This launches a minimal demo with:
- A moving ball (circular motion)
- Smooth-follow camera
- Manual camera control support

### Running the Full Demo

```bash
python app_3d_camera.py
```

Full demo includes:
- Multiple follow modes (F1 to cycle)
- Multiple moving targets
- Collision avoidance
- Ground plane and visual markers

### Running Tests

```bash
python test_camera.py
```

## Usage Guide

### Basic Camera Setup

```python
import robocute as rbc
from camera_controller import CameraController, FollowMode

# Initialize app
app = rbc.app.App()
app.init(project_path=path, backend_name="dx")
app.init_display(1280, 720)

# Create camera controller
camera = CameraController(app)
camera.initialize()

# Set target to follow
target_entity = scene.add_entity()
camera.set_target(target_entity)

# Choose follow mode
camera.set_mode(FollowMode.SMOOTH)

# In your game loop:
camera.update(delta_time)
```

### Using Core Math Only

```python
from camera_math import Vector3, smooth_damp, lerp

# Vector operations
v1 = Vector3(1, 2, 3)
v2 = Vector3(4, 5, 6)
v3 = v1 + v2  # Vector addition
length = v3.length()
unit = v3.normalized()

# Smooth interpolation
current = Vector3(0, 0, 0)
target = Vector3(10, 0, 0)
velocity = Vector3(0, 0, 0)

# Each frame:
current, velocity = smooth_damp(current, target, velocity, smooth_time=0.3, delta_time=0.016)
```

### Configuration Options

```python
# Offset from target (default: behind and above)
camera.offset = Vector3(0, 5, -10)

# Follow speeds
camera.position_smooth_speed = 5.0  # Higher = faster follow
camera.rotation_smooth_speed = 5.0

# Distance limits
camera.min_distance = 2.0
camera.max_distance = 50.0

# Spring parameters (for SPRING mode)
camera.spring_stiffness = 150.0
camera.spring_damping = 10.0
camera.spring_mass = 1.0

# Predictive follow (for PREDICTIVE mode)
camera.look_ahead_factor = 0.3  # Seconds to predict ahead

# Orbital follow (for ORBITAL mode)
camera.orbit_distance = 10.0
camera.orbit_height = 5.0
camera.orbit_speed = 1.0  # radians per second
```

## Implementation Details

### Vector3 Class

Custom 3D vector implementation with full math operations:

```python
v1 = Vector3(1, 2, 3)
v2 = Vector3(4, 5, 6)

# Operations
v3 = v1 + v2        # Addition
v4 = v2 - v1        # Subtraction
v5 = v1 * 2.0       # Scalar multiplication
length = v1.length()  # Magnitude
unit = v1.normalized()  # Unit vector

# Conversion to RoboCute types (when using camera_controller)
luisa_vec = v1.to_luisa()  # Returns lc.double3
```

### CameraController Class

Main controller class providing:

1. **Target Management**
   - `set_target(entity)`: Set entity to follow
   - `update(delta_time)`: Update camera position/rotation

2. **Mode Switching**
   - `set_mode(FollowMode)`: Switch between follow algorithms

3. **Position Calculation**
   - `_calculate_smooth_position()`: Basic offset positioning
   - `_calculate_predictive_position()`: Velocity-based prediction
   - `_calculate_orbital_position()`: Orbital motion

4. **Follow Algorithms**
   - `_apply_smooth_follow()`: Smooth damping
   - `_apply_spring_follow()`: Spring physics

5. **Safety Features**
   - `_apply_collision_avoidance()`: Push camera away if too close
   - `_clamp_distance()`: Keep within min/max distance

### MovingTarget Class

Demonstrates various movement patterns:

| Pattern | Description |
|---------|-------------|
| `circle` | Circular motion around center |
| `figure8` | Figure-8 (lemniscate) pattern |
| `linear` | Back-and-forth linear motion |
| `random` | Random walk with smooth transitions |
| `spiral` | Spiral with changing radius and height |

## Mathematical Formulas

### Critically Damped Spring

The smooth damp algorithm uses a critically damped spring system:

```
ω = 2 / smooth_time
spring_accel = ω² * displacement
damping_accel = -2ω * velocity
acceleration = spring_accel + damping_accel

velocity += acceleration * Δt
position += velocity * Δt
```

Critical damping provides the fastest return to equilibrium without oscillation.

### Spring System

Hooke's law with damping:

```
F_spring = -k * (x - target)
F_damping = -c * v
F_total = F_spring + F_damping

a = F_total / m
v = v + a * Δt
x = x + v * Δt
```

### Predictive Follow

Linear velocity prediction:

```
velocity = (pos_current - pos_last) / Δt
predicted_pos = target_pos + velocity * look_ahead_time
```

## Performance Considerations

1. **Update Frequency**: Camera update should run every frame
2. **Delta Time**: Always use proper delta time for frame-rate independence
3. **Smooth Time**: Typical values are 0.1 - 1.0 seconds
4. **Prediction**: Only enable for fast-moving targets
5. **Spring Parameters**: Adjust stiffness/damping ratio to prevent oscillation

## Extending the System

### Adding Custom Follow Modes

```python
class MyCustomCamera(CameraController):
    def __init__(self, app):
        super().__init__(app)
        self.mode = FollowMode(100)  # Custom mode ID
    
    def update(self, delta_time):
        if self.mode == FollowMode(100):
            # Custom follow logic
            desired_pos = self._my_custom_position()
            self.current_position = self._apply_smooth_follow(desired_pos, delta_time)
        else:
            super().update(delta_time)
```

### Integrating with Physics

For collision detection integration:

```python
def _apply_collision_avoidance(self):
    # Cast ray from target to desired camera position
    direction = self.current_position - self.target_position
    distance = direction.length()
    
    # Use physics engine raycast
    hit = physics.raycast(
        origin=self.target_position,
        direction=direction.normalized(),
        max_distance=distance
    )
    
    if hit:
        # Place camera before collision point
        self.current_position = hit.point + hit.normal * buffer_distance
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera too jerky | Increase `smooth_time` or decrease `smooth_speed` |
| Camera lags too far behind | Decrease smooth_time or increase smooth_speed |
| Camera oscillates | Check spring parameters (increase damping) |
| Camera goes through walls | Enable collision avoidance, check collision layers |
| Target not found | Ensure target entity has TransformComponent |

## API Reference

### camera_math module

| Function/Class | Description |
|----------------|-------------|
| `Vector3(x, y, z)` | 3D vector with full math operations |
| `lerp(a, b, t)` | Linear interpolation between scalars |
| `lerp_vector(a, b, t)` | Linear interpolation between vectors |
| `smooth_damp(c, t, v, st, dt)` | Critically damped spring smoothing |
| `FollowMode` | Enum of follow modes |

### camera_controller module

| Class | Description |
|-------|-------------|
| `CameraController(app)` | Main camera controller |
| `CameraManager(app)` | Manager for multiple cameras |
| `MovingTarget(entity, scene)` | Demo moving target class |

## References

- Based on Unity's SmoothDamp and camera follow implementations
- Spring physics: Critically damped harmonic oscillator
- Predictive follow: Linear extrapolation technique

## License

Part of the RoboCute project. See main project license for details.
