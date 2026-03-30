"""
Pure Pursuit path tracking algorithm implementation.

This module implements the Pure Pursuit algorithm for autonomous vehicle/robot
path tracking. The algorithm calculates steering angles based on a lookahead
point on the desired path.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Pose:
    """Represents a 2D pose with position (x, y) and heading angle (theta)."""
    x: float
    y: float
    theta: float  # heading angle in radians


@dataclass
class PathPoint:
    """Represents a point on the path with position (x, y)."""
    x: float
    y: float


@dataclass
class ControlCommand:
    """Control command output from the Pure Pursuit algorithm."""
    steering_angle: float  # steering angle in radians
    curvature: float       # curvature (1 / turning radius)
    target_point: Optional[Tuple[float, float]]  # selected lookahead point


class PurePursuitTracker:
    """
    Pure Pursuit path tracking controller.
    
    The Pure Pursuit algorithm is a geometric path tracking method that
    calculates steering commands based on a lookahead point on the path.
    It assumes the vehicle follows an arc that passes through the current
    position and the lookahead point.
    
    Attributes:
        lookahead_distance: Distance to lookahead point (meters)
        wheelbase: Distance between front and rear axles (meters)
        max_steering_angle: Maximum allowable steering angle (radians)
    """
    
    def __init__(
        self,
        lookahead_distance: float,
        wheelbase: float,
        max_steering_angle: float = math.pi / 4
    ) -> None:
        """
        Initialize the Pure Pursuit tracker.
        
        Args:
            lookahead_distance: Distance to look ahead on the path (meters)
            wheelbase: Vehicle wheelbase - distance between axles (meters)
            max_steering_angle: Maximum steering angle limit (radians)
        """
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle
    
    def compute_control(
        self,
        current_pose: Pose,
        path: List[PathPoint]
    ) -> ControlCommand:
        """
        Compute steering control to track the given path.
        
        Args:
            current_pose: Current vehicle pose (position and heading)
            path: List of points defining the desired path
            
        Returns:
            ControlCommand containing steering angle and curvature
        """
        # Find the lookahead point on the path
        target_point = self._find_lookahead_point(current_pose, path)
        
        if target_point is None:
            # No valid lookahead point found, return zero control
            return ControlCommand(
                steering_angle=0.0,
                curvature=0.0,
                target_point=None
            )
        
        # Transform target to vehicle coordinate frame
        dx = target_point[0] - current_pose.x
        dy = target_point[1] - current_pose.y
        
        # Rotate to vehicle frame
        cos_theta = math.cos(current_pose.theta)
        sin_theta = math.sin(current_pose.theta)
        
        local_x = cos_theta * dx + sin_theta * dy
        local_y = -sin_theta * dx + cos_theta * dy
        
        # Calculate curvature using pure pursuit formula
        # kappa = 2 * y / L^2
        # where y is the lateral offset and L is the lookahead distance
        if abs(local_x) < 1e-6:
            # Target is directly ahead or behind
            curvature = 0.0
        else:
            # Calculate curvature from geometry
            lookahead_sq = self.lookahead_distance ** 2
            curvature = 2.0 * local_y / lookahead_sq
        
        # Calculate steering angle from curvature and wheelbase
        # gamma = arctan(kappa * L)
        steering_angle = math.atan(curvature * self.wheelbase)
        
        # Clamp steering angle to limits
        steering_angle = max(
            -self.max_steering_angle,
            min(self.max_steering_angle, steering_angle)
        )
        
        return ControlCommand(
            steering_angle=steering_angle,
            curvature=curvature,
            target_point=target_point
        )
    
    def _find_lookahead_point(
        self,
        current_pose: Pose,
        path: List[PathPoint]
    ) -> Optional[Tuple[float, float]]:
        """
        Find the lookahead point on the path.
        
        Searches for a point on the path that is approximately
        lookahead_distance away from the current position.
        
        Args:
            current_pose: Current vehicle pose
            path: List of path points
            
        Returns:
            Coordinates of the lookahead point, or None if not found
        """
        if not path:
            return None
        
        # Find the closest point on the path
        closest_idx = 0
        closest_dist = float('inf')
        
        for i, point in enumerate(path):
            dist = math.hypot(point.x - current_pose.x, point.y - current_pose.y)
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        
        # Search forward from closest point for lookahead point
        for i in range(closest_idx, len(path)):
            dist = math.hypot(
                path[i].x - current_pose.x,
                path[i].y - current_pose.y
            )
            
            # Check if this point is at approximately lookahead distance
            if dist >= self.lookahead_distance:
                # Linear interpolation for more accurate lookahead
                if i > 0:
                    prev_dist = math.hypot(
                        path[i-1].x - current_pose.x,
                        path[i-1].y - current_pose.y
                    )
                    if abs(dist - prev_dist) > 1e-6:
                        # Interpolate between points
                        t = (self.lookahead_distance - prev_dist) / (dist - prev_dist)
                        t = max(0.0, min(1.0, t))
                        
                        x = path[i-1].x + t * (path[i].x - path[i-1].x)
                        y = path[i-1].y + t * (path[i].y - path[i-1].y)
                        return (x, y)
                
                return (path[i].x, path[i].y)
        
        # If no point at lookahead distance, return the last point
        if path:
            return (path[-1].x, path[-1].y)
        
        return None
    
    def update_lookahead_distance(
        self,
        current_speed: float,
        min_distance: float = 1.0,
        max_distance: float = 10.0,
        speed_factor: float = 0.5
    ) -> None:
        """
        Dynamically update lookahead distance based on current speed.
        
        Higher speeds require longer lookahead distances for stability.
        
        Args:
            current_speed: Current vehicle speed (m/s)
            min_distance: Minimum lookahead distance
            max_distance: Maximum lookahead distance
            speed_factor: Multiplier for speed-based adjustment
        """
        self.lookahead_distance = min_distance + speed_factor * current_speed
        self.lookahead_distance = max(min_distance, min(max_distance, self.lookahead_distance))


def generate_straight_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_points: int = 50
) -> List[PathPoint]:
    """
    Generate a straight line path.
    
    Args:
        start: Starting point (x, y)
        end: Ending point (x, y)
        num_points: Number of points to generate
        
    Returns:
        List of path points
    """
    path = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        path.append(PathPoint(x, y))
    return path


def generate_circular_path(
    center: Tuple[float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    num_points: int = 50
) -> List[PathPoint]:
    """
    Generate a circular arc path.
    
    Args:
        center: Circle center (x, y)
        radius: Circle radius
        start_angle: Starting angle in radians
        end_angle: Ending angle in radians
        num_points: Number of points to generate
        
    Returns:
        List of path points
    """
    path = []
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = start_angle + t * (end_angle - start_angle)
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        path.append(PathPoint(x, y))
    return path


def generate_sine_path(
    start: Tuple[float, float],
    length: float,
    amplitude: float,
    frequency: float,
    num_points: int = 100
) -> List[PathPoint]:
    """
    Generate a sinusoidal path.
    
    Args:
        start: Starting point (x, y)
        length: Path length in x direction
        amplitude: Sine wave amplitude
        frequency: Number of complete cycles
        num_points: Number of points to generate
        
    Returns:
        List of path points
    """
    path = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = start[0] + t * length
        y = start[1] + amplitude * math.sin(2 * math.pi * frequency * t)
        path.append(PathPoint(x, y))
    return path


def simulate_vehicle(
    tracker: PurePursuitTracker,
    initial_pose: Pose,
    path: List[PathPoint],
    speed: float,
    dt: float,
    max_steps: int = 1000,
    goal_tolerance: float = 0.5
) -> List[Pose]:
    """
    Simulate vehicle following a path using Pure Pursuit.
    
    Args:
        tracker: Pure Pursuit tracker instance
        initial_pose: Starting vehicle pose
        path: Target path to follow
        speed: Constant forward speed (m/s)
        dt: Time step for simulation (seconds)
        max_steps: Maximum simulation steps
        goal_tolerance: Distance to goal to consider reached
        
    Returns:
        List of poses representing vehicle trajectory
    """
    trajectory = [initial_pose]
    current_pose = initial_pose
    
    for _ in range(max_steps):
        # Compute control command
        command = tracker.compute_control(current_pose, path)
        
        # Check if reached end of path
        if command.target_point is None:
            break
        
        dist_to_goal = math.hypot(
            path[-1].x - current_pose.x,
            path[-1].y - current_pose.y
        )
        if dist_to_goal < goal_tolerance:
            break
        
        # Update vehicle state using bicycle model
        # dx = v * cos(theta)
        # dy = v * sin(theta)
        # dtheta = v * tan(gamma) / L
        
        new_x = current_pose.x + speed * math.cos(current_pose.theta) * dt
        new_y = current_pose.y + speed * math.sin(current_pose.theta) * dt
        new_theta = current_pose.theta + (
            speed * math.tan(command.steering_angle) / tracker.wheelbase
        ) * dt
        
        # Normalize angle to [-pi, pi]
        new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))
        
        current_pose = Pose(new_x, new_y, new_theta)
        trajectory.append(current_pose)
    
    return trajectory


def main() -> None:
    """Demonstration of the Pure Pursuit algorithm."""
    print("=" * 60)
    print("Pure Pursuit Path Tracking Demo")
    print("=" * 60)
    
    # Create a tracker with parameters
    lookahead_distance = 2.0  # meters
    wheelbase = 1.0           # meters
    
    tracker = PurePursuitTracker(
        lookahead_distance=lookahead_distance,
        wheelbase=wheelbase,
        max_steering_angle=math.pi / 3
    )
    
    # Generate a test path (sine wave)
    path = generate_sine_path(
        start=(0.0, 0.0),
        length=20.0,
        amplitude=2.0,
        frequency=1.0,
        num_points=100
    )
    
    print(f"\nPath generated: {len(path)} points")
    print(f"Start: ({path[0].x:.2f}, {path[0].y:.2f})")
    print(f"End: ({path[-1].x:.2f}, {path[-1].y:.2f})")
    
    # Initial vehicle pose (slightly offset from path start)
    initial_pose = Pose(x=0.0, y=1.0, theta=0.0)
    
    print(f"\nInitial pose: ({initial_pose.x:.2f}, {initial_pose.y:.2f}, "
          f"{math.degrees(initial_pose.theta):.1f}°)")
    
    # Simulation parameters
    speed = 2.0      # m/s
    dt = 0.05        # 50ms time step
    
    print(f"\nSimulation parameters:")
    print(f"  Speed: {speed} m/s")
    print(f"  Time step: {dt} s")
    print(f"  Lookahead distance: {lookahead_distance} m")
    print(f"  Wheelbase: {wheelbase} m")
    
    # Run simulation
    print("\nRunning simulation...")
    trajectory = simulate_vehicle(
        tracker=tracker,
        initial_pose=initial_pose,
        path=path,
        speed=speed,
        dt=dt,
        max_steps=1000,
        goal_tolerance=0.5
    )
    
    print(f"Simulation completed: {len(trajectory)} steps")
    
    # Calculate tracking error
    final_pose = trajectory[-1]
    final_error = math.hypot(
        path[-1].x - final_pose.x,
        path[-1].y - final_pose.y
    )
    print(f"Final tracking error: {final_error:.3f} m")
    
    # Compute and display a few control commands
    print("\nSample control commands:")
    for i in range(0, min(len(trajectory), 10)):
        pose = trajectory[i * len(trajectory) // 10 if i < 9 else -1]
        cmd = tracker.compute_control(pose, path)
        if cmd.target_point:
            print(f"  Step {i}: steering={math.degrees(cmd.steering_angle):6.2f}°, "
                  f"curvature={cmd.curvature:6.3f}")


if __name__ == "__main__":
    main()
