"""
Visualization of Pure Pursuit path tracking.

This script visualizes the Pure Pursuit algorithm with matplotlib.
Run this after installing matplotlib: uv add matplotlib
"""

import math
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Install with: uv add matplotlib")

from pure_pursuit import (
    PurePursuitTracker,
    Pose,
    PathPoint,
    generate_sine_path,
    generate_circular_path,
    generate_straight_path,
    simulate_vehicle,
)


def plot_path(
    ax,
    path: List[PathPoint],
    color: str = 'blue',
    label: str = 'Path',
    linewidth: float = 2.0
) -> None:
    """Plot the reference path."""
    x_coords = [p.x for p in path]
    y_coords = [p.y for p in path]
    ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, label=label)


def plot_trajectory(
    ax,
    trajectory: List[Pose],
    color: str = 'red',
    label: str = 'Trajectory',
    linewidth: float = 1.5
) -> None:
    """Plot the vehicle trajectory."""
    x_coords = [p.x for p in trajectory]
    y_coords = [p.y for p in trajectory]
    ax.plot(x_coords, y_coords, '--', color=color, linewidth=linewidth, label=label)


def plot_vehicle_pose(
    ax,
    pose: Pose,
    wheelbase: float,
    color: str = 'green',
    size: float = 0.3
) -> None:
    """Plot a vehicle pose as an arrow."""
    arrow = FancyArrowPatch(
        (pose.x, pose.y),
        (
            pose.x + size * 2 * math.cos(pose.theta),
            pose.y + size * 2 * math.sin(pose.theta)
        ),
        arrowstyle='->',
        mutation_scale=20,
        color=color,
        linewidth=2
    )
    ax.add_patch(arrow)
    
    # Draw vehicle body
    body_x = pose.x + size * math.cos(pose.theta)
    body_y = pose.y + size * math.sin(pose.theta)
    ax.plot(body_x, body_y, 'o', color=color, markersize=8)


def plot_lookahead_point(
    ax,
    point: Tuple[float, float],
    color: str = 'orange',
    size: float = 50
) -> None:
    """Plot the lookahead point."""
    ax.scatter(point[0], point[1], color=color, s=size, marker='*', 
               edgecolors='black', linewidths=1, label='Lookahead', zorder=5)


def create_comparison_plot() -> None:
    """Create a comparison plot of different path tracking scenarios."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is required for visualization.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Pure Pursuit Path Tracking - Demo Scenarios', fontsize=16)
    
    scenarios = [
        {
            'name': 'Sine Wave Path',
            'path_gen': lambda: generate_sine_path(
                start=(0.0, 0.0), length=20.0, amplitude=2.0, 
                frequency=1.0, num_points=100
            ),
            'initial_pose': Pose(x=0.0, y=1.0, theta=0.0),
            'lookahead': 2.0,
        },
        {
            'name': 'Circular Path',
            'path_gen': lambda: generate_circular_path(
                center=(5.0, 0.0), radius=5.0,
                start_angle=math.pi, end_angle=2 * math.pi, num_points=80
            ),
            'initial_pose': Pose(x=0.0, y=1.0, theta=0.0),
            'lookahead': 1.5,
        },
        {
            'name': 'Straight Path',
            'path_gen': lambda: generate_straight_path(
                start=(0.0, 0.0), end=(15.0, 5.0), num_points=50
            ),
            'initial_pose': Pose(x=0.0, y=2.0, theta=math.pi/8),
            'lookahead': 2.0,
        },
        {
            'name': 'High Frequency Sine',
            'path_gen': lambda: generate_sine_path(
                start=(0.0, 0.0), length=15.0, amplitude=1.5,
                frequency=2.0, num_points=150
            ),
            'initial_pose': Pose(x=0.0, y=0.5, theta=0.0),
            'lookahead': 1.0,
        },
    ]
    
    speed = 2.0
    dt = 0.05
    wheelbase = 1.0
    
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]
        
        # Generate path
        path = scenario['path_gen']()
        
        # Create tracker
        tracker = PurePursuitTracker(
            lookahead_distance=scenario['lookahead'],
            wheelbase=wheelbase,
            max_steering_angle=math.pi / 3
        )
        
        # Simulate
        trajectory = simulate_vehicle(
            tracker=tracker,
            initial_pose=scenario['initial_pose'],
            path=path,
            speed=speed,
            dt=dt,
            max_steps=1000,
            goal_tolerance=0.5
        )
        
        # Plot
        plot_path(ax, path, color='blue', label='Reference Path')
        plot_trajectory(ax, trajectory, color='red', label='Vehicle Trajectory')
        plot_vehicle_pose(ax, trajectory[0], wheelbase, color='green', size=0.2)
        plot_vehicle_pose(ax, trajectory[-1], wheelbase, color='purple', size=0.2)
        
        # Calculate error
        final_error = math.hypot(
            path[-1].x - trajectory[-1].x,
            path[-1].y - trajectory[-1].y
        )
        
        ax.set_title(f"{scenario['name']}\n"
                    f"Steps: {len(trajectory)}, Final Error: {final_error:.3f}m")
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('pure_pursuit_demo.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: samples/robot/pure_pursuit_demo.png")
    plt.show()


def create_single_detailed_plot() -> None:
    """Create a detailed plot showing lookahead points."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is required for visualization.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate path
    path = generate_sine_path(
        start=(0.0, 0.0), length=20.0, amplitude=2.0,
        frequency=1.0, num_points=100
    )
    
    # Setup tracker
    lookahead = 2.0
    wheelbase = 1.0
    tracker = PurePursuitTracker(
        lookahead_distance=lookahead,
        wheelbase=wheelbase,
        max_steering_angle=math.pi / 3
    )
    
    initial_pose = Pose(x=0.0, y=1.0, theta=0.0)
    speed = 2.0
    dt = 0.05
    
    # Simulate
    trajectory = simulate_vehicle(
        tracker=tracker,
        initial_pose=initial_pose,
        path=path,
        speed=speed,
        dt=dt,
        max_steps=1000,
        goal_tolerance=0.5
    )
    
    # Plot reference path
    plot_path(ax, path, color='blue', linewidth=2.5, label='Reference Path')
    
    # Plot trajectory
    plot_trajectory(ax, trajectory, color='red', linewidth=1.5, 
                    label='Vehicle Trajectory')
    
    # Plot some lookahead points
    sample_indices = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4]
    for i in sample_indices:
        if i < len(trajectory):
            pose = trajectory[i]
            cmd = tracker.compute_control(pose, path)
            if cmd.target_point:
                # Draw line to lookahead
                ax.plot([pose.x, cmd.target_point[0]], 
                       [pose.y, cmd.target_point[1]],
                       'g--', alpha=0.5, linewidth=1)
                plot_lookahead_point(ax, cmd.target_point, size=100)
    
    # Plot start and end poses
    plot_vehicle_pose(ax, trajectory[0], wheelbase, color='green', size=0.3)
    plot_vehicle_pose(ax, trajectory[-1], wheelbase, color='purple', size=0.3)
    
    # Add legend entries for start/end
    ax.plot([], [], 'go', markersize=10, label='Start Position')
    ax.plot([], [], 'mo', markersize=10, label='End Position')
    
    ax.set_title('Pure Pursuit Tracking with Lookahead Points', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('pure_pursuit_detail.png', dpi=150, bbox_inches='tight')
    print("Detailed plot saved to: samples/robot/pure_pursuit_detail.png")
    plt.show()


def main() -> None:
    """Run visualization demos."""
    print("=" * 60)
    print("Pure Pursuit Visualization")
    print("=" * 60)
    
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not found. Please install it first:")
        print("  uv add matplotlib")
        return
    
    print("\nGenerating comparison plot...")
    create_comparison_plot()
    
    print("\nGenerating detailed plot...")
    create_single_detailed_plot()


if __name__ == "__main__":
    main()
