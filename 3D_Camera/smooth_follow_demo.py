"""
Smooth Follow Camera Demo

Demonstrates CameraController following a MovingTarget using smooth follow mode.
This is the standard camera follow behavior used in most 3D games.
"""

import math
from pathlib import Path
from typing import Optional

import robocute as rbc
import robocute.rbc_ext as re
import robocute.rbc_ext.luisa as lc

from camera_controller import CameraController, CameraManager, FollowMode
from camera_math import Vector3
from moving_target import MovingTarget, TargetConfig, MovementPattern, create_moving_target


class SmoothFollowDemo:
    """
    Demo showcasing smooth camera follow behavior.
    
    Features:
    - Camera smoothly follows a moving target
    - Configurable follow speed and offset
    - Target moves in various patterns (circle, figure8, etc.)
    - Adjustable smooth damping parameters
    """
    
    def __init__(self):
        self.app: Optional[rbc.app.App] = None
        self.ctx: Optional[re.world.RBCContext] = None
        self.scene: Optional[re.world.Scene] = None
        
        self.camera_manager: Optional[CameraManager] = None
        self.target: Optional[MovingTarget] = None
        
        # Demo settings
        self.show_debug_info = True
        self.frame_count = 0
    
    def initialize(self, project_path: Optional[Path] = None, headless: bool = False, backend: str = "dx"):
        """
        Initialize the demo
        
        Args:
            project_path: Path to project directory
            headless: If True, don't create display window
            backend: Backend name for rendering (e.g., "dx", "cuda")
        """
        if project_path is None:
            project_path = Path(__file__).parent
        
        # Initialize app
        print('1')
        
        self.app = rbc.app.App()
        self.app.init(project_path=project_path, backend_name=backend)
        self.app._tick_stage = re.world.TickStage.RasterPreview
        print('init')
        
        if not headless:
            self.app.init_display(1280, 720)
        
        self.ctx = self.app.ctx
        self.scene = self.app.scene
        
        print("=" * 60)
        print("Smooth Follow Camera Demo")
        print("=" * 60)
        print(f"Mode: {'Headless' if headless else 'Display'}")
    
    def setup_camera(self) -> CameraController:
        """Setup camera controller with smooth follow settings"""
        # Create camera manager and main camera
        self.camera_manager = CameraManager(self.app)
        camera = self.camera_manager.create_camera("main")
        
        # Configure smooth follow mode
        camera.set_mode(FollowMode.SMOOTH)
        
        # Camera offset: behind and above the target
        # This creates a typical 3rd-person view
        camera.offset = Vector3(0, 6, -12)
        
        # Smooth follow parameters
        # Higher speed = faster response, lower = smoother/more lag
        camera.position_smooth_speed = 4.0
        camera.rotation_smooth_speed = 5.0
        
        # Distance constraints
        camera.min_distance = 3.0
        camera.max_distance = 30.0
        
        # Enable collision avoidance to prevent camera from going through objects
        camera.enable_collision_avoidance = True
        camera.collision_buffer = 0.5
        
        print("-" * 60)
        print("Camera Configuration:")
        print(f"  Mode: {camera.mode.name}")
        print(f"  Offset: {camera.offset}")
        print(f"  Position Smooth Speed: {camera.position_smooth_speed}")
        print(f"  Rotation Smooth Speed: {camera.rotation_smooth_speed}")
        print(f"  Min Distance: {camera.min_distance}")
        print(f"  Max Distance: {camera.max_distance}")
        print("-" * 60)
        
        return camera
    
    def setup_target(self, pattern: str = "circle") -> MovingTarget:
        """Create and configure the moving target"""
        # Create target with specific pattern
        config = TargetConfig(
            pattern=MovementPattern.CIRCLE,
            speed=1.5,  # Moderate speed for smooth following
            radius=10.0,
            base_height=1.0
        )
        self.target = MovingTarget(self.scene, "player_target", config)
        
        print("Target Configuration:")
        print(f"  Pattern: {pattern}")
        print(f"  Speed: {self.target.config.speed}")
        print(f"  Radius: {self.target.config.radius}")
        print("=" * 60)
        
        return self.target
    
    def setup(self, pattern: str = "circle"):
        """Setup camera and target"""
        # Setup target first
        target = self.setup_target(pattern)
        
        # Setup camera and set target
        camera = self.setup_camera()
        camera.set_target(target.entity)
        
        # Enable camera control for manual override (if display mode)
        if hasattr(self.ctx, 'enable_camera_control'):
            self.ctx.enable_camera_control()
    
    def update(self, delta_time: float):
        """Update called every frame"""
        self.frame_count += 1
        
        # Update target position
        if self.target:
            self.target.update(delta_time)
        
        # Update camera to follow target
        if self.camera_manager:
            self.camera_manager.update(delta_time)
        
        # Debug info
        if self.show_debug_info and self.frame_count % 60 == 0:
            self._print_debug_info()
    
    def _print_debug_info(self):
        """Print camera debug info every second"""
        camera = self.camera_manager.get_camera("main")
        if camera and self.target:
            target_pos = self.target.get_position()
            distance = (camera.current_position - target_pos).length()
            
            print(f"[Frame {self.frame_count}] "
                  f"Camera: ({camera.current_position.x:.1f}, "
                  f"{camera.current_position.y:.1f}, "
                  f"{camera.current_position.z:.1f}) | "
                  f"Distance: {distance:.2f}")
    
    def run(self, limit_frames: int = 256):
        """
        Run the demo for exactly specified number of frames
        
        Args:
            limit_frames: Number of frames to run (default: 256)
        """
        print(f"\nRunning demo for exactly {limit_frames} frames...")
        print("-" * 60)
        print(f"{'Frame':>6} | {'Camera X':>8} | {'Camera Y':>8} | {'Camera Z':>8} | {'Distance':>10}")
        print("-" * 60)
        
        def tick_logic():
            self.update(self.app._delta_time)
            
            # Get camera and target info
            camera = self.camera_manager.get_camera("main")
            target_pos = self.target.get_position()
            cam_pos = camera.current_position
            distance = (cam_pos - target_pos).length()
            
            # Print frame counter and tracking info every frame
            print(f"{self.frame_count:>6} | {cam_pos.x:>8.3f} | {cam_pos.y:>8.3f} | {cam_pos.z:>8.3f} | {distance:>10.3f}")
        
        self.app.set_user_callback(tick_logic)
        
        try:
            # Run for exactly limit_frames frames
            self.app.run(limit_frame=limit_frames)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            print("-" * 60)
            print(f"Demo complete: {self.frame_count} frames rendered")
            self.shutdown()
    
    def shutdown(self):
        """Cleanup resources"""
        print("\nShutting down...")


def run_smooth_follow_demo(
    pattern: str = "circle",
    limit_frames: int = 256,
    headless: bool = False,
    project_path: Optional[Path] = None,
    backend: str = "dx"
):
    """
    Run smooth follow demo with configurable options
    
    Args:
        pattern: Target movement pattern (circle, figure8, linear, spiral)
        limit_frames: Number of frames to run (default: 256)
        headless: Run without display window
        project_path: Path to project directory
        backend: Backend name for rendering (e.g., "dx", "cuda")
    """
    demo = SmoothFollowDemo()
    
    try:
        demo.initialize(headless=headless, project_path=project_path, backend=backend)
        demo.setup(pattern=pattern)
        demo.run(limit_frames=limit_frames)
        
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        demo.shutdown()


def main():
    """Main entry point"""
    import sys
    
    args = sys.argv[1:]
    
    # Parse arguments
    pattern = "circle"
    limit_frames = None
    headless = False
    project_path = None
    backend = "dx"
    
    for i, arg in enumerate(args):
        if arg == "--pattern" and i + 1 < len(args):
            pattern = args[i + 1]
        elif arg == "--frames" and i + 1 < len(args):
            limit_frames = int(args[i + 1])
        elif arg == "--headless":
            headless = True
        elif arg == "--project" and i + 1 < len(args):
            project_path = Path(args[i + 1])
        elif arg == "--backend" and i + 1 < len(args):
            backend = args[i + 1]
        elif arg in ("--help", "-h"):
            print("""
Smooth Follow Camera Demo

Usage:
    python smooth_follow_demo.py [options]

Options:
    --pattern PATTERN    Movement pattern: circle, figure8, linear, spiral (default: circle)
    --frames N           Run for exactly N frames (default: 256)
    --headless           Run without display window
    --project PATH       Path to project directory
    --backend NAME       Backend name: dx, cuda (default: dx)
    --help, -h           Show this help

Examples:
    # Run with display (256 frames)
    python smooth_follow_demo.py
    python smooth_follow_demo.py --pattern figure8
    
    # Run headless with custom frame count
    python smooth_follow_demo.py --headless --frames 512
    
    # Run with custom project path and backend
    python smooth_follow_demo.py --project /path/to/project --backend cuda
            """)
            return
    run_smooth_follow_demo(
        pattern=pattern,
        limit_frames=limit_frames,
        headless=headless,
        project_path=project_path,
        backend=backend
    )


if __name__ == "__main__":
    main()
