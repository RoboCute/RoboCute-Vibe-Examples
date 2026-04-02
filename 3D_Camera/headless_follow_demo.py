"""
Headless 3D Camera Follow Demo using RoboCute API

Runs the camera follow simulation without creating a display window.
Uses actual RoboCute app initialization but skips init_display().
Useful for testing camera logic with real engine integration.
"""

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import robocute as rbc
import robocute.rbc_ext as re
import robocute.rbc_ext.luisa as lc

from camera_controller import CameraController, CameraManager, FollowMode, Vector3


class MovingTargetEntity:
    """Creates and controls a moving target entity in the scene"""
    
    PATTERNS = ["circle", "figure8", "linear", "spiral"]
    
    def __init__(self, scene: re.world.Scene, name: str = "target", pattern: str = "circle"):
        self.scene = scene
        self.name = name
        self.pattern = pattern
        self.time = 0.0
        self.speed = 2.0
        self.radius = 10.0
        self.base_height = 1.0
        
        # Create entity
        self.entity = scene.add_entity()
        self.entity.set_name(name)
        
        # Get transform
        self.transform = re.world.TransformComponent(self.entity.add_component("TransformComponent"))
        self.transform.set_pos(lc.double3(0, self.base_height, 0), recursive=False)
        
        # Create visual mesh
        self._create_mesh()
        
        # Track velocity for predictive camera
        self.last_position = Vector3(0, self.base_height, 0)
        self.velocity = Vector3(0, 0, 0)
    
    def _create_mesh(self):
        """Create a simple cube mesh as visual"""
        render = re.world.RenderComponent(self.entity.add_component("RenderComponent"))
        
        mesh = re.world.MeshResource()
        submesh_offsets = np.array([0], dtype=np.uint32)
        mesh.create_empty(submesh_offsets, 8, 12, uv_count=0,
                         contained_normal=False, contained_tangent=False)
        
        pos_buffer = mesh.pos_buffer()
        r = 0.5
        pos_buffer[0] = lc.double3(-r, 0, -r)
        pos_buffer[1] = lc.double3(r, 0, -r)
        pos_buffer[2] = lc.double3(r, 2*r, -r)
        pos_buffer[3] = lc.double3(-r, 2*r, -r)
        pos_buffer[4] = lc.double3(-r, 0, r)
        pos_buffer[5] = lc.double3(r, 0, r)
        pos_buffer[6] = lc.double3(r, 2*r, r)
        pos_buffer[7] = lc.double3(-r, 2*r, r)
        
        mat = re.world.MaterialResource()
        mat_json = '''{
            "type": "pbr",
            "base_color": [0.9, 0.3, 0.2],
            "roughness": 0.4,
            "metallic": 0.2
        }'''
        mat.load_from_json(mat_json)
        
        mat_vector = lc.capsule_vector()
        mat_vector.emplace_back(mat._handle)
        
        render.update_object(mat_vector, mesh)
        mesh.install()
    
    def update(self, delta_time: float):
        """Update target position based on pattern"""
        self.time += delta_time * self.speed
        
        # Calculate new position
        if self.pattern == "circle":
            pos = self._circle_motion()
        elif self.pattern == "figure8":
            pos = self._figure8_motion()
        elif self.pattern == "linear":
            pos = self._linear_motion()
        elif self.pattern == "spiral":
            pos = self._spiral_motion()
        else:
            pos = self._circle_motion()
        
        # Calculate velocity
        self.velocity = (pos - self.last_position) / delta_time if delta_time > 0 else Vector3(0, 0, 0)
        self.last_position = pos
        
        # Update transform
        self.transform.set_pos(lc.double3(pos.x, pos.y, pos.z), recursive=False)
    
    def _circle_motion(self) -> Vector3:
        x = math.cos(self.time) * self.radius
        z = math.sin(self.time) * self.radius
        return Vector3(x, self.base_height, z)
    
    def _figure8_motion(self) -> Vector3:
        x = math.sin(self.time * 2) * self.radius
        z = math.sin(self.time) * self.radius
        return Vector3(x, self.base_height, z)
    
    def _linear_motion(self) -> Vector3:
        x = math.sin(self.time) * self.radius
        return Vector3(x, self.base_height, 0)
    
    def _spiral_motion(self) -> Vector3:
        angle = self.time
        r = (math.sin(self.time * 0.5) + 1) * 0.5 * self.radius
        x = math.cos(angle) * r
        z = math.sin(angle) * r
        y = self.base_height + math.sin(self.time * 2) * 2
        return Vector3(x, y, z)
    
    def get_position(self) -> Vector3:
        """Get current position as Vector3"""
        pos = self.transform.position()
        return Vector3(pos.x, pos.y, pos.z)


class HeadlessCameraDemo:
    """
    Headless camera demo using RoboCute API without display
    Initializes app but does NOT call init_display()
    """
    
    def __init__(self):
        self.app = rbc.app.App()
        self.ctx: Optional[re.world.RBCContext] = None
        self.scene: Optional[re.world.Scene] = None
        self.camera_manager: Optional[CameraManager] = None
        self.target: Optional[MovingTargetEntity] = None
        
        # Statistics
        self.frame_count = 0
        self.start_time = 0.0
    
    def initialize(self, project_path: Optional[Path] = None):
        """Initialize app in headless mode (no display)"""
        if project_path is None:
            project_path = Path(__file__).parent
        
        # Initialize app WITHOUT display
        self.app.init(project_path=project_path, backend_name="dx")
        # NOTE: We do NOT call app.init_display() - this is headless mode
        
        self.ctx = self.app.ctx
        self.scene = self.ctx.scene()
        
        # Setup camera manager
        self.camera_manager = CameraManager(self.app)
        camera = self.camera_manager.create_camera("main")
        
        print("=" * 60)
        print("Headless 3D Camera Follow Demo (RoboCute API)")
        print("=" * 60)
        print("Mode: HEADLESS (no display window)")
        print("=" * 60)
    
    def setup_scene(self, pattern: str = "circle"):
        """Create target and setup scene"""
        # Create moving target
        self.target = MovingTargetEntity(self.scene, "moving_target", pattern=pattern)
        
        # Set as camera target
        camera = self.camera_manager.get_camera("main")
        if camera:
            camera.set_target(self.target.entity)
            camera.offset = Vector3(0, 5, -10)
            camera.position_smooth_speed = 4.0
        
        print(f"Created target with pattern: {pattern}")
        print(f"Camera offset: {camera.offset}")
    
    def run_simulation(self, duration_seconds: float = 10.0, fps: int = 60) -> dict:
        """
        Run headless simulation loop
        
        Args:
            duration_seconds: How long to run
            fps: Target frames per second
            
        Returns:
            dict with simulation results
        """
        delta_time = 1.0 / fps
        total_frames = int(duration_seconds * fps)
        
        camera = self.camera_manager.get_camera("main")
        
        print(f"\nRunning simulation: {duration_seconds}s ({total_frames} frames @ {fps} FPS)")
        print("-" * 60)
        
        self.start_time = time.time()
        
        for frame in range(total_frames):
            self.frame_count += 1
            
            # Update target
            self.target.update(delta_time)
            
            # Update camera
            camera.update(delta_time)
            
            # Print progress every second
            if frame % fps == 0:
                elapsed = frame * delta_time
                target_pos = self.target.get_position()
                distance = (camera.current_position - target_pos).length()
                print(f"[t={elapsed:5.1f}s] Camera: {camera.current_position} | "
                      f"Target: {target_pos} | Dist: {distance:.2f}")
        
        real_time = time.time() - self.start_time
        
        # Results
        print("-" * 60)
        print("Simulation Complete")
        print("=" * 60)
        
        results = {
            "camera_info": camera.get_camera_info(),
            "target_final_pos": str(self.target.get_position()),
            "simulation_time": duration_seconds,
            "frames": self.frame_count,
            "real_time": real_time,
            "effective_fps": self.frame_count / real_time if real_time > 0 else 0,
        }
        
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        return results
    
    def shutdown(self):
        """Cleanup resources"""
        print("\nShutting down...")
        # Resources cleaned up automatically by RoboCute


def run_headless_demo(
    duration_seconds: float = 10.0,
    fps: int = 60,
    mode: FollowMode = FollowMode.SMOOTH,
    pattern: str = "circle"
) -> dict:
    """
    Run headless camera follow demo using RoboCute API
    
    Args:
        duration_seconds: How long to run the simulation
        fps: Target frames per second
        mode: Camera follow mode
        pattern: Target movement pattern
    
    Returns:
        dict with simulation results
    """
    demo = HeadlessCameraDemo()
    
    try:
        demo.initialize()
        demo.setup_scene(pattern=pattern)
        
        # Set camera mode
        camera = demo.camera_manager.get_camera("main")
        camera.set_mode(mode)
        
        print(f"\nMode: {mode.name}")
        print(f"Pattern: {pattern}")
        
        results = demo.run_simulation(duration_seconds, fps)
        return results
        
    finally:
        demo.shutdown()


def benchmark_all_modes(duration: float = 5.0, fps: int = 60):
    """Benchmark all follow modes"""
    print("\n" + "=" * 60)
    print("Benchmarking All Camera Follow Modes (Headless)")
    print("=" * 60)
    
    results = []
    
    for mode in FollowMode:
        print(f"\n{'=' * 60}")
        print(f"Testing {mode.name} mode")
        print("=" * 60)
        result = run_headless_demo(
            duration_seconds=duration,
            fps=fps,
            mode=mode,
            pattern="circle"
        )
        results.append((mode.name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    for mode_name, result in results:
        info = result["camera_info"]
        print(f"{mode_name:12} | Distance: {info['distance_to_target']:.2f} | "
              f"Frames: {info.get('frames', result['frames'])}")
    
    return results


def main():
    """Main entry point"""
    import sys
    
    args = sys.argv[1:]
    
    if "--help" in args or "-h" in args:
        print("""
Headless Camera Follow Demo (RoboCute API)

Runs camera simulation WITHOUT creating a display window.
Uses actual RoboCute app initialization but skips init_display().

Usage:
    python headless_follow_demo.py [options]

Options:
    --benchmark, -b       Benchmark all follow modes
    --duration N          Set simulation duration in seconds (default: 10)
    --mode MODE           Set follow mode: SMOOTH, SPRING, PREDICTIVE, ORBITAL
    --pattern PATTERN     Set pattern: circle, figure8, linear, spiral
    --fps N               Set target FPS (default: 60)
    --help, -h            Show this help

Examples:
    python headless_follow_demo.py
    python headless_follow_demo.py --mode SPRING --duration 20
    python headless_follow_demo.py --benchmark --duration 10
        """)
        return
    
    # Parse options
    mode = FollowMode.SMOOTH
    pattern = "circle"
    duration = 10.0
    fps = 60
    benchmark = False
    
    for i, arg in enumerate(args):
        if arg in ("--benchmark", "-b"):
            benchmark = True
        elif arg == "--mode" and i + 1 < len(args):
            mode = FollowMode[args[i + 1].upper()]
        elif arg == "--pattern" and i + 1 < len(args):
            pattern = args[i + 1]
        elif arg == "--duration" and i + 1 < len(args):
            duration = float(args[i + 1])
        elif arg == "--fps" and i + 1 < len(args):
            fps = int(args[i + 1])
    
    if benchmark:
        benchmark_all_modes(duration=duration, fps=fps)
    else:
        run_headless_demo(
            duration_seconds=duration,
            fps=fps,
            mode=mode,
            pattern=pattern
        )


if __name__ == "__main__":
    main()
