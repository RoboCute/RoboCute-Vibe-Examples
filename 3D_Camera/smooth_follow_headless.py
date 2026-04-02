"""
Smooth Follow Camera Demo - Headless/Simulation Mode

Pure simulation without rendering. Tests camera following logic
without requiring display initialization.
"""

import time
from pathlib import Path
from typing import Optional

# Only import math components, not robocute
from camera_controller import CameraController, CameraManager, FollowMode
from camera_math import Vector3, smooth_damp
from moving_target import MovingTarget, TargetConfig, MovementPattern


class SimulatedScene:
    """Minimal scene simulation for headless testing"""
    def __init__(self):
        self.entities = {}
        self._entity_counter = 0
    
    def add_entity(self, name: str = "entity"):
        """Simulate adding an entity"""
        self._entity_counter += 1
        entity_id = f"{name}_{self._entity_counter}"
        self.entities[entity_id] = {
            "id": entity_id,
            "position": Vector3(0, 0, 0),
            "velocity": Vector3(0, 0, 0),
        }
        return SimulatedEntity(self, entity_id)
    
    def get_entity_position(self, entity_id: str) -> Vector3:
        """Get entity position"""
        if entity_id in self.entities:
            return self.entities[entity_id]["position"].copy()
        return Vector3(0, 0, 0)
    
    def set_entity_position(self, entity_id: str, position: Vector3):
        """Set entity position"""
        if entity_id in self.entities:
            self.entities[entity_id]["position"] = position.copy()


class SimulatedEntity:
    """Minimal entity for headless testing"""
    def __init__(self, scene: SimulatedScene, entity_id: str):
        self.scene = scene
        self.entity_id = entity_id
        self._position = Vector3(0, 0, 0)
    
    def get_component(self, name: str):
        """Simulate getting a component"""
        return self
    
    def position(self):
        """Get position"""
        return self._position
    
    def set_pos(self, pos, recursive=False):
        """Set position"""
        self._position = pos.copy() if hasattr(pos, 'copy') else Vector3(pos.x, pos.y, pos.z)
        self.scene.set_entity_position(self.entity_id, self._position)


class SimulatedApp:
    """Minimal app simulation for headless testing"""
    def __init__(self):
        self._scene = SimulatedScene()
        self._display_transform = None
    
    def init(self, project_path=None, backend_name="dx"):
        """Simulate app initialization"""
        print("[SimulatedApp] Initialized")
    
    def get_display_transform(self):
        """Return None in headless mode"""
        return None
    
    def ctx(self):
        """Simulate context"""
        return self
    
    def scene(self):
        """Get scene"""
        return self._scene
    
    def run(self, limit_frame=None):
        """Simulated run loop"""
        # This is handled by our demo class
        pass


class HeadlessCameraController:
    """
    Camera controller that works without display transform
    Pure simulation of camera following logic
    """
    
    def __init__(self, name: str = "main_camera"):
        self.name = name
        self.mode = FollowMode.SMOOTH
        
        # Target
        self.target_entity = None
        self.target_position = Vector3(0, 0, 0)
        self.target_velocity = Vector3(0, 0, 0)
        self.target_last_position = Vector3(0, 0, 0)
        
        # Settings
        self.offset = Vector3(0, 6, -12)
        self.position_smooth_speed = 4.0
        self.rotation_smooth_speed = 5.0
        self.min_distance = 3.0
        self.max_distance = 30.0
        self.enable_collision_avoidance = True
        self.collision_buffer = 0.5
        
        # State
        self.current_position = Vector3(0, 6, -12)
        self.current_velocity = Vector3(0, 0, 0)
        self.current_rotation = Vector3(0, 0, 0)
        
        # Stats
        self.frame_count = 0
        self.total_distance = 0.0
    
    def initialize(self):
        """Initialize (no display transform needed)"""
        pass
    
    def set_target(self, target):
        """Set target entity"""
        self.target_entity = target
        if target:
            # Get initial target position
            pos = target.position()
            self.target_position = Vector3(pos.x, pos.y, pos.z)
            self.target_last_position = self.target_position.copy()
            self.current_position = self.target_position + self.offset
    
    def set_mode(self, mode: FollowMode):
        """Set follow mode"""
        self.mode = mode
    
    def update(self, delta_time: float):
        """Update camera position"""
        if not self.target_entity:
            return
        
        self.frame_count += 1
        
        # Update target position and velocity
        current_target_pos = self.target_entity.position()
        new_target_pos = Vector3(current_target_pos.x, current_target_pos.y, current_target_pos.z)
        
        if delta_time > 0:
            self.target_velocity = (new_target_pos - self.target_last_position) / delta_time
        
        self.target_last_position = self.target_position.copy()
        self.target_position = new_target_pos
        
        # Calculate desired position
        desired_pos = self.target_position + self.offset
        
        # Apply smooth damping
        smooth_time = 1.0 / self.position_smooth_speed
        new_pos, new_vel = smooth_damp(
            self.current_position,
            desired_pos,
            self.current_velocity,
            smooth_time,
            delta_time
        )
        
        self.current_velocity = new_vel
        self.current_position = new_pos
        
        # Apply collision avoidance
        if self.enable_collision_avoidance:
            self._apply_collision_avoidance()
        
        # Clamp distance
        self._clamp_distance()
        
        # Track total distance
        frame_dist = self.current_velocity.length() * delta_time
        self.total_distance += frame_dist
    
    def _apply_collision_avoidance(self):
        """Keep camera at minimum distance"""
        direction = self.current_position - self.target_position
        distance = direction.length()
        
        if distance < self.min_distance + self.collision_buffer:
            push_dir = direction.normalized()
            if push_dir.length() < 0.001:
                push_dir = Vector3(0, 1, 0)
            target_dist = self.min_distance + self.collision_buffer
            self.current_position = self.target_position + push_dir * target_dist
    
    def _clamp_distance(self):
        """Clamp camera distance"""
        direction = self.current_position - self.target_position
        distance = direction.length()
        
        if distance < self.min_distance:
            dir_norm = direction.normalized()
            if dir_norm.length() < 0.001:
                dir_norm = Vector3(0, 0, 1)
            self.current_position = self.target_position + dir_norm * self.min_distance
        elif distance > self.max_distance:
            dir_norm = direction.normalized()
            self.current_position = self.target_position + dir_norm * self.max_distance
    
    def get_distance_to_target(self) -> float:
        """Get distance to target"""
        return (self.current_position - self.target_position).length()


class SimulatedMovingTarget:
    """Moving target without rendering"""
    
    def __init__(self, scene: SimulatedScene, name: str = "target", pattern: str = "circle"):
        self.scene = scene
        self.name = name
        self.pattern = pattern
        self.time = 0.0
        self.speed = 2.0
        self.radius = 10.0
        self.base_height = 1.0
        
        self.entity = scene.add_entity(name)
        self.entity.set_pos(Vector3(0, self.base_height, 0))
        
        self.velocity = Vector3(0, 0, 0)
        self.last_position = Vector3(0, self.base_height, 0)
    
    def update(self, delta_time: float):
        """Update target position"""
        import math
        
        self.time += delta_time * self.speed
        self.last_position = self.position()
        
        if self.pattern == "circle":
            x = math.cos(self.time) * self.radius
            z = math.sin(self.time) * self.radius
            new_pos = Vector3(x, self.base_height, z)
        elif self.pattern == "figure8":
            x = math.sin(self.time * 2) * self.radius
            z = math.sin(self.time) * self.radius
            new_pos = Vector3(x, self.base_height, z)
        elif self.pattern == "linear":
            x = math.sin(self.time) * self.radius
            new_pos = Vector3(x, self.base_height, 0)
        elif self.pattern == "spiral":
            angle = self.time
            r = (math.sin(self.time * 0.5) + 1) * 0.5 * self.radius
            x = math.cos(angle) * r
            z = math.sin(angle) * r
            y = self.base_height + math.sin(self.time * 2) * 2
            new_pos = Vector3(x, y, z)
        else:
            new_pos = Vector3(0, self.base_height, 0)
        
        self.entity.set_pos(new_pos)
        
        if delta_time > 0:
            self.velocity = (new_pos - self.last_position) / delta_time
    
    def position(self) -> Vector3:
        """Get current position"""
        return self.entity.position()


class SmoothFollowHeadlessDemo:
    """Headless demo using pure simulation"""
    
    def __init__(self):
        self.scene = SimulatedScene()
        self.camera = None
        self.target = None
        self.frame_count = 0
    
    def setup(self, pattern: str = "circle"):
        """Setup camera and target"""
        # Create target
        self.target = SimulatedMovingTarget(self.scene, "target", pattern)
        
        # Create camera
        self.camera = HeadlessCameraController("main")
        self.camera.set_target(self.target.entity)
        self.camera.offset = Vector3(0, 6, -12)
        self.camera.position_smooth_speed = 4.0
        
        print("=" * 70)
        print("Smooth Follow Camera Demo - Headless Mode")
        print("=" * 70)
        print(f"Pattern: {pattern}")
        print(f"Camera Offset: {self.camera.offset}")
        print(f"Smooth Speed: {self.camera.position_smooth_speed}")
        print("=" * 70)
        print(f"{'Frame':>6} | {'Camera X':>9} | {'Camera Y':>9} | {'Camera Z':>9} | {'Distance':>10}")
        print("-" * 70)
    
    def run(self, limit_frames: int = 256):
        """Run simulation for specified frames"""
        delta_time = 1.0 / 60.0  # 60 FPS simulation
        
        start_time = time.time()
        
        for frame in range(1, limit_frames + 1):
            self.frame_count = frame
            
            # Update target
            self.target.update(delta_time)
            
            # Update camera
            self.camera.update(delta_time)
            
            # Print frame info
            cam_pos = self.camera.current_position
            distance = self.camera.get_distance_to_target()
            print(f"{frame:>6} | {cam_pos.x:>9.4f} | {cam_pos.y:>9.4f} | {cam_pos.z:>9.4f} | {distance:>10.4f}")
        
        elapsed = time.time() - start_time
        
        print("-" * 70)
        print(f"Demo complete: {self.frame_count} frames in {elapsed:.3f}s")
        print(f"Effective FPS: {self.frame_count / elapsed:.1f}")
        print(f"Total camera distance traveled: {self.camera.total_distance:.4f}")
        print(f"Final distance to target: {self.camera.get_distance_to_target():.4f}")


def main():
    """Main entry point"""
    import sys
    
    args = sys.argv[1:]
    
    # Parse arguments
    pattern = "circle"
    limit_frames = 256
    
    for i, arg in enumerate(args):
        if arg == "--pattern" and i + 1 < len(args):
            pattern = args[i + 1]
        elif arg == "--frames" and i + 1 < len(args):
            limit_frames = int(args[i + 1])
        elif arg in ("--help", "-h"):
            print("""
Smooth Follow Camera Demo - Headless Mode

Usage:
    python smooth_follow_headless.py [options]

Options:
    --pattern PATTERN    Movement pattern: circle, figure8, linear, spiral (default: circle)
    --frames N           Run for exactly N frames (default: 256)
    --help, -h           Show this help

Examples:
    python smooth_follow_headless.py
    python smooth_follow_headless.py --pattern figure8 --frames 512
            """)
            return
    
    # Run demo
    demo = SmoothFollowHeadlessDemo()
    demo.setup(pattern=pattern)
    demo.run(limit_frames=limit_frames)


if __name__ == "__main__":
    main()
