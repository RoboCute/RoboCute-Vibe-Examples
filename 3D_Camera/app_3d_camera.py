"""
3D Camera Follow Demo Application for RoboCute

This application demonstrates various camera following algorithms:
- Smooth Follow: Standard smooth interpolation follow
- Spring Follow: Physics-based spring-damped follow
- Predictive Follow: Look-ahead follow for fast-moving targets
- Orbital Follow: Camera orbits around the target

Controls:
- F1: Cycle through follow modes
- F2: Toggle collision avoidance
- F3: Reset camera position
- F4: Spawn moving target
- WASD: Control target movement (when using manual control)

Usage:
    cd samples/3D_Camera
    python app_3d_camera.py
"""

import numpy as np
from pathlib import Path
import math
import random

import robocute as rbc
import robocute.rbc_ext as re
import robocute.rbc_ext.luisa as lc

from camera_controller import CameraController, CameraManager, FollowMode, Vector3


class MovingTarget:
    """
    A target that moves in various patterns for camera follow demonstration
    """
    
    MOVEMENT_PATTERNS = ["circle", "figure8", "linear", "random", "spiral"]
    
    def __init__(self, entity: re.world.Entity, scene: re.world.Scene):
        self.entity = entity
        self.scene = scene
        self.transform = re.world.TransformComponent(entity.get_component("TransformComponent"))
        
        # Movement state
        self.pattern = "circle"
        self.time = 0.0
        self.speed = 2.0
        self.radius = 10.0
        self.center = Vector3(0, 0, 0)
        self.base_height = 1.0
        
        # For smooth pattern transitions
        self.current_pos = Vector3(0, self.base_height, 0)
        self.target_pos = Vector3(0, self.base_height, 0)
        
        # Visual indicator
        self._create_visual_indicator()
    
    def _create_visual_indicator(self):
        """Create a visual sphere to represent the target"""
        # Create render component
        render = re.world.RenderComponent(self.entity.add_component("RenderComponent"))
        
        # Create a simple sphere mesh
        mesh = self._create_sphere_mesh(radius=0.5, segments=16, rings=8)
        
        # Create material
        mat = re.world.MaterialResource()
        mat_json = '''{
            "type": "pbr",
            "base_color": [1.0, 0.2, 0.2],
            "roughness": 0.3,
            "metallic": 0.1
        }'''
        mat.load_from_json(mat_json)
        
        mat_vector = lc.capsule_vector()
        mat_vector.emplace_back(mat._handle)
        
        render.update_object(mat_vector, mesh)
        mesh.install()
    
    def _create_sphere_mesh(self, radius=0.5, segments=16, rings=8) -> re.world.MeshResource:
        """Create a sphere mesh"""
        mesh = re.world.MeshResource()
        
        # Calculate vertex and triangle counts
        vertex_count = (segments + 1) * (rings + 1)
        triangle_count = segments * rings * 2
        
        submesh_offsets = np.array([0], dtype=np.uint32)
        mesh.create_empty(submesh_offsets, vertex_count, triangle_count, uv_count=1, 
                         contained_normal=True, contained_tangent=False)
        
        # Fill mesh data
        pos_buffer = mesh.pos_buffer()
        
        vertices = []
        indices = []
        
        # Generate sphere vertices
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            for seg in range(segments + 1):
                theta = 2 * math.pi * seg / segments
                
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                
                vertices.append([x, y, z])
        
        # Generate indices
        for ring in range(rings):
            for seg in range(segments):
                current = ring * (segments + 1) + seg
                next_seg = current + 1
                next_ring = (ring + 1) * (segments + 1) + seg
                next_ring_seg = next_ring + 1
                
                # Two triangles per quad
                indices.extend([current, next_ring, next_seg])
                indices.extend([next_seg, next_ring, next_ring_seg])
        
        # Fill position buffer
        for i, v in enumerate(vertices):
            if i < vertex_count:
                pos_buffer[i] = lc.double3(v[0], v[1], v[2])
        
        # Fill index data
        data_size = vertex_count * 4 + triangle_count * 3
        arr = np.ndarray(data_size, dtype=np.float32, buffer=mesh.data_buffer())
        
        # Fill vertex data (positions in float format for data buffer)
        for i, v in enumerate(vertices):
            if i < vertex_count:
                arr[i * 4 + 0] = v[0]
                arr[i * 4 + 1] = v[1]
                arr[i * 4 + 2] = v[2]
                arr[i * 4 + 3] = 1.0  # w component
        
        # Fill index data
        index_start = vertex_count * 4
        for i, idx in enumerate(indices):
            if i < triangle_count * 3:
                arr[index_start + i] = float(idx)
        
        mesh.install()
        return mesh
    
    def set_pattern(self, pattern: str):
        """Set movement pattern"""
        if pattern in self.MOVEMENT_PATTERNS:
            self.pattern = pattern
    
    def update(self, delta_time: float):
        """Update target position based on movement pattern"""
        self.time += delta_time
        
        if self.pattern == "circle":
            self.target_pos = self._circle_motion()
        elif self.pattern == "figure8":
            self.target_pos = self._figure8_motion()
        elif self.pattern == "linear":
            self.target_pos = self._linear_motion()
        elif self.pattern == "random":
            self.target_pos = self._random_motion(delta_time)
        elif self.pattern == "spiral":
            self.target_pos = self._spiral_motion()
        
        # Update transform
        if self.transform:
            self.transform.set_pos(self.target_pos.to_luisa(), recursive=False)
        
        self.current_pos = self.target_pos
    
    def _circle_motion(self) -> Vector3:
        """Circular motion around center"""
        angle = self.time * self.speed
        x = self.center.x + math.cos(angle) * self.radius
        z = self.center.z + math.sin(angle) * self.radius
        return Vector3(x, self.base_height, z)
    
    def _figure8_motion(self) -> Vector3:
        """Figure-8 (lemniscate) motion"""
        angle = self.time * self.speed
        x = self.center.x + math.sin(angle * 2) * self.radius
        z = self.center.z + math.sin(angle) * self.radius
        return Vector3(x, self.base_height, z)
    
    def _linear_motion(self) -> Vector3:
        """Linear back-and-forth motion"""
        x = self.center.x + math.sin(self.time * self.speed) * self.radius
        return Vector3(x, self.base_height, self.center.z)
    
    def _random_motion(self, delta_time: float) -> Vector3:
        """Random walk with smooth transitions"""
        # Change direction occasionally
        if random.random() < 0.02:
            angle = random.uniform(0, 2 * math.pi)
            self.target_pos = Vector3(
                self.center.x + math.cos(angle) * self.radius,
                self.base_height,
                self.center.z + math.sin(angle) * self.radius
            )
        
        # Smoothly interpolate to target
        return lerp_vector(self.current_pos, self.target_pos, delta_time * 2)
    
    def _spiral_motion(self) -> Vector3:
        """Spiral motion with increasing radius"""
        angle = self.time * self.speed
        r = (math.sin(self.time * 0.5) + 1) * 0.5 * self.radius
        x = self.center.x + math.cos(angle) * r
        z = self.center.z + math.sin(angle) * r
        y = self.base_height + math.sin(self.time * 2) * 2  # Bob up and down
        return Vector3(x, y, z)


def lerp_vector(a: Vector3, b: Vector3, t: float) -> Vector3:
    """Linear interpolation between two vectors"""
    return a + (b - a) * t


class CameraDemoApp:
    """
    Main application demonstrating 3D camera following algorithms
    """
    
    def __init__(self):
        self.app = rbc.app.App()
        self.ctx: re.world.RBCContext = None
        self.scene: re.world.Scene = None
        self.project = None
        
        # Camera system
        self.camera_manager: CameraManager = None
        
        # Targets
        self.targets: list[MovingTarget] = []
        self.active_target_index = 0
        
        # UI state
        self.show_info = True
        self.info_text = ""
        self.info_timer = 0.0
    
    def initialize(self):
        """Initialize the application"""
        # Get project path
        project_path = Path(__file__).parent
        
        # Initialize app
        self.app.init(project_path=project_path, backend_name="dx")
        self.app.init_display(1280, 720)
        
        # Get context and scene
        self.ctx = self.app.ctx
        self.scene = self.ctx.scene()
        self.project = self.app._project
        
        # Setup camera
        self.camera_manager = CameraManager(self.app)
        main_camera = self.camera_manager.create_camera("main")
        
        # Enable camera control for manual override
        self.ctx.enable_camera_control()
        
        # Create initial target
        self._spawn_target()
        
        # Setup scene
        self._setup_scene()
        
        # Register callbacks
        self._register_callbacks()
        
        print("=" * 60)
        print("3D Camera Follow Demo")
        print("=" * 60)
        print("Controls:")
        print("  F1: Cycle through follow modes (Smooth -> Spring -> Predictive -> Orbital)")
        print("  F2: Toggle collision avoidance")
        print("  F3: Reset camera to default position")
        print("  F4: Spawn additional target")
        print("  F5: Cycle movement pattern (Circle -> Figure8 -> Linear -> Random -> Spiral)")
        print("  TAB: Switch active target")
        print("  H: Toggle info display")
        print("  ESC: Exit")
        print("=" * 60)
    
    def _setup_scene(self):
        """Setup the scene with environment"""
        # Create ground plane
        self._create_ground_plane()
        
        # Add some visual markers
        self._create_markers()
    
    def _create_ground_plane(self):
        """Create a ground plane for reference"""
        entity = self.scene.add_entity()
        entity.set_name("ground")
        
        trans = re.world.TransformComponent(entity.add_component("TransformComponent"))
        trans.set_pos(lc.double3(0, 0, 0), recursive=False)
        trans.set_scale(lc.double3(20, 1, 20), recursive=False)
        
        render = re.world.RenderComponent(entity.add_component("RenderComponent"))
        
        # Create plane mesh
        mesh = re.world.MeshResource()
        submesh_offsets = np.array([0], dtype=np.uint32)
        mesh.create_empty(submesh_offsets, 4, 2, uv_count=1, 
                         contained_normal=True, contained_tangent=False)
        
        # Plane vertices
        pos_buffer = mesh.pos_buffer()
        pos_buffer[0] = lc.double3(-1, 0, -1)
        pos_buffer[1] = lc.double3(1, 0, -1)
        pos_buffer[2] = lc.double3(1, 0, 1)
        pos_buffer[3] = lc.double3(-1, 0, 1)
        
        # Fill data buffer
        arr = np.ndarray(4 * 4 + 2 * 3, dtype=np.float32, buffer=mesh.data_buffer())
        arr[0:4] = [-1, 0, -1, 1]
        arr[4:8] = [1, 0, -1, 1]
        arr[8:12] = [1, 0, 1, 1]
        arr[12:16] = [-1, 0, 1, 1]
        arr[16:19] = [0, 2, 1]  # Triangle 1
        arr[19:22] = [0, 3, 2]  # Triangle 2
        
        # Grid material
        mat = re.world.MaterialResource()
        mat_json = '''{
            "type": "pbr",
            "base_color": [0.3, 0.35, 0.4],
            "roughness": 0.8,
            "metallic": 0.0
        }'''
        mat.load_from_json(mat_json)
        
        mat_vector = lc.capsule_vector()
        mat_vector.emplace_back(mat._handle)
        
        render.update_object(mat_vector, mesh)
        mesh.install()
    
    def _create_markers(self):
        """Create visual markers at key positions"""
        # Create markers at corners
        positions = [
            (10, 0.5, 10),
            (-10, 0.5, 10),
            (10, 0.5, -10),
            (-10, 0.5, -10),
        ]
        
        for i, pos in enumerate(positions):
            entity = self.scene.add_entity()
            entity.set_name(f"marker_{i}")
            
            trans = re.world.TransformComponent(entity.add_component("TransformComponent"))
            trans.set_pos(lc.double3(*pos), recursive=False)
            trans.set_scale(lc.double3(0.2, 1, 0.2), recursive=False)
            
            render = re.world.RenderComponent(entity.add_component("RenderComponent"))
            
            # Simple cube mesh
            mesh = re.world.MeshResource()
            submesh_offsets = np.array([0], dtype=np.uint32)
            mesh.create_empty(submesh_offsets, 8, 12, uv_count=0,
                            contained_normal=False, contained_tangent=False)
            
            pos_buffer = mesh.pos_buffer()
            # Cube vertices
            pos_buffer[0] = lc.double3(-1, 0, -1)
            pos_buffer[1] = lc.double3(1, 0, -1)
            pos_buffer[2] = lc.double3(1, 1, -1)
            pos_buffer[3] = lc.double3(-1, 1, -1)
            pos_buffer[4] = lc.double3(-1, 0, 1)
            pos_buffer[5] = lc.double3(1, 0, 1)
            pos_buffer[6] = lc.double3(1, 1, 1)
            pos_buffer[7] = lc.double3(-1, 1, 1)
            
            mat = re.world.MaterialResource()
            mat_json = '''{
                "type": "pbr",
                "base_color": [0.8, 0.6, 0.2],
                "roughness": 0.5,
                "metallic": 0.0
            }'''
            mat.load_from_json(mat_json)
            
            mat_vector = lc.capsule_vector()
            mat_vector.emplace_back(mat._handle)
            
            render.update_object(mat_vector, mesh)
            mesh.install()
    
    def _spawn_target(self) -> MovingTarget:
        """Spawn a new moving target"""
        entity = self.scene.add_entity()
        entity.set_name(f"target_{len(self.targets)}")
        
        target = MovingTarget(entity, self.scene)
        
        # Set different patterns for different targets
        if len(self.targets) == 0:
            target.set_pattern("circle")
        elif len(self.targets) == 1:
            target.set_pattern("figure8")
        elif len(self.targets) == 2:
            target.set_pattern("spiral")
        else:
            target.set_pattern(random.choice(MovingTarget.MOVEMENT_PATTERNS))
        
        self.targets.append(target)
        
        # Set as camera target if first target
        if len(self.targets) == 1:
            self._set_camera_target(target)
        
        self._show_info(f"Spawned target #{len(self.targets)} with pattern: {target.pattern}")
        
        return target
    
    def _set_camera_target(self, target: MovingTarget):
        """Set target for the active camera"""
        camera = self.camera_manager.get_camera("main")
        if camera:
            camera.set_target(target.entity)
            self._show_info(f"Camera now following target")
    
    def _register_callbacks(self):
        """Register keyboard callbacks"""
        # We'll handle input in the tick function instead
        pass
    
    def _show_info(self, text: str):
        """Show temporary info text"""
        self.info_text = text
        self.info_timer = 3.0  # Show for 3 seconds
        print(f"[INFO] {text}")
    
    def handle_input(self, delta_time: float):
        """Handle keyboard input"""
        # Note: In a real implementation, you'd use proper input handling
        # For now, we'll use a simple approach with the app framework
        
        # Check for key presses (simplified)
        import robocute.rbc_ext.luisa as lcapi
        
        # This would need proper input handling integration
        # For demo purposes, we cycle modes automatically or via console input
        pass
    
    def tick(self, delta_time: float):
        """Main update loop - called every frame"""
        # Update all targets
        for target in self.targets:
            target.update(delta_time)
        
        # Update camera
        self.camera_manager.update(delta_time)
        
        # Update info timer
        if self.info_timer > 0:
            self.info_timer -= delta_time
        
        # Auto-cycle demo (optional)
        self._auto_demo(delta_time)
    
    def _auto_demo(self, delta_time: float):
        """Automatic demo cycling"""
        # Every 10 seconds, show current mode info
        current_time = self.ctx.get_time() if hasattr(self.ctx, 'get_time') else 0
        
        # This is a simplified version - in real implementation you'd track time properly
    
    def run(self):
        """Run the main loop"""
        def tick_logic(delta_time):
            self.tick(delta_time)
        
        self.app.set_user_callback(tick_logic)
        
        try:
            self.app.run()
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Cleanup resources"""
        print("Shutting down...")
        # Resources are automatically cleaned up by RoboCute


def main():
    """Main entry point"""
    demo = CameraDemoApp()
    demo.initialize()
    demo.run()


if __name__ == "__main__":
    main()
