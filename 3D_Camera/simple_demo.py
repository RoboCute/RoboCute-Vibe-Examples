"""
Simplified 3D Camera Follow Demo

A minimal example demonstrating the camera follow algorithm.
This version is easier to run and understand.
"""

import numpy as np
from pathlib import Path
import math
import time

import robocute as rbc
import robocute.rbc_ext as re
import robocute.rbc_ext.luisa as lc


class SimpleFollowCamera:
    """
    Simple smooth-follow camera implementation
    """
    
    def __init__(self, app: rbc.app.App):
        self.app = app
        self.transform = app.get_display_transform()
        
        # Camera settings
        self.offset = lc.double3(0, 5, -10)  # Behind and above
        self.smooth_speed = 3.0  # Follow speed
        
        # Current state
        self.current_pos = lc.double3(0, 5, -10)
        self.current_vel = lc.double3(0, 0, 0)
        
        # Target
        self.target_entity = None
    
    def set_target(self, entity: re.world.Entity):
        """Set entity to follow"""
        self.target_entity = entity
    
    def update(self, delta_time: float):
        """Update camera position"""
        if not self.target_entity:
            return
        
        # Get target position
        trans = re.world.TransformComponent(self.target_entity.get_component("TransformComponent"))
        target_pos = trans.position()
        
        # Calculate desired position (target + offset)
        desired_pos = lc.double3(
            target_pos.x + self.offset.x,
            target_pos.y + self.offset.y,
            target_pos.z + self.offset.z
        )
        
        # Smooth damp towards desired position
        self.current_pos, self.current_vel = self._smooth_damp(
            self.current_pos,
            desired_pos,
            self.current_vel,
            1.0 / self.smooth_speed,
            delta_time
        )
        
        # Update transform
        self.transform.set_pos(self.current_pos, recursive=False)
        
        # Look at target
        look_dir = lc.double3(
            target_pos.x - self.current_pos.x,
            target_pos.y - self.current_pos.y,
            target_pos.z - self.current_pos.z
        )
        
        # Calculate rotation to look at target
        self._look_at(look_dir)
    
    def _smooth_damp(self, current, target, velocity, smooth_time, delta_time):
        """Smooth damping - Unity-style"""
        omega = 2.0 / smooth_time
        x = omega * delta_time
        exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        
        change_x = current.x - target.x
        change_y = current.y - target.y
        change_z = current.z - target.z
        
        temp_x = velocity.x + omega * change_x * delta_time
        temp_y = velocity.y + omega * change_y * delta_time
        temp_z = velocity.z + omega * change_z * delta_time
        
        new_vel_x = (velocity.x - omega * temp_x) * exp
        new_vel_y = (velocity.y - omega * temp_y) * exp
        new_vel_z = (velocity.z - omega * temp_z) * exp
        
        result_x = target.x + (change_x + temp_x) * exp
        result_y = target.y + (change_y + temp_y) * exp
        result_z = target.z + (change_z + temp_z) * exp
        
        return lc.double3(result_x, result_y, result_z), lc.double3(new_vel_x, new_vel_y, new_vel_z)
    
    def _look_at(self, direction):
        """Make camera look at direction"""
        # Normalize direction
        length = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
        if length > 0.001:
            # Calculate yaw (rotation around Y axis)
            yaw = math.atan2(direction.x, direction.z)
            
            # Create rotation quaternion (yaw only, keeping camera level)
            half_yaw = yaw * 0.5
            quat = lc.float4(0, math.sin(half_yaw), 0, math.cos(half_yaw))
            self.transform.set_rotation(quat, recursive=False)


class MovingBall:
    """A simple moving ball for demonstration"""
    
    def __init__(self, scene: re.world.Scene, pattern="circle"):
        self.scene = scene
        self.time = 0.0
        self.pattern = pattern
        self.speed = 1.5
        self.radius = 8.0
        
        # Create entity
        self.entity = scene.add_entity()
        self.entity.set_name("moving_ball")
        
        self.transform = re.world.TransformComponent(self.entity.add_component("TransformComponent"))
        self.transform.set_pos(lc.double3(0, 1, 0), recursive=False)
        
        # Create visual
        self._create_mesh()
    
    def _create_mesh(self):
        """Create a sphere mesh"""
        render = re.world.RenderComponent(self.entity.add_component("RenderComponent"))
        
        # Create simple sphere using MeshResource
        mesh = re.world.MeshResource()
        
        # Simple cube as placeholder (sphere would need more vertices)
        submesh_offsets = np.array([0], dtype=np.uint32)
        mesh.create_empty(submesh_offsets, 8, 12, uv_count=0, 
                         contained_normal=False, contained_tangent=False)
        
        pos_buffer = mesh.pos_buffer()
        r = 0.5  # radius
        pos_buffer[0] = lc.double3(-r, 0, -r)
        pos_buffer[1] = lc.double3(r, 0, -r)
        pos_buffer[2] = lc.double3(r, 2*r, -r)
        pos_buffer[3] = lc.double3(-r, 2*r, -r)
        pos_buffer[4] = lc.double3(-r, 0, r)
        pos_buffer[5] = lc.double3(r, 0, r)
        pos_buffer[6] = lc.double3(r, 2*r, r)
        pos_buffer[7] = lc.double3(-r, 2*r, r)
        
        # Material
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
        """Update position"""
        self.time += delta_time * self.speed
        
        if self.pattern == "circle":
            x = math.cos(self.time) * self.radius
            z = math.sin(self.time) * self.radius
            y = 1.0
        elif self.pattern == "figure8":
            x = math.sin(self.time * 2) * self.radius
            z = math.sin(self.time) * self.radius
            y = 1.0 + math.sin(self.time * 4) * 0.5
        else:  # linear
            x = math.sin(self.time) * self.radius
            z = 0
            y = 1.0
        
        self.transform.set_pos(lc.double3(x, y, z), recursive=False)


def main():
    """Run simple demo"""
    print("=" * 50)
    print("Simple 3D Camera Follow Demo")
    print("=" * 50)
    
    # Initialize app
    app = rbc.app.App()
    project_path = Path(__file__).parent
    app.init(project_path=project_path, backend_name="dx")
    app.init_display(1280, 720)
    
    ctx = app.ctx
    scene = ctx.scene()
    
    # Create moving target
    ball = MovingBall(scene, pattern="circle")
    
    # Create camera controller
    camera = SimpleFollowCamera(app)
    camera.set_target(ball.entity)
    camera.smooth_speed = 4.0  # Higher = faster follow
    
    # Enable user camera control (can override)
    ctx.enable_camera_control()
    
    print("\nCamera is following a moving ball.")
    print("You can also use mouse to control the camera manually.")
    print("Close the window to exit.\n")
    
    # Main loop
    def tick_logic(delta_time):
        ball.update(delta_time)
        camera.update(delta_time)
    
    app.set_user_callback(tick_logic)
    app.run()


if __name__ == "__main__":
    main()
