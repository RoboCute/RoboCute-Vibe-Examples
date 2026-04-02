"""
Moving Target Entity for Camera Follow Demos

Provides a configurable moving target that can follow various patterns:
- circle: Circular motion around a center point
- figure8: Figure-8 (lemniscate) motion
- linear: Back-and-forth linear motion
- spiral: Spiral motion with varying radius
- random: Random walk with smooth transitions
"""

import math
import random
from typing import Optional, Callable
from enum import Enum, auto
from dataclasses import dataclass

import numpy as np
import robocute as rbc
import robocute.rbc_ext as re
import robocute.rbc_ext.luisa as lc

from camera_math import Vector3, lerp_vector
from mesh_builder import MeshBuilder


class MovementPattern(Enum):
    """Available movement patterns"""
    CIRCLE = auto()
    FIGURE8 = auto()
    LINEAR = auto()
    SPIRAL = auto()
    RANDOM = auto()


@dataclass
class TargetConfig:
    """Configuration for moving target"""
    pattern: MovementPattern = MovementPattern.CIRCLE
    speed: float = 2.0
    radius: float = 10.0
    base_height: float = 1.0
    center: Vector3 = None
    
    def __post_init__(self):
        if self.center is None:
            self.center = Vector3(0, 0, 0)


class MovingTarget:
    """
    A moving target entity for camera follow demonstrations.
    Creates a visual mesh and moves it in configurable patterns.
    """
    
    def __init__(
        self,
        scene: re.world.Scene,
        name: str = "moving_target",
        config: Optional[TargetConfig] = None
    ):
        self.scene = scene
        self.name = name
        self.config = config or TargetConfig()
        
        # Movement state
        self.time = 0.0
        self.current_position = Vector3(0, self.config.base_height, 0)
        self.velocity = Vector3(0, 0, 0)
        self.last_position = self.current_position.copy()
        
        # Random walk state
        self.random_target = self.current_position.copy()
        
        # Create entity and visual
        print(0)
        self.entity = self._create_entity()
        print(1)
        self.transform = re.world.TransformComponent(
            self.entity.get_component("TransformComponent")
        )
        print(2)
        
        # Event callbacks
        self.on_position_changed: Optional[Callable[[Vector3], None]] = None
    
    def _create_entity(self) -> re.world.Entity:
        """Create the entity with visual mesh"""
        entity = self.scene.add_entity()
        entity.set_name(self.name)
        
        # Add transform
        trans = re.world.TransformComponent(entity.add_component("TransformComponent"))
        trans.set_pos(
            lc.double3(
                self.current_position.x,
                self.current_position.y,
                self.current_position.z
            ),
            recursive=False
        )
        # Create visual mesh (sphere-like)
        self._create_sphere_mesh(entity)
        
        return entity
    
    def _create_sphere_mesh(self, entity: re.world.Entity):
        """Create a simple sphere mesh using MeshBuilder"""
        render = re.world.RenderComponent(entity.add_component("RenderComponent"))
        
        # Create mesh using MeshBuilder
        builder = MeshBuilder()
        
        # Sphere parameters
        segments = 16
        rings = 8
        radius = 0.5
        
        # Add a submesh for the sphere
        submesh_idx = builder.add_submesh()
        
        # Add a UV set
        uv_set_idx = builder.add_uv_set()
        
        # Generate sphere vertices
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            v_coord = 1.0 - (ring / rings)  # Flip V for correct texture orientation
            
            for seg in range(segments + 1):
                theta = 2 * math.pi * seg / segments
                u_coord = seg / segments
                
                # Calculate position
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                
                # Add vertex
                vertex_idx = builder.add_vertex((x, y, z))
                
                # Calculate normal (normalized position for sphere)
                normal = np.array([x, y, z], dtype=np.float32)
                normal_len = np.linalg.norm(normal)
                if normal_len > 0:
                    normal = normal / normal_len
                
                # Add normal
                if builder.normal.shape[0] == 0:
                    builder.normal = normal.reshape(1, 3)
                else:
                    builder.normal = np.vstack([builder.normal, normal.reshape(1, 3)])
                
                # Add UV
                uv = np.array([u_coord, v_coord], dtype=np.float32).reshape(1, 2)
                if builder.uvs[uv_set_idx].shape[0] == 0:
                    builder.uvs[uv_set_idx] = uv
                else:
                    builder.uvs[uv_set_idx] = np.vstack([builder.uvs[uv_set_idx], uv])
        
        # Generate triangles
        for ring in range(rings):
            for seg in range(segments):
                current = ring * (segments + 1) + seg
                next_seg = current + 1
                next_ring = (ring + 1) * (segments + 1) + seg
                next_ring_seg = next_ring + 1
                
                # Add two triangles per quad
                builder.add_triangle(submesh_idx, current, next_ring, next_seg)
                builder.add_triangle(submesh_idx, next_seg, next_ring, next_ring_seg)
        
        # Validate and write mesh
        mesh = builder.write_to_mesh()
        
        # Create material (red ball)
        mat = re.world.MaterialResource()
        mat_json = '''{
            "type": "pbr",
            "base_color": [0.9, 0.2, 0.2],
            "roughness": 0.3,
            "metallic": 0.1
        }'''
        mat.load_from_json(mat_json)
        print('finished')
        mat_vector = lc.capsule_vector()
        print(2)
        mat_vector.emplace_back(mat._handle)
        print(3)
        mesh.install()
        print(6)
        render.update_object(mat_vector, mesh)
        print(4)
        print(5)
    
    def update(self, delta_time: float):
        """Update target position based on movement pattern"""
        self.time += delta_time * self.config.speed
        self.last_position = self.current_position.copy()
        
        # Calculate new position based on pattern
        if self.config.pattern == MovementPattern.CIRCLE:
            new_pos = self._circle_motion()
        elif self.config.pattern == MovementPattern.FIGURE8:
            new_pos = self._figure8_motion()
        elif self.config.pattern == MovementPattern.LINEAR:
            new_pos = self._linear_motion()
        elif self.config.pattern == MovementPattern.SPIRAL:
            new_pos = self._spiral_motion()
        elif self.config.pattern == MovementPattern.RANDOM:
            new_pos = self._random_motion(delta_time)
        else:
            new_pos = self._circle_motion()
        
        self.current_position = new_pos
        
        # Calculate velocity
        if delta_time > 0:
            self.velocity = (self.current_position - self.last_position) / delta_time
        
        # Update transform
        self.transform.set_pos(
            lc.double3(
                self.current_position.x,
                self.current_position.y,
                self.current_position.z
            ),
            recursive=False
        )
        
        # Trigger callback if set
        if self.on_position_changed:
            self.on_position_changed(self.current_position)
    
    def _circle_motion(self) -> Vector3:
        """Circular motion around center"""
        angle = self.time
        x = self.config.center.x + math.cos(angle) * self.config.radius
        z = self.config.center.z + math.sin(angle) * self.config.radius
        return Vector3(x, self.config.base_height, z)
    
    def _figure8_motion(self) -> Vector3:
        """Figure-8 (lemniscate) motion"""
        angle = self.time
        x = self.config.center.x + math.sin(angle * 2) * self.config.radius
        z = self.config.center.z + math.sin(angle) * self.config.radius
        return Vector3(x, self.config.base_height, z)
    
    def _linear_motion(self) -> Vector3:
        """Linear back-and-forth motion"""
        x = self.config.center.x + math.sin(self.time) * self.config.radius
        return Vector3(x, self.config.base_height, self.config.center.z)
    
    def _spiral_motion(self) -> Vector3:
        """Spiral motion with varying radius"""
        angle = self.time
        r = (math.sin(self.time * 0.5) + 1) * 0.5 * self.config.radius
        x = self.config.center.x + math.cos(angle) * r
        z = self.config.center.z + math.sin(angle) * r
        y = self.config.base_height + math.sin(self.time * 2) * 2
        return Vector3(x, y, z)
    
    def _random_motion(self, delta_time: float) -> Vector3:
        """Random walk with smooth transitions"""
        # Change target occasionally
        if random.random() < 0.02:
            angle = random.uniform(0, 2 * math.pi)
            self.random_target = Vector3(
                self.config.center.x + math.cos(angle) * self.config.radius,
                self.config.base_height,
                self.config.center.z + math.sin(angle) * self.config.radius
            )
        
        # Smoothly interpolate to target
        return lerp_vector(self.current_position, self.random_target, delta_time * 2)
    
    def set_pattern(self, pattern: MovementPattern):
        """Change movement pattern"""
        self.config.pattern = pattern
        self.time = 0.0  # Reset time for consistent start
    
    def set_speed(self, speed: float):
        """Set movement speed"""
        self.config.speed = speed
    
    def set_radius(self, radius: float):
        """Set movement radius"""
        self.config.radius = radius
    
    def get_position(self) -> Vector3:
        """Get current position"""
        return self.current_position.copy()
    
    def get_velocity(self) -> Vector3:
        """Get current velocity"""
        return self.velocity.copy()
    
    def reset(self):
        """Reset to starting position"""
        self.time = 0.0
        self.current_position = Vector3(0, self.config.base_height, 0)
        self.velocity = Vector3(0, 0, 0)
        self.transform.set_pos(
            lc.double3(
                self.current_position.x,
                self.current_position.y,
                self.current_position.z
            ),
            recursive=False
        )


def create_moving_target(
    scene: re.world.Scene,
    name: str = "target",
    pattern: str = "circle",
    speed: float = 2.0,
    radius: float = 10.0
) -> MovingTarget:
    """
    Convenience factory function to create a moving target
    
    Args:
        scene: The scene to add the target to
        name: Entity name
        pattern: One of "circle", "figure8", "linear", "spiral", "random"
        speed: Movement speed multiplier
        radius: Movement radius
    
    Returns:
        Configured MovingTarget instance
    """
    pattern_map = {
        "circle": MovementPattern.CIRCLE,
        "figure8": MovementPattern.FIGURE8,
        "linear": MovementPattern.LINEAR,
        "spiral": MovementPattern.SPIRAL,
        "random": MovementPattern.RANDOM,
    }
    
    config = TargetConfig(
        pattern=pattern_map.get(pattern.lower(), MovementPattern.CIRCLE),
        speed=speed,
        radius=radius
    )
    
    return MovingTarget(scene, name, config)


# Example usage and test
if __name__ == "__main__":
    print("Moving Target Entity Module")
    print("=" * 40)
    print("This module provides MovingTarget class for camera follow demos.")
    print()
    print("Usage:")
    print("  from moving_target import MovingTarget, TargetConfig, MovementPattern")
    print("  ")
    print("  # Create with default config")
    print("  target = MovingTarget(scene, 'my_target')")
    print("  ")
    print("  # Create with custom config")
    print("  config = TargetConfig(")
    print("      pattern=MovementPattern.FIGURE8,")
    print("      speed=3.0,")
    print("      radius=15.0")
    print("  )")
    print("  target = MovingTarget(scene, 'target', config)")
    print("  ")
    print("  # In update loop")
    print("  target.update(delta_time)")
    print()
    print("Available patterns:")
    for p in MovementPattern:
        print(f"  - {p.name.lower()}")
