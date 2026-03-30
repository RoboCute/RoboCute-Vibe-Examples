"""
3D Camera Following Algorithms for RoboCute

This module implements various camera following algorithms:
1. Smooth Follow (Lerp-based)
2. Spring-Damped Follow (Physics-based)
3. Predictive Follow (Look-ahead)
4. Orbital Follow (Fixed distance with rotation)
"""

import numpy as np
import math
from typing import Optional, Callable

import robocute as rbc
import robocute.rbc_ext as re
import robocute.rbc_ext.luisa as lc

# Import core math from separate module
from camera_math import Vector3, lerp, lerp_vector, smooth_damp, FollowMode


# Add Luisa conversion methods to Vector3 (monkey patch for compatibility)
def _to_luisa(self):
    """Convert to Luisa double3"""
    return lc.double3(self.x, self.y, self.z)


def _from_luisa(cls, v):
    """Create from Luisa vector"""
    return cls(v.x, v.y, v.z)


Vector3.to_luisa = _to_luisa
Vector3.from_luisa = classmethod(_from_luisa)


class CameraController:
    """
    Advanced 3D Camera Controller with multiple follow modes
    
    Features:
    - Smooth follow with configurable parameters
    - Spring-damped physics-based follow
    - Predictive look-ahead follow
    - Orbital follow with rotation
    - Collision avoidance
    - Smooth rotation interpolation
    """
    
    def __init__(self, app: rbc.app.App):
        self.app = app
        self.mode = FollowMode.SMOOTH
        
        # Target settings
        self.target_entity: Optional[re.world.Entity] = None
        self.target_position = Vector3(0, 0, 0)
        self.target_velocity = Vector3(0, 0, 0)
        self.target_last_position = Vector3(0, 0, 0)
        
        # Offset settings
        self.offset = Vector3(0, 5, -10)  # Default: behind and above target
        self.min_distance = 2.0
        self.max_distance = 50.0
        
        # Smooth follow parameters
        self.position_smooth_speed = 5.0
        self.rotation_smooth_speed = 5.0
        self.use_fixed_height = False
        self.fixed_height = 10.0
        
        # Spring system parameters
        self.spring_stiffness = 150.0
        self.spring_damping = 10.0
        self.spring_mass = 1.0
        self.spring_velocity = Vector3(0, 0, 0)
        
        # Predictive follow parameters
        self.look_ahead_factor = 0.3
        self.enable_prediction = False
        
        # Orbital parameters
        self.orbit_distance = 10.0
        self.orbit_height = 5.0
        self.orbit_angle = 0.0  # Horizontal angle in radians
        self.orbit_speed = 1.0  # Orbit rotation speed
        
        # Collision avoidance
        self.enable_collision_avoidance = True
        self.collision_buffer = 0.5
        
        # Internal state
        self.current_position = Vector3(0, 5, -10)
        self.current_velocity = Vector3(0, 0, 0)
        self.current_rotation = Vector3(0, 0, 0)  # Euler angles
        
        # Display transform reference
        self._transform = None
    
    def initialize(self):
        """Initialize camera controller with display transform"""
        self._transform = self.app.get_display_transform()
        if self._transform:
            pos = self._transform.position()
            self.current_position = Vector3(pos.x, pos.y, pos.z)
    
    def set_target(self, target: re.world.Entity):
        """Set the target entity to follow"""
        self.target_entity = target
        if target:
            # Get initial target position
            trans = re.world.TransformComponent(target.get_component("TransformComponent"))
            if trans:
                pos = trans.position()
                self.target_position = Vector3(pos.x, pos.y, pos.z)
                self.target_last_position = self.target_position.copy()
                
                # Initialize camera position relative to target
                self.current_position = self.target_position + self.offset
                self._update_transform_position()
    
    def set_mode(self, mode: FollowMode):
        """Switch follow mode"""
        self.mode = mode
        # Reset spring velocity when switching modes
        if mode == FollowMode.SPRING:
            self.spring_velocity = Vector3(0, 0, 0)
    
    def update(self, delta_time: float):
        """
        Update camera position and rotation
        Call this every frame
        """
        if not self.target_entity:
            return
        
        # Get current target position
        self._update_target_position(delta_time)
        
        # Calculate desired position based on follow mode
        if self.mode == FollowMode.SMOOTH:
            desired_pos = self._calculate_smooth_position()
            self.current_position = self._apply_smooth_follow(desired_pos, delta_time)
        
        elif self.mode == FollowMode.SPRING:
            desired_pos = self._calculate_smooth_position()
            self.current_position = self._apply_spring_follow(desired_pos, delta_time)
        
        elif self.mode == FollowMode.PREDICTIVE:
            desired_pos = self._calculate_predictive_position(delta_time)
            self.current_position = self._apply_smooth_follow(desired_pos, delta_time)
        
        elif self.mode == FollowMode.ORBITAL:
            desired_pos = self._calculate_orbital_position(delta_time)
            self.current_position = self._apply_smooth_follow(desired_pos, delta_time)
        
        # Apply collision avoidance
        if self.enable_collision_avoidance:
            self.current_position = self._apply_collision_avoidance()
        
        # Clamp distance
        self._clamp_distance()
        
        # Update transform
        self._update_transform_position()
        self._update_transform_rotation()
    
    def _update_target_position(self, delta_time: float):
        """Update target position and calculate velocity"""
        if not self.target_entity:
            return
        
        trans = re.world.TransformComponent(self.target_entity.get_component("TransformComponent"))
        if trans:
            pos = trans.position()
            new_pos = Vector3(pos.x, pos.y, pos.z)
            
            # Calculate velocity
            if delta_time > 0:
                self.target_velocity = (new_pos - self.target_last_position) / delta_time
            
            self.target_last_position = self.target_position.copy()
            self.target_position = new_pos
    
    def _calculate_smooth_position(self) -> Vector3:
        """Calculate desired position with offset"""
        desired = self.target_position + self.offset
        
        if self.use_fixed_height:
            desired.y = self.fixed_height
        
        return desired
    
    def _calculate_predictive_position(self, delta_time: float) -> Vector3:
        """Calculate predictive position with look-ahead"""
        base_pos = self.target_position + self.offset
        
        if self.use_fixed_height:
            base_pos.y = self.fixed_height
        
        # Add velocity prediction
        prediction = self.target_velocity * self.look_ahead_factor
        return base_pos + prediction
    
    def _calculate_orbital_position(self, delta_time: float) -> Vector3:
        """Calculate orbital position around target"""
        # Update orbit angle
        self.orbit_angle += self.orbit_speed * delta_time
        
        # Calculate position on circle
        x = math.cos(self.orbit_angle) * self.orbit_distance
        z = math.sin(self.orbit_angle) * self.orbit_distance
        
        return Vector3(
            self.target_position.x + x,
            self.target_position.y + self.orbit_height,
            self.target_position.z + z
        )
    
    def _apply_smooth_follow(self, desired_pos: Vector3, delta_time: float) -> Vector3:
        """Apply smooth damping to move towards desired position"""
        smooth_time = 1.0 / self.position_smooth_speed
        new_pos, new_vel = smooth_damp(
            self.current_position, 
            desired_pos, 
            self.current_velocity,
            smooth_time, 
            delta_time
        )
        self.current_velocity = new_vel
        return new_pos
    
    def _apply_spring_follow(self, desired_pos: Vector3, delta_time: float) -> Vector3:
        """Apply spring physics to follow target"""
        # Calculate displacement from desired position
        displacement = self.current_position - desired_pos
        
        # Spring force: F = -k * displacement - c * velocity
        spring_force = displacement * -self.spring_stiffness
        damping_force = self.spring_velocity * -self.spring_damping
        
        total_force = spring_force + damping_force
        
        # F = ma, so a = F/m
        acceleration = total_force / self.spring_mass
        
        # Integrate
        self.spring_velocity = self.spring_velocity + acceleration * delta_time
        new_pos = self.current_position + self.spring_velocity * delta_time
        
        return new_pos
    
    def _apply_collision_avoidance(self) -> Vector3:
        """Simple collision avoidance by pushing camera towards target"""
        direction = self.current_position - self.target_position
        distance = direction.length()
        
        # If too close, push back
        if distance < self.min_distance + self.collision_buffer:
            push_dir = direction.normalized()
            if push_dir.length() < 0.001:
                push_dir = Vector3(0, 1, 0)
            target_dist = self.min_distance + self.collision_buffer
            return self.target_position + push_dir * target_dist
        
        return self.current_position
    
    def _clamp_distance(self):
        """Keep camera within min/max distance from target"""
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
    
    def _update_transform_position(self):
        """Update the display transform position"""
        if self._transform:
            self._transform.set_pos(self.current_position.to_luisa(), recursive=False)
    
    def _update_transform_rotation(self):
        """Update camera rotation to look at target"""
        if not self._transform or not self.target_entity:
            return
        
        # Calculate look direction
        look_dir = self.target_position - self.current_position
        look_dir.y = 0  # Keep horizontal rotation only
        
        if look_dir.length() > 0.001:
            # Calculate yaw angle
            yaw = math.atan2(look_dir.x, look_dir.z)
            
            # Smooth rotation interpolation
            current_yaw = self.current_rotation.y
            yaw_diff = yaw - current_yaw
            
            # Normalize angle difference to [-pi, pi]
            while yaw_diff > math.pi:
                yaw_diff -= 2 * math.pi
            while yaw_diff < -math.pi:
                yaw_diff += 2 * math.pi
            
            # Apply smooth rotation
            new_yaw = current_yaw + yaw_diff * self.rotation_smooth_speed * 0.016
            self.current_rotation.y = new_yaw
            
            # Create rotation quaternion (yaw only)
            half_yaw = new_yaw * 0.5
            quat = lc.float4(0, math.sin(half_yaw), 0, math.cos(half_yaw))
            self._transform.set_rotation(quat, recursive=False)
    
    def get_camera_info(self) -> dict:
        """Get current camera state information"""
        return {
            "mode": self.mode.name,
            "position": str(self.current_position),
            "target_position": str(self.target_position),
            "distance_to_target": (self.current_position - self.target_position).length(),
            "target_velocity": str(self.target_velocity),
        }
    
    def set_offset(self, x: float, y: float, z: float):
        """Set camera offset from target"""
        self.offset = Vector3(x, y, z)
    
    def set_orbital_params(self, distance: float, height: float, speed: float):
        """Set orbital follow parameters"""
        self.orbit_distance = distance
        self.orbit_height = height
        self.orbit_speed = speed


class CameraManager:
    """
    Manager for multiple camera controllers or camera states
    """
    
    def __init__(self, app: rbc.app.App):
        self.app = app
        self.cameras: dict[str, CameraController] = {}
        self.active_camera: Optional[str] = None
    
    def create_camera(self, name: str) -> CameraController:
        """Create a new named camera controller"""
        camera = CameraController(self.app)
        camera.initialize()
        self.cameras[name] = camera
        if not self.active_camera:
            self.active_camera = name
        return camera
    
    def get_camera(self, name: str) -> Optional[CameraController]:
        """Get camera by name"""
        return self.cameras.get(name)
    
    def set_active_camera(self, name: str):
        """Set active camera"""
        if name in self.cameras:
            self.active_camera = name
    
    def update(self, delta_time: float):
        """Update active camera"""
        if self.active_camera and self.active_camera in self.cameras:
            self.cameras[self.active_camera].update(delta_time)
    
    def cycle_mode(self):
        """Cycle through follow modes on active camera"""
        if self.active_camera and self.active_camera in self.cameras:
            camera = self.cameras[self.active_camera]
            modes = list(FollowMode)
            current_idx = modes.index(camera.mode)
            next_idx = (current_idx + 1) % len(modes)
            camera.set_mode(modes[next_idx])
            return camera.mode.name
        return None
