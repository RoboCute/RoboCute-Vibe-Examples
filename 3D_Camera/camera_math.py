"""
Core camera math utilities - No external dependencies
"""

import math
from dataclasses import dataclass
from enum import Enum, auto


class FollowMode(Enum):
    """Camera follow modes"""
    SMOOTH = auto()      # Smooth interpolation follow
    SPRING = auto()      # Spring-damped physics follow
    PREDICTIVE = auto()  # Predictive look-ahead follow
    ORBITAL = auto()     # Orbital rotation follow


@dataclass
class Vector3:
    """3D Vector with math operations"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
        return Vector3(self.x + other, self.y + other, self.z + other)
    
    def __sub__(self, other):
        if isinstance(other, Vector3):
            return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vector3(self.x - other, self.y - other, self.z - other)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def length_squared(self):
        return self.x**2 + self.y**2 + self.z**2
    
    def normalized(self):
        length = self.length()
        if length < 1e-6:
            return Vector3(0, 0, 0)
        return self / length
    
    def copy(self):
        return Vector3(self.x, self.y, self.z)
    
    def __repr__(self):
        return f"Vector3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between two values"""
    return a + (b - a) * t


def lerp_vector(a: Vector3, b: Vector3, t: float) -> Vector3:
    """Linear interpolation between two vectors"""
    return a + (b - a) * t


def smooth_damp(current: Vector3, target: Vector3, velocity: Vector3, 
                smooth_time: float, delta_time: float) -> tuple[Vector3, Vector3]:
    """
    Smooth damping - critically damped spring approach
    Returns (new_position, new_velocity)
    
    This implementation uses a critically damped spring system that
    is more numerically stable than the exact Unity implementation.
    """
    # Handle edge cases
    if delta_time < 1e-6:
        # No time passed, return current position
        return current.copy(), velocity.copy()
    
    if smooth_time < 0.001:
        smooth_time = 0.001
    
    # Spring constants for critical damping
    # omega = 2 / smooth_time gives us the response time
    omega = 2.0 / smooth_time
    
    # Calculate displacement from target
    displacement = target - current
    
    # Spring force (proportional to displacement)
    spring_accel = displacement * (omega * omega)
    
    # Damping force (proportional to velocity, critically damped)
    # For critical damping: damping = 2 * sqrt(mass * stiffness)
    # Since we're using omega directly: damping = 2 * omega
    damping_accel = velocity * (-2.0 * omega)
    
    # Total acceleration
    acceleration = spring_accel + damping_accel
    
    # Semi-implicit Euler integration (more stable)
    new_velocity = velocity + acceleration * delta_time
    new_position = current + new_velocity * delta_time
    
    return new_position, new_velocity
