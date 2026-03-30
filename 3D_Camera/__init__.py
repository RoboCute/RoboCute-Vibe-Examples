"""
3D Camera Follow Module for RoboCute

This module provides camera following algorithms for 3D applications.

Example:
    >>> from camera_controller import CameraController, FollowMode
    >>> camera = CameraController(app)
    >>> camera.set_target(target_entity)
    >>> camera.set_mode(FollowMode.SMOOTH)
    >>> camera.update(delta_time)
"""

from .camera_math import (
    Vector3,
    lerp,
    lerp_vector,
    smooth_damp,
    FollowMode,
)

# CameraController requires robocute, import separately if needed
try:
    from .camera_controller import (
        CameraController,
        CameraManager,
    )
    __all__ = [
        "CameraController",
        "CameraManager",
        "FollowMode",
        "Vector3",
        "lerp",
        "lerp_vector",
        "smooth_damp",
    ]
except ImportError:
    # robocute not available, export only core math
    __all__ = [
        "FollowMode",
        "Vector3",
        "lerp",
        "lerp_vector",
        "smooth_damp",
    ]

__version__ = "1.0.0"
