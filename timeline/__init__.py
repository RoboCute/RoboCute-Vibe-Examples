"""
Timeline module for managing keyframe-based animations.
"""

from .keyframe import KeyFrame
from .interpolation import Interpolation, InterpolationType
from .timeline import Timeline

__all__ = ['KeyFrame', 'Interpolation', 'InterpolationType', 'Timeline']
