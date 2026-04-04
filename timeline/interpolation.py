import numpy as np
from enum import Enum
from typing import Optional


class InterpolationType(Enum):
    """Enumeration of supported interpolation types."""
    LINEAR = "linear"
    BEZIER = "bezier"


class Interpolation:
    """
    Defines transition methods between two keyframes.
    
    Supports linear and Bezier interpolation algorithms using numpy arrays.
    
    Attributes:
        interp_type: The type of interpolation to use (LINEAR or BEZIER)
        control_points: Optional control points for Bezier interpolation (numpy array)
    """
    
    def __init__(
        self,
        interp_type: InterpolationType = InterpolationType.LINEAR,
        control_points: Optional[np.ndarray] = None
    ):
        """
        Initialize an Interpolation.
        
        Args:
            interp_type: The type of interpolation algorithm to use
            control_points: Control points for Bezier interpolation (ignored for linear)
        """
        self.interp_type = interp_type
        if control_points is not None:
            self.control_points = np.asarray(control_points, dtype=np.float64)
        else:
            self.control_points = None
    
    def __repr__(self) -> str:
        return f"Interpolation(type={self.interp_type.value}, control_points={self.control_points is not None})"
    
    def interpolate(
        self,
        start_data: np.ndarray,
        end_data: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Interpolate between start and end data at parameter t.
        
        Args:
            start_data: Starting keyframe data (numpy array)
            end_data: Ending keyframe data (numpy array)
            t: Interpolation parameter in range [0, 1]
        
        Returns:
            Interpolated data as numpy array
        """
        start = np.asarray(start_data, dtype=np.float64)
        end = np.asarray(end_data, dtype=np.float64)
        t = np.clip(float(t), 0.0, 1.0)
        
        if self.interp_type == InterpolationType.LINEAR:
            return self._linear_interpolate(start, end, t)
        elif self.interp_type == InterpolationType.BEZIER:
            return self._bezier_interpolate(start, end, t)
        else:
            raise ValueError(f"Unknown interpolation type: {self.interp_type}")
    
    def _linear_interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Perform linear interpolation between start and end.
        
        Formula: result = start + (end - start) * t
        """
        return start + (end - start) * t
    
    def _bezier_interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Perform cubic Bezier interpolation between start and end.
        
        Uses control points if provided, otherwise defaults to smooth control points.
        For cubic Bezier: B(t) = (1-t)^3 * P0 + 3(1-t)^2*t * P1 + 3(1-t)*t^2 * P2 + t^3 * P3
        where P0=start, P3=end, P1 and P2 are control points.
        """
        # Default control points if not provided
        if self.control_points is None:
            # Create default smooth control points
            # P1 = start + (end - start) * 0.33
            # P2 = start + (end - start) * 0.66
            diff = end - start
            p1 = start + diff * 0.33
            p2 = start + diff * 0.66
        else:
            # Use provided control points
            # Expected shape: (2, n) where control_points[0] = P1, control_points[1] = P2
            cp = self.control_points
            if cp.shape[0] == 2:
                p1 = np.asarray(cp[0], dtype=np.float64)
                p2 = np.asarray(cp[1], dtype=np.float64)
            else:
                # Single control point - use it for both
                p1 = p2 = np.asarray(cp[0] if len(cp.shape) > 1 else cp, dtype=np.float64)
        
        # Cubic Bezier formula
        one_minus_t = 1.0 - t
        result = (
            one_minus_t ** 3 * start +
            3 * one_minus_t ** 2 * t * p1 +
            3 * one_minus_t * t ** 2 * p2 +
            t ** 3 * end
        )
        return result
    
    def sample(
        self,
        start_data: np.ndarray,
        end_data: np.ndarray,
        num_samples: int
    ) -> np.ndarray:
        """
        Sample multiple points along the interpolation path.
        
        Args:
            start_data: Starting keyframe data (numpy array)
            end_data: Ending keyframe data (numpy array)
            num_samples: Number of samples to generate (including start and end)
        
        Returns:
            Array of interpolated samples with shape (num_samples, data_dim)
        """
        samples = []
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0.0
            sample = self.interpolate(start_data, end_data, t)
            samples.append(sample)
        return np.array(samples)
