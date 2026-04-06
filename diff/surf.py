"""
SURF (Speeded-Up Robust Features) implementation for image difference detection.

This module provides SURF-based image comparison capabilities for detecting
structural differences between images, with scale and rotation invariance.
SURF is faster than SIFT due to the use of integral images and Haar wavelets.

Reference:
    Bay, H., Ess, A., Tuytelaars, T., & Van Gool, L. (2008).
    Speeded-up robust features (SURF).
    Computer vision and image understanding, 110(3), 346-359.

Example:
    >>> import numpy as np
    >>> from diff.surf import SURF, surf_diff
    >>> 
    >>> # Create two images
    >>> img1 = np.random.rand(256, 256)
    >>> img2 = img1 + np.random.randn(256, 256) * 0.1
    >>> 
    >>> # Compute SURF difference
    >>> result = surf_diff(img1, img2)
    >>> print(f"Match ratio: {result['match_ratio']:.3f}")

Attributes:
    __all__: List of exported functions and classes.

Author: RoboCute Team
Version: 1.0.0
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import IntEnum
import math
import warnings

__all__ = [
    'SURFInputError',
    'SURFConfig',
    'SURFKeypoint',
    'SURFMatch',
    'IntegralImage',
    'FastHessian',
    'SURF',
    'OpenCVSURF',
    'SURFMatcher',
    'SURFDiff',
    'surf_diff',
    'compute_surf_diff',
    'create_surf',
    'has_opencv_surf',
]


def has_opencv_surf() -> bool:
    """
    Check if OpenCV SURF is available (requires opencv-contrib-python).
    
    Returns:
        True if cv2.xfeatures2d.SURF_create() is available
    
    Example:
        >>> from diff.surf import has_opencv_surf
        >>> if has_opencv_surf():
        ...     print("OpenCV SURF is available")
        ... else:
        ...     print("Using pure Python fallback")
    """
    try:
        import cv2
        # Check if SURF_create is available (in contrib module, may be patented)
        return hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SURF_create')
    except ImportError:
        return False


def _has_opencv_surf() -> bool:
    """Check if OpenCV SURF is available (requires opencv-contrib-python)."""
    try:
        import cv2
        # Check if SURF_create is available (in contrib module, may be patented)
        return hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SURF_create')
    except ImportError:
        return False


def _cv_keypoint_to_custom(cv_kp: 'cv2.KeyPoint', descriptor: Optional[np.ndarray] = None) -> SURFKeypoint:
    """Convert OpenCV KeyPoint to custom SURFKeyPoint."""
    return SURFKeypoint(
        x=cv_kp.pt[0],
        y=cv_kp.pt[1],
        octave=cv_kp.octave,
        scale_level=0,  # OpenCV doesn't expose this directly
        size=cv_kp.size,
        orientation=np.radians(cv_kp.angle),
        response=cv_kp.response,
        laplacian=0,  # OpenCV doesn't expose this directly
        descriptor=descriptor
    )


class OpenCVSURF:
    """
    OpenCV-based SURF implementation using cv2.xfeatures2d.SURF_create().
    
    This class provides a wrapper around OpenCV's SURF implementation from
    opencv-contrib-python. It automatically falls back to the pure Python
    implementation if OpenCV SURF is not available.
    
    Note:
        SURF is patented and requires opencv-contrib-python to be installed.
        If not available, the class will fall back to the pure Python SURF.
    
    Example:
        >>> import numpy as np
        >>> from diff.surf import OpenCVSURF
        >>> 
        >>> img = np.random.rand(256, 256).astype(np.float32)
        >>> surf = OpenCVSURF(hessian_threshold=400)
        >>> keypoints = surf.detect_and_compute(img)
        >>> print(f"Found {len(keypoints)} keypoints")
        
        # With fallback
        >>> surf = OpenCVSURF(fallback=True)  # Uses pure Python if OpenCV unavailable
    
    Attributes:
        config: SURFConfig instance with algorithm parameters
        _surf: OpenCV SURF detector or None if unavailable
        _fallback_surf: Pure Python SURF instance for fallback
        _use_fallback: Whether to use fallback implementation
    """
    
    def __init__(
        self,
        hessian_threshold: float = 100.0,
        n_octaves: int = 4,
        n_octave_layers: int = 3,
        extended: bool = True,
        upright: bool = False,
        config: Optional[SURFConfig] = None,
        fallback: bool = True
    ):
        """
        Initialize OpenCV SURF with optional fallback.
        
        Args:
            hessian_threshold: Threshold for Hessian keypoint detector
            n_octaves: Number of pyramid octaves
            n_octave_layers: Number of layers per octave
            extended: Use extended 128-element descriptor (vs 64)
            upright: Compute upright descriptors (no orientation)
            config: SURFConfig instance (alternative to individual params)
            fallback: If True, use pure Python SURF when OpenCV unavailable
        """
        self._use_fallback = False
        self._fallback_surf: Optional[SURF] = None
        self._surf = None
        
        # Use config if provided, otherwise create from parameters
        if config is not None:
            self.config = config
        else:
            self.config = SURFConfig(
                hessian_threshold=hessian_threshold,
                n_octaves=n_octaves,
                n_octave_layers=n_octave_layers,
                extended=extended,
                upright=upright
            )
        
        # Try to initialize OpenCV SURF
        if _has_opencv_surf():
            try:
                import cv2
                self._surf = cv2.xfeatures2d.SURF_create(
                    hessianThreshold=self.config.hessian_threshold,
                    nOctaves=self.config.n_octaves,
                    nOctaveLayers=self.config.n_octave_layers,
                    extended=self.config.extended,
                    upright=self.config.upright
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize OpenCV SURF: {e}. "
                            f"{'Will fallback to pure Python.' if fallback else 'No fallback enabled.'}")
                if fallback:
                    self._use_fallback = True
        elif fallback:
            self._use_fallback = True
            warnings.warn("OpenCV SURF not available (requires opencv-contrib-python). "
                         "Using pure Python fallback implementation.")
        else:
            raise RuntimeError(
                "OpenCV SURF is not available. Install opencv-contrib-python "
                "or enable fallback=True to use pure Python implementation."
            )
        
        # Initialize fallback if needed
        if self._use_fallback:
            self._fallback_surf = SURF(self.config)
    
    @property
    def is_opencv(self) -> bool:
        """Return True if using OpenCV implementation, False if using fallback."""
        return self._surf is not None and not self._use_fallback
    
    @property
    def is_fallback(self) -> bool:
        """Return True if using fallback implementation."""
        return self._use_fallback
    
    def detect(self, image: np.ndarray) -> List[SURFKeypoint]:
        """
        Detect keypoints in an image.
        
        Args:
            image: Input image as numpy array (2D or 3D)
        
        Returns:
            List of SURFKeypoint objects
        """
        if self._use_fallback or self._fallback_surf is not None:
            return self._fallback_surf.detect(image)
        
        # Convert image to uint8 for OpenCV if needed
        img_uint8 = self._prepare_image(image)
        
        # Detect keypoints
        cv_kps = self._surf.detect(img_uint8, None)
        
        # Convert to custom format
        return [_cv_keypoint_to_custom(kp) for kp in cv_kps]
    
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[SURFKeypoint]
    ) -> List[SURFKeypoint]:
        """
        Compute descriptors for given keypoints.
        
        Args:
            image: Input image as numpy array
            keypoints: List of keypoints to compute descriptors for
        
        Returns:
            List of SURFKeypoint objects with descriptors
        """
        if self._use_fallback or self._fallback_surf is not None:
            return self._fallback_surf.compute(image, keypoints)
        
        # Convert image to uint8 for OpenCV if needed
        img_uint8 = self._prepare_image(image)
        
        # Convert custom keypoints to OpenCV format
        cv_kps = self._custom_to_cv_keypoints(keypoints)
        
        # Compute descriptors
        cv_kps_out, descriptors = self._surf.compute(img_uint8, cv_kps)
        
        # Convert back to custom format with descriptors
        result = []
        for i, kp in enumerate(cv_kps_out):
            desc = descriptors[i] if descriptors is not None and i < len(descriptors) else None
            result.append(_cv_keypoint_to_custom(kp, desc))
        
        return result
    
    def detect_and_compute(self, image: np.ndarray) -> List[SURFKeypoint]:
        """
        Detect keypoints and compute descriptors in one call.
        
        Args:
            image: Input image as numpy array (2D or 3D)
        
        Returns:
            List of SURFKeypoint objects with descriptors
        """
        if self._use_fallback or self._fallback_surf is not None:
            return self._fallback_surf.detect_and_compute(image)
        
        # Convert image to uint8 for OpenCV if needed
        img_uint8 = self._prepare_image(image)
        
        # Detect and compute
        cv_kps, descriptors = self._surf.detectAndCompute(img_uint8, None)
        
        if cv_kps is None or len(cv_kps) == 0:
            return []
        
        # Convert to custom format
        result = []
        for i, kp in enumerate(cv_kps):
            desc = descriptors[i] if descriptors is not None else None
            result.append(_cv_keypoint_to_custom(kp, desc))
        
        return result
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for OpenCV SURF processing.
        
        Converts to grayscale uint8 format as required by OpenCV.
        
        Args:
            image: Input image array
        
        Returns:
            uint8 grayscale image
        """
        # Handle color images
        if image.ndim == 3:
            if image.shape[2] == 3:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                # Take first channel
                image = image[..., 0]
        
        # Normalize to uint8 range
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _custom_to_cv_keypoints(
        self,
        keypoints: List[SURFKeypoint]
    ) -> List['cv2.KeyPoint']:
        """
        Convert custom SURFKeypoint objects to OpenCV KeyPoints.
        
        Args:
            keypoints: List of SURFKeypoint objects
        
        Returns:
            List of cv2.KeyPoint objects
        """
        import cv2
        
        cv_kps = []
        for kp in keypoints:
            cv_kp = cv2.KeyPoint(
                x=kp.x,
                y=kp.y,
                _size=kp.size,
                _angle=np.degrees(kp.orientation),
                _response=kp.response,
                _octave=kp.octave,
                _class_id=0
            )
            cv_kps.append(cv_kp)
        
        return cv_kps


class SURFInputError(ValueError):
    """Raised when SURF input validation fails."""
    pass


def _validate_inputs(
    img1: np.ndarray,
    img2: np.ndarray,
    allow_different_dtypes: bool = False
) -> None:
    """
    Validate that two numpy arrays are compatible for SURF comparison.
    
    Validates:
    - Both inputs are numpy arrays
    - Arrays have the same dimensions (ndim)
    - Arrays have the same shape
    - Arrays have the same size
    - Arrays have the same dtype (unless allow_different_dtypes=True)
    - Arrays are not empty
    - Arrays have valid dimensions (2D or 3D for images)
    
    Args:
        img1: First input array
        img2: Second input array
        allow_different_dtypes: If True, allow different dtypes (will be converted)
    
    Raises:
        SURFInputError: If validation fails with detailed error message
        TypeError: If inputs are not numpy arrays
    """
    # Type validation
    if not isinstance(img1, np.ndarray):
        raise TypeError(f"img1 must be a numpy array, got {type(img1).__name__}")
    if not isinstance(img2, np.ndarray):
        raise TypeError(f"img2 must be a numpy array, got {type(img2).__name__}")
    
    # Dimension validation
    if img1.ndim != img2.ndim:
        raise SURFInputError(
            f"Dimension mismatch: img1 has {img1.ndim}D, img2 has {img2.ndim}D. "
            f"Both arrays must have the same number of dimensions."
        )
    
    # Shape validation
    if img1.shape != img2.shape:
        raise SURFInputError(
            f"Shape mismatch: img1 shape {img1.shape} != img2 shape {img2.shape}. "
            f"Both arrays must have identical shapes."
        )
    
    # Size validation (redundant with shape but explicit)
    if img1.size != img2.size:
        raise SURFInputError(
            f"Size mismatch: img1 has {img1.size} elements, img2 has {img2.size} elements. "
            f"Both arrays must have the same number of elements."
        )
    
    # Empty array validation
    if img1.size == 0 or img2.size == 0:
        raise SURFInputError(
            f"Empty input arrays: cannot compute SURF on empty arrays "
            f"(img1 shape: {img1.shape}, img2 shape: {img2.shape})"
        )
    
    # Dtype validation
    if not allow_different_dtypes and img1.dtype != img2.dtype:
        raise SURFInputError(
            f"Dtype mismatch: img1 dtype '{img1.dtype}' != img2 dtype '{img2.dtype}'. "
            f"Both arrays should have the same data type for consistent comparison."
        )
    
    # Valid dimensions check (2D or 3D for images)
    if img1.ndim not in (2, 3):
        raise SURFInputError(
            f"Invalid dimensions: expected 2D or 3D arrays for image comparison, "
            f"got {img1.ndim}D array with shape {img1.shape}"
        )


class SURFConfig:
    """Configuration parameters for SURF algorithm."""
    
    def __init__(
        self,
        hessian_threshold: float = 100.0,
        n_octaves: int = 4,
        n_octave_layers: int = 3,
        extended: bool = True,
        upright: bool = False,
        descriptor_size: int = 64,
        match_ratio_threshold: float = 0.8,
    ):
        self.hessian_threshold = hessian_threshold
        self.n_octaves = n_octaves
        self.n_octave_layers = n_octave_layers
        self.extended = extended
        self.upright = upright
        self.descriptor_size = 128 if extended else 64
        self.match_ratio_threshold = match_ratio_threshold


@dataclass
class SURFKeypoint:
    """SURF keypoint with location, scale, and orientation."""
    x: float
    y: float
    octave: int
    scale_level: int
    size: float
    orientation: float = 0.0
    response: float = 0.0
    laplacian: int = 0
    descriptor: Optional[np.ndarray] = None
    
    @property
    def scale(self) -> float:
        """Return the effective scale of the keypoint."""
        return self.size


@dataclass
class SURFMatch:
    """Match between two SURF keypoints."""
    kp1: SURFKeypoint
    kp2: SURFKeypoint
    distance: float


class IntegralImage:
    """Integral image for fast Haar wavelet computation."""
    
    def __init__(self, image: np.ndarray):
        self.image = self._normalize_image(image)
        self.integral = self._compute_integral(self.image)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to float64 and single channel."""
        if image.ndim == 3:
            # Convert to grayscale using standard coefficients
            image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        
        return image.astype(np.float64)
    
    def _compute_integral(self, image: np.ndarray) -> np.ndarray:
        """Compute integral image (summed area table)."""
        return np.cumsum(np.cumsum(image, axis=0), axis=1)
    
    def get_sum(self, x: int, y: int, width: int, height: int) -> float:
        """
        Get sum of pixels in rectangular region using integral image.
        
        Args:
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Width of rectangle
            height: Height of rectangle
        
        Returns:
            Sum of pixels in the region
        """
        x2 = min(x + width - 1, self.integral.shape[1] - 1)
        y2 = min(y + height - 1, self.integral.shape[0] - 1)
        x = max(x, 0)
        y = max(y, 0)
        
        if x > x2 or y > y2:
            return 0.0
        
        total = self.integral[y2, x2]
        
        if x > 0:
            total -= self.integral[y2, x - 1]
        if y > 0:
            total -= self.integral[y - 1, x2]
        if x > 0 and y > 0:
            total += self.integral[y - 1, x - 1]
        
        return total
    
    def get_haar_x(self, x: int, y: int, size: int) -> float:
        """
        Compute Haar wavelet response in x direction.
        
        Uses two rectangular regions: left negative, right positive.
        
        Args:
            x: Center x coordinate
            y: Center y coordinate
            size: Size of the wavelet (must be divisible by 2)
        
        Returns:
            Haar x response
        """
        half = size // 2
        
        # Left region (negative)
        left = self.get_sum(x - size, y - half, half, size)
        
        # Right region (positive)
        right = self.get_sum(x, y - half, half, size)
        
        return right - left
    
    def get_haar_y(self, x: int, y: int, size: int) -> float:
        """
        Compute Haar wavelet response in y direction.
        
        Uses two rectangular regions: top negative, bottom positive.
        
        Args:
            x: Center x coordinate
            y: Center y coordinate
            size: Size of the wavelet (must be divisible by 2)
        
        Returns:
            Haar y response
        """
        half = size // 2
        
        # Top region (negative)
        top = self.get_sum(x - half, y - size, size, half)
        
        # Bottom region (positive)
        bottom = self.get_sum(x - half, y, size, half)
        
        return bottom - top


class FastHessian:
    """Fast Hessian detector using box filters and integral images."""
    
    def __init__(self, config: SURFConfig):
        self.config = config
    
    def detect(self, integral: IntegralImage) -> List[SURFKeypoint]:
        """
        Detect keypoints using Fast Hessian detector.
        
        Args:
            integral: Integral image of the input
        
        Returns:
            List of detected keypoints
        """
        keypoints = []
        
        # Build scale space and detect
        for octave in range(self.config.n_octaves):
            octave_keypoints = self._detect_octave(integral, octave)
            keypoints.extend(octave_keypoints)
        
        # Apply NMS and threshold
        keypoints = self._filter_keypoints(keypoints)
        
        return keypoints
    
    def _detect_octave(
        self,
        integral: IntegralImage,
        octave: int
    ) -> List[SURFKeypoint]:
        """Detect keypoints in a single octave."""
        keypoints = []
        
        # Scale factor for this octave
        scale_factor = 2 ** octave
        
        # Sample scales within octave
        for layer in range(self.config.n_octave_layers):
            # Base filter size increases with octave
            base_size = 9  # Smallest filter size
            filter_size = int(base_size * scale_factor * (1.2 ** layer))
            
            if filter_size > min(integral.image.shape) // 4:
                continue
            
            # Compute Hessian response map
            responses = self._compute_hessian_response(integral, filter_size, octave)
            
            # Find local maxima
            layer_kps = self._find_local_maxima(
                responses, filter_size, octave, layer
            )
            keypoints.extend(layer_kps)
        
        return keypoints
    
    def _compute_hessian_response(
        self,
        integral: IntegralImage,
        filter_size: int,
        octave: int
    ) -> np.ndarray:
        """
        Compute Hessian determinant response using box filters.
        
        Uses approximations of second derivatives with box filters:
        - Dxx: Horizontal second derivative
        - Dyy: Vertical second derivative
        - Dxy: Mixed derivative
        
        Args:
            integral: Integral image
            filter_size: Size of the filter
            octave: Current octave
        
        Returns:
            Hessian response map
        """
        h, w = integral.image.shape
        responses = np.zeros((h, w), dtype=np.float64)
        
        # Scale lobe size based on filter size
        lobe = filter_size // 3
        border = filter_size // 2 + 1
        
        # Weight for determinant normalization
        w_norm = 0.9  # Approximation factor for box filters
        
        for y in range(border, h - border):
            for x in range(border, w - border):
                # Dxx approximation using box filters
                dxx = self._compute_dxx(integral, x, y, lobe)
                
                # Dyy approximation using box filters
                dyy = self._compute_dyy(integral, x, y, lobe)
                
                # Dxy approximation using box filters
                dxy = self._compute_dxy(integral, x, y, lobe)
                
                # Determinant of Hessian
                det = dxx * dyy - w_norm * dxy * dxy
                
                responses[y, x] = det
        
        return responses
    
    def _compute_dxx(self, integral: IntegralImage, x: int, y: int, lobe: int) -> float:
        """Compute Dxx (second derivative in x) using box filters."""
        # Three horizontal boxes: positive, negative, positive
        size = 3 * lobe
        
        # Left positive
        left = integral.get_sum(x - size, y - lobe // 2, lobe, lobe)
        
        # Center negative (2x width)
        center = integral.get_sum(x - lobe, y - lobe // 2, lobe, lobe)
        
        # Right positive
        right = integral.get_sum(x + lobe, y - lobe // 2, lobe, lobe)
        
        return left - 2 * center + right
    
    def _compute_dyy(self, integral: IntegralImage, x: int, y: int, lobe: int) -> float:
        """Compute Dyy (second derivative in y) using box filters."""
        # Three vertical boxes: positive, negative, positive
        size = 3 * lobe
        
        # Top positive
        top = integral.get_sum(x - lobe // 2, y - size, lobe, lobe)
        
        # Center negative (2x height)
        center = integral.get_sum(x - lobe // 2, y - lobe, lobe, lobe)
        
        # Bottom positive
        bottom = integral.get_sum(x - lobe // 2, y + lobe, lobe, lobe)
        
        return top - 2 * center + bottom
    
    def _compute_dxy(self, integral: IntegralImage, x: int, y: int, lobe: int) -> float:
        """Compute Dxy (mixed derivative) using box filters."""
        # Diagonal pattern of four boxes
        half = lobe // 2
        
        # Top-left positive
        tl = integral.get_sum(x - lobe, y - lobe, half, half)
        
        # Top-right negative
        tr = integral.get_sum(x + half, y - lobe, half, half)
        
        # Bottom-left negative
        bl = integral.get_sum(x - lobe, y + half, half, half)
        
        # Bottom-right positive
        br = integral.get_sum(x + half, y + half, half, half)
        
        return tl - tr - bl + br
    
    def _find_local_maxima(
        self,
        responses: np.ndarray,
        filter_size: int,
        octave: int,
        layer: int
    ) -> List[SURFKeypoint]:
        """Find local maxima in 3x3x3 neighborhood across scales."""
        keypoints = []
        h, w = responses.shape
        
        border = filter_size // 2 + 2
        
        for y in range(border, h - border):
            for x in range(border, w - border):
                value = responses[y, x]
                
                # Check threshold
                if value < self.config.hessian_threshold:
                    continue
                
                # Check 3x3 neighborhood for local maximum
                if not self._is_local_maximum(responses, x, y, value):
                    continue
                
                # Determine Laplacian sign (trace of Hessian)
                laplacian = 1 if value > 0 else -1
                
                # Create keypoint
                kp = SURFKeypoint(
                    x=float(x),
                    y=float(y),
                    octave=octave,
                    scale_level=layer,
                    size=float(filter_size),
                    response=abs(value),
                    laplacian=laplacian
                )
                keypoints.append(kp)
        
        return keypoints
    
    def _is_local_maximum(
        self,
        responses: np.ndarray,
        x: int,
        y: int,
        value: float
    ) -> bool:
        """Check if point is local maximum in 3x3 neighborhood."""
        h, w = responses.shape
        
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if responses[ny, nx] >= value:
                        return False
        
        return True
    
    def _filter_keypoints(self, keypoints: List[SURFKeypoint]) -> List[SURFKeypoint]:
        """Filter keypoints by threshold and apply NMS."""
        # Sort by response
        keypoints.sort(key=lambda kp: kp.response, reverse=True)
        
        # Simple NMS: keep only strongest in local neighborhood
        filtered = []
        min_distance = 10  # Minimum distance between keypoints
        
        for kp in keypoints:
            # Check if too close to existing keypoint
            too_close = False
            for existing in filtered:
                dist = np.sqrt((kp.x - existing.x) ** 2 + (kp.y - existing.y) ** 2)
                if dist < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered.append(kp)
        
        return filtered


class SURFOrientation:
    """Assign orientation to SURF keypoints using Haar wavelets."""
    
    def __init__(self, config: SURFConfig):
        self.config = config
    
    def assign(
        self,
        keypoints: List[SURFKeypoint],
        integral: IntegralImage
    ) -> List[SURFKeypoint]:
        """Assign orientation to all keypoints."""
        if self.config.upright:
            # Upright SURF - no rotation invariance
            for kp in keypoints:
                kp.orientation = 0.0
            return keypoints
        
        oriented_keypoints = []
        
        for kp in keypoints:
            orientation = self._compute_orientation(kp, integral)
            kp.orientation = orientation
            oriented_keypoints.append(kp)
        
        return oriented_keypoints
    
    def _compute_orientation(
        self,
        kp: SURFKeypoint,
        integral: IntegralImage
    ) -> float:
        """
        Compute dominant orientation using Haar wavelets.
        
        Computes Haar wavelet responses in a circular neighborhood
        and finds the dominant orientation using a sliding window.
        """
        # Sampling radius
        radius = int(round(2 * kp.size))
        radius = max(radius, 6)
        
        # Sample step
        step = radius // 2
        
        # Collect responses
        responses_x = []
        responses_y = []
        angles = []
        
        h, w = integral.image.shape
        
        for dy in range(-radius, radius + 1, step):
            for dx in range(-radius, radius + 1, step):
                x = int(kp.x) + dx
                y = int(kp.y) + dy
                
                # Check bounds and circular region
                if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
                    continue
                
                if dx * dx + dy * dy > radius * radius:
                    continue
                
                # Haar wavelet responses
                haar_size = max(2, int(kp.size))
                
                haar_x = integral.get_haar_x(x, y, haar_size)
                haar_y = integral.get_haar_y(x, y, haar_size)
                
                # Gaussian weight based on distance from center
                dist_sq = dx * dx + dy * dy
                weight = np.exp(-dist_sq / (2 * (radius / 3) ** 2))
                
                responses_x.append(weight * haar_x)
                responses_y.append(weight * haar_y)
                angles.append(np.arctan2(haar_y, haar_x))
        
        if not responses_x:
            return 0.0
        
        # Compute dominant orientation using histogram
        n_bins = 36
        histogram = np.zeros(n_bins)
        
        for hx, hy in zip(responses_x, responses_y):
            magnitude = np.sqrt(hx * hx + hy * hy)
            angle = np.arctan2(hy, hx)
            
            # Map to bin
            bin_idx = int((angle + np.pi) / (2 * np.pi) * n_bins) % n_bins
            histogram[bin_idx] += magnitude
        
        # Smooth histogram
        for _ in range(2):
            temp = histogram.copy()
            for i in range(n_bins):
                histogram[i] = (
                    temp[(i - 1) % n_bins] +
                    temp[i] +
                    temp[(i + 1) % n_bins]
                ) / 3.0
        
        # Find maximum
        max_bin = np.argmax(histogram)
        max_val = histogram[max_bin]
        
        # Parabolic interpolation for sub-bin accuracy
        left = histogram[(max_bin - 1) % n_bins]
        right = histogram[(max_bin + 1) % n_bins]
        
        offset = 0.5 * (left - right) / (left - 2 * max_val + right + 1e-10)
        
        orientation = 2 * np.pi * (max_bin + offset) / n_bins - np.pi
        
        return orientation


class SURFDescriptor:
    """Compute SURF descriptors using Haar wavelets."""
    
    def __init__(self, config: SURFConfig):
        self.config = config
    
    def compute(
        self,
        keypoints: List[SURFKeypoint],
        integral: IntegralImage
    ) -> List[SURFKeypoint]:
        """Compute descriptors for all keypoints."""
        for kp in keypoints:
            kp.descriptor = self._compute_descriptor(kp, integral)
        
        return keypoints
    
    def _compute_descriptor(
        self,
        kp: SURFKeypoint,
        integral: IntegralImage
    ) -> Optional[np.ndarray]:
        """
        Compute SURF descriptor for a keypoint.
        
        The descriptor consists of Haar wavelet responses computed
        over a 4x4 grid of sub-regions around the keypoint.
        """
        h, w = integral.image.shape
        
        # Descriptor window size
        window_size = 20 * kp.size / 9  # Scale with keypoint size
        
        # Number of sub-regions
        n_regions = 4
        region_size = window_size / n_regions
        
        # Sample step
        sample_step = region_size / 2
        
        # Descriptor vector
        desc_size = self.config.descriptor_size
        descriptor = np.zeros(desc_size)
        
        # Precompute rotation
        cos_theta = np.cos(kp.orientation)
        sin_theta = np.sin(kp.orientation)
        
        idx = 0
        
        for i in range(n_regions):
            for j in range(n_regions):
                # Sub-region center
                cx = (j - 1.5) * region_size
                cy = (i - 1.5) * region_size
                
                # Haar responses for this sub-region
                sum_dx = 0.0
                sum_dy = 0.0
                sum_abs_dx = 0.0
                sum_abs_dy = 0.0
                count = 0
                
                # Sample points in sub-region
                for sy in range(5):
                    for sx in range(5):
                        # Sample position relative to sub-region center
                        sx_rel = cx + (sx - 2) * sample_step
                        sy_rel = cy + (sy - 2) * sample_step
                        
                        # Rotate by keypoint orientation
                        x_rot = sx_rel * cos_theta - sy_rel * sin_theta
                        y_rot = sx_rel * sin_theta + sy_rel * cos_theta
                        
                        # Absolute position
                        x = int(kp.x + x_rot)
                        y = int(kp.y + y_rot)
                        
                        # Check bounds
                        if x < 1 or x >= w - 1 or y < 1 or y >= h - 1:
                            continue
                        
                        # Haar wavelet size
                        haar_size = max(2, int(kp.size / 2))
                        
                        # Compute Haar responses
                        dx = integral.get_haar_x(x, y, haar_size)
                        dy = integral.get_haar_y(x, y, haar_size)
                        
                        # Rotate responses to keypoint orientation
                        dx_rot = dx * cos_theta + dy * sin_theta
                        dy_rot = -dx * sin_theta + dy * cos_theta
                        
                        # Gaussian weight based on distance from center
                        dist_sq = sx_rel * sx_rel + sy_rel * sy_rel
                        weight = np.exp(-dist_sq / (2 * (window_size / 2) ** 2))
                        
                        sum_dx += weight * dx_rot
                        sum_dy += weight * dy_rot
                        sum_abs_dx += weight * abs(dx_rot)
                        sum_abs_dy += weight * abs(dy_rot)
                        count += 1
                
                if count > 0:
                    # Normalize by count
                    descriptor[idx] = sum_dx / count
                    descriptor[idx + 1] = sum_dy / count
                    descriptor[idx + 2] = sum_abs_dx / count
                    descriptor[idx + 3] = sum_abs_dy / count
                    idx += 4
        
        # Normalize descriptor
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        # Clamp values for illumination invariance
        descriptor = np.clip(descriptor, 0, 0.2)
        
        # Renormalize
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        # Convert to uint8
        descriptor = np.clip(descriptor * 512, 0, 255).astype(np.uint8)
        
        return descriptor


class SURF:
    """Complete SURF implementation for image feature detection."""
    
    def __init__(self, config: Optional[SURFConfig] = None):
        self.config = config or SURFConfig()
        self.hessian_detector = FastHessian(self.config)
        self.orientation_assigner = SURFOrientation(self.config)
        self.descriptor_computer = SURFDescriptor(self.config)
    
    def detect_and_compute(self, image: np.ndarray) -> List[SURFKeypoint]:
        """Detect keypoints and compute descriptors for an image."""
        # Create integral image
        integral = IntegralImage(image)
        
        # Detect keypoints using Fast Hessian
        keypoints = self.hessian_detector.detect(integral)
        
        # Assign orientations
        keypoints = self.orientation_assigner.assign(keypoints, integral)
        
        # Compute descriptors
        keypoints = self.descriptor_computer.compute(keypoints, integral)
        
        return keypoints
    
    def detect(self, image: np.ndarray) -> List[SURFKeypoint]:
        """Detect keypoints only (no descriptors)."""
        # Create integral image
        integral = IntegralImage(image)
        
        # Detect keypoints using Fast Hessian
        keypoints = self.hessian_detector.detect(integral)
        
        # Assign orientations
        keypoints = self.orientation_assigner.assign(keypoints, integral)
        
        return keypoints


class SURFMatcher:
    """Match SURF features between images."""
    
    def __init__(self, config: Optional[SURFConfig] = None):
        self.config = config or SURFConfig()
    
    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[List[SURFKeypoint], List[SURFKeypoint], List[SURFMatch]]:
        """Find matching features between two images."""
        # Validate inputs
        _validate_inputs(image1, image2, allow_different_dtypes=True)
        
        # Detect features in both images using separate SURF instances
        surf1 = SURF(self.config)
        surf2 = SURF(self.config)
        keypoints1 = surf1.detect_and_compute(image1)
        keypoints2 = surf2.detect_and_compute(image2)
        
        # Match features
        matches = self._match_keypoints(keypoints1, keypoints2)
        
        return keypoints1, keypoints2, matches
    
    def _match_keypoints(
        self,
        keypoints1: List[SURFKeypoint],
        keypoints2: List[SURFKeypoint]
    ) -> List[SURFMatch]:
        """Match keypoints using nearest neighbor search."""
        matches = []
        
        # Filter keypoints with descriptors
        kp1_with_desc = [kp for kp in keypoints1 if kp.descriptor is not None]
        kp2_with_desc = [kp for kp in keypoints2 if kp.descriptor is not None]
        
        if not kp1_with_desc or not kp2_with_desc:
            return matches
        
        # Build descriptor arrays
        desc1 = np.array([kp.descriptor for kp in kp1_with_desc], dtype=np.float32)
        desc2 = np.array([kp.descriptor for kp in kp2_with_desc], dtype=np.float32)
        
        # Normalize descriptors for comparison
        desc1 = desc1 / 255.0
        desc2 = desc2 / 255.0
        
        # Find nearest neighbors for each keypoint in image1
        for i, kp1 in enumerate(kp1_with_desc):
            distances = np.linalg.norm(desc2 - desc1[i], axis=1)
            
            # Find two nearest neighbors
            sorted_indices = np.argsort(distances)
            
            if len(sorted_indices) < 2:
                continue
            
            best_idx = sorted_indices[0]
            second_best_idx = sorted_indices[1]
            
            best_dist = distances[best_idx]
            second_best_dist = distances[second_best_idx]
            
            # Ratio test (Lowe's criterion)
            if best_dist < 0.001:  # Perfect or near-perfect match
                matches.append(SURFMatch(kp1, kp2_with_desc[best_idx], float(best_dist)))
            elif second_best_dist > 0 and best_dist / second_best_dist < self.config.match_ratio_threshold:
                matches.append(SURFMatch(kp1, kp2_with_desc[best_idx], float(best_dist)))
        
        return matches


class SURFDiff:
    """SURF-based image difference detection."""
    
    def __init__(self, config: Optional[SURFConfig] = None):
        self.config = config or SURFConfig()
        self.matcher = SURFMatcher(self.config)
    
    def compare(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> dict:
        """
        Compare two images using SURF features.
        
        Args:
            image1: First image as numpy array (H, W) or (H, W, C)
            image2: Second image as numpy array (H, W) or (H, W, C)
        
        Returns:
            Dictionary with comparison results
        
        Raises:
            SURFInputError: If input arrays have incompatible dimensions, size, or dtype
            TypeError: If inputs are not numpy arrays
        """
        # Validate inputs
        _validate_inputs(image1, image2, allow_different_dtypes=True)
        
        # Match features
        keypoints1, keypoints2, matches = self.matcher.match(image1, image2)
        
        # Calculate metrics
        n_kp1 = len([kp for kp in keypoints1 if kp.descriptor is not None])
        n_kp2 = len([kp for kp in keypoints2 if kp.descriptor is not None])
        n_matches = len(matches)
        
        if n_kp1 > 0:
            match_ratio = n_matches / n_kp1
        else:
            match_ratio = 0.0
        
        if matches:
            avg_distance = np.mean([m.distance for m in matches])
        else:
            avg_distance = float('inf')
        
        # Find matched and unmatched regions
        matched1 = set()
        matched2 = set()
        
        for match in matches:
            matched1.add((int(match.kp1.x), int(match.kp1.y)))
            matched2.add((int(match.kp2.x), int(match.kp2.y)))
        
        unmatched1 = []
        unmatched2 = []
        
        for kp in keypoints1:
            if (int(kp.x), int(kp.y)) not in matched1:
                unmatched1.append((int(kp.x), int(kp.y), kp.scale))
        
        for kp in keypoints2:
            if (int(kp.x), int(kp.y)) not in matched2:
                unmatched2.append((int(kp.x), int(kp.y), kp.scale))
        
        return {
            'match_ratio': match_ratio,
            'avg_distance': avg_distance,
            'n_keypoints1': n_kp1,
            'n_keypoints2': n_kp2,
            'n_matches': n_matches,
            'matched_regions': list(matched1),
            'unmatched_regions1': unmatched1,
            'unmatched_regions2': unmatched2,
            'keypoints1': keypoints1,
            'keypoints2': keypoints2,
            'matches': matches
        }
    
    def detect_changes(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        match_threshold: float = 0.5
    ) -> Tuple[bool, dict]:
        """
        Detect if significant changes exist between two images.
        
        Args:
            image1: First image
            image2: Second image
            match_threshold: Minimum match ratio to consider images similar
        
        Returns:
            Tuple of (has_changes, comparison_results)
        """
        results = self.compare(image1, image2)
        
        has_changes = results['match_ratio'] < match_threshold
        
        return has_changes, results


def surf_diff(
    image1: np.ndarray,
    image2: np.ndarray,
    **kwargs
) -> dict:
    """
    Convenience function for SURF-based image difference detection.
    
    Args:
        image1: First image (H x W or H x W x C)
        image2: Second image (H x W or H x W x C)
        **kwargs: Additional configuration parameters
    
    Returns:
        Dictionary with comparison results
    """
    config = SURFConfig(**kwargs)
    diff = SURFDiff(config)
    return diff.compare(image1, image2)


def compute_surf_diff(
    image1: np.ndarray,
    image2: np.ndarray,
    **kwargs
) -> dict:
    """
    Compute SURF-based difference metrics between two images.
    
    This is an alias for surf_diff() that provides a more explicit function name
    for computing difference metrics using SURF feature matching.
    
    Args:
        image1: First image as numpy array (H, W) or (H, W, C)
        image2: Second image as numpy array (H, W) or (H, W, C)
        **kwargs: Additional configuration parameters (e.g., hessian_threshold,
                  n_octaves, match_ratio_threshold)
    
    Returns:
        Dictionary with difference metrics including:
        - match_ratio: Ratio of matched features to total features
        - avg_distance: Average descriptor distance for matches
        - n_keypoints1: Number of keypoints in image1
        - n_keypoints2: Number of keypoints in image2
        - n_matches: Number of matched features
        - matched_regions: List of matched region coordinates
        - unmatched_regions1: List of unmatched regions in image1
        - unmatched_regions2: List of unmatched regions in image2
        - keypoints1: List of keypoint objects from image1
        - keypoints2: List of keypoint objects from image2
        - matches: List of match objects
    
    Raises:
        SURFInputError: If input arrays have incompatible dimensions, size, or dtype
        TypeError: If inputs are not numpy arrays
    
    Example:
        >>> import numpy as np
        >>> img1 = np.random.rand(256, 256)
        >>> img2 = np.random.rand(256, 256)
        >>> result = compute_surf_diff(img1, img2)
        >>> print(f"Match ratio: {result['match_ratio']:.3f}")
    """
    return surf_diff(image1, image2, **kwargs)


def create_surf(
    backend: str = 'auto',
    config: Optional[SURFConfig] = None,
    fallback: bool = True,
    **kwargs
) -> SURF:
    """
    Factory function to create SURF instance with specified backend.
    
    Automatically selects the best available implementation based on the
    backend parameter and availability of OpenCV SURF.
    
    Args:
        backend: Backend to use ('auto', 'opencv', or 'numpy')
        config: Optional SURF configuration
        fallback: If True, fallback to pure Python when OpenCV unavailable
        **kwargs: Additional configuration parameters (merged with config)
    
    Returns:
        SURF instance (OpenCVSURF if available, otherwise pure Python SURF)
    
    Raises:
        ImportError: If 'opencv' backend is specified but not available and fallback=False
        ValueError: If invalid backend specified
    
    Example:
        >>> from diff.surf import create_surf
        >>> 
        >>> # Auto-detect best available
        >>> surf = create_surf()
        >>> 
        >>> # Force OpenCV with fallback
        >>> surf = create_surf('opencv', fallback=True)
        >>> 
        >>> # Use pure Python implementation
        >>> surf = create_surf('numpy')
        >>> 
        >>> # Custom parameters
        >>> surf = create_surf(hessian_threshold=400, n_octaves=3)
    """
    if config is None:
        config = SURFConfig(**kwargs)
    else:
        # Update config with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    if backend == 'auto':
        if has_opencv_surf():
            return OpenCVSURF(config=config, fallback=fallback)
        return SURF(config)
    
    elif backend == 'opencv':
        return OpenCVSURF(config=config, fallback=fallback)
    
    elif backend == 'numpy':
        return SURF(config)
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'opencv', or 'numpy'")
