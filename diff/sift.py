"""
SIFT (Scale-Invariant Feature Transform) implementation for image difference detection.

This module provides SIFT-based image comparison capabilities for detecting
structural differences between images, with scale and rotation invariance.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import IntEnum
import math
import warnings

__all__ = [
    'SIFTInputError',
    'SIFTConfig',
    'Keypoint',
    'Match',
    'SIFT',
    'SIFTMatcher',
    'SIFTDiff',
    'sift_diff',
    'compute_sift_diff',
    'detectAndCompute',
    'OpenCVSIFT',
    'create_sift',
]


def _has_opencv_sift() -> bool:
    """Check if OpenCV SIFT is available (requires opencv-contrib-python)."""
    try:
        import cv2
        # Check if SIFT_create is available (in contrib module)
        return hasattr(cv2, 'SIFT_create')
    except ImportError:
        return False


def _cv_keypoint_to_custom(cv_kp: 'cv2.KeyPoint') -> Keypoint:
    """Convert OpenCV KeyPoint to custom KeyPoint."""
    return Keypoint(
        x=cv_kp.pt[0],
        y=cv_kp.pt[1],
        octave=cv_kp.octave,
        scale_level=0,  # OpenCV doesn't expose this directly
        sigma=cv_kp.size / 2,
        orientation=np.radians(cv_kp.angle),
        descriptor=None
    )


class SIFTInputError(ValueError):
    """Raised when SIFT input validation fails."""
    pass


def _validate_inputs(
    img1: np.ndarray,
    img2: np.ndarray,
    allow_different_dtypes: bool = False
) -> None:
    """
    Validate that two numpy arrays are compatible for SIFT comparison.
    
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
        SIFTInputError: If validation fails with detailed error message
        TypeError: If inputs are not numpy arrays
    """
    # Type validation
    if not isinstance(img1, np.ndarray):
        raise TypeError(f"img1 must be a numpy array, got {type(img1).__name__}")
    if not isinstance(img2, np.ndarray):
        raise TypeError(f"img2 must be a numpy array, got {type(img2).__name__}")
    
    # Dimension validation
    if img1.ndim != img2.ndim:
        raise SIFTInputError(
            f"Dimension mismatch: img1 has {img1.ndim}D, img2 has {img2.ndim}D. "
            f"Both arrays must have the same number of dimensions."
        )
    
    # Shape validation
    if img1.shape != img2.shape:
        raise SIFTInputError(
            f"Shape mismatch: img1 shape {img1.shape} != img2 shape {img2.shape}. "
            f"Both arrays must have identical shapes."
        )
    
    # Size validation (redundant with shape but explicit)
    if img1.size != img2.size:
        raise SIFTInputError(
            f"Size mismatch: img1 has {img1.size} elements, img2 has {img2.size} elements. "
            f"Both arrays must have the same number of elements."
        )
    
    # Empty array validation
    if img1.size == 0 or img2.size == 0:
        raise SIFTInputError(
            f"Empty input arrays: cannot compute SIFT on empty arrays "
            f"(img1 shape: {img1.shape}, img2 shape: {img2.shape})"
        )
    
    # Dtype validation
    if not allow_different_dtypes and img1.dtype != img2.dtype:
        raise SIFTInputError(
            f"Dtype mismatch: img1 dtype '{img1.dtype}' != img2 dtype '{img2.dtype}'. "
            f"Both arrays should have the same data type for consistent comparison."
        )
    
    # Valid dimensions check (2D or 3D for images)
    if img1.ndim not in (2, 3):
        raise SIFTInputError(
            f"Invalid dimensions: expected 2D or 3D arrays for image comparison, "
            f"got {img1.ndim}D array with shape {img1.shape}"
        )


class SIFTConfig:
    """Configuration parameters for SIFT algorithm."""
    
    def __init__(
        self,
        n_octaves: int = 4,
        n_scales_per_octave: int = 3,
        sigma_init: float = 1.6,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        descriptor_radius_factor: float = 3.0,
        n_bins_orientation: int = 36,
        n_bins_descriptor: int = 8,
        descriptor_window_size: int = 4,
        match_ratio_threshold: float = 0.8,
    ):
        self.n_octaves = n_octaves
        self.n_scales_per_octave = n_scales_per_octave
        self.sigma_init = sigma_init
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.descriptor_radius_factor = descriptor_radius_factor
        self.n_bins_orientation = n_bins_orientation
        self.n_bins_descriptor = n_bins_descriptor
        self.descriptor_window_size = descriptor_window_size
        self.match_ratio_threshold = match_ratio_threshold


@dataclass
class Keypoint:
    """SIFT keypoint with location, scale, and orientation."""
    x: float
    y: float
    octave: int
    scale_level: int
    sigma: float
    orientation: float = 0.0
    descriptor: Optional[np.ndarray] = None
    
    @property
    def scale(self) -> float:
        """Return the effective scale of the keypoint."""
        return self.sigma * (2 ** self.octave)


@dataclass
class Match:
    """Match between two keypoints."""
    kp1: Keypoint
    kp2: Keypoint
    distance: float
    
    
class GaussianPyramid:
    """Gaussian pyramid for multi-scale analysis."""
    
    def __init__(self, config: SIFTConfig):
        self.config = config
        self.octaves: List[List[np.ndarray]] = []
        self.sigmas: List[List[float]] = []
    
    def build(self, image: np.ndarray) -> None:
        """Build Gaussian pyramid from input image."""
        image = self._normalize_image(image)
        
        # Calculate number of Gaussian images per octave (n_scales + 3 for DoG)
        n_images = self.config.n_scales_per_octave + 3
        k = 2 ** (1.0 / self.config.n_scales_per_octave)
        
        for octave in range(self.config.n_octaves):
            octave_images = []
            octave_sigmas = []
            
            if octave == 0:
                # First octave - start with original or upsampled image
                base_image = image
            else:
                # Downsample from previous octave
                prev_octave = self.octaves[octave - 1]
                base_image = self._downsample(prev_octave[self.config.n_scales_per_octave])
            
            # Generate Gaussian images for this octave
            for scale in range(n_images):
                if scale == 0:
                    # First image in octave
                    sigma = self.config.sigma_init
                    blurred = self._gaussian_blur(base_image, sigma)
                else:
                    # Subsequent images
                    sigma_prev = octave_sigmas[-1]
                    sigma = sigma_prev * k
                    # Incremental blur
                    sigma_inc = np.sqrt(sigma ** 2 - sigma_prev ** 2)
                    blurred = self._gaussian_blur(octave_images[-1], sigma_inc)
                
                octave_images.append(blurred)
                octave_sigmas.append(sigma)
            
            self.octaves.append(octave_images)
            self.sigmas.append(octave_sigmas)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to float64 and single channel."""
        if image.ndim == 3:
            # Convert to grayscale using standard coefficients
            image = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        
        image = image.astype(np.float64)
        
        # Pre-blur if image is larger than double size
        if image.shape[0] > 2 * 512 and image.shape[1] > 2 * 512:
            image = self._downsample(image)
        
        return image
    
    def _gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian blur to image."""
        if sigma < 0.1:
            return image.copy()
        
        kernel_size = int(6 * sigma) | 1  # Ensure odd
        kernel_size = max(kernel_size, 3)
        
        kernel = self._gaussian_kernel(kernel_size, sigma)
        
        # Separable convolution for efficiency
        temp = self._convolve1d(image, kernel, axis=0)
        result = self._convolve1d(temp, kernel, axis=1)
        
        return result
    
    def _gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Generate 1D Gaussian kernel."""
        x = np.arange(size) - size // 2
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        return kernel / kernel.sum()
    
    def _convolve1d(self, image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
        """1D convolution along specified axis."""
        from scipy.ndimage import convolve1d
        return convolve1d(image, kernel, axis=axis, mode='reflect')
    
    def _downsample(self, image: np.ndarray) -> np.ndarray:
        """Downsample image by factor of 2."""
        return image[::2, ::2]


class DoGPyramid:
    """Difference of Gaussians pyramid for keypoint detection."""
    
    def __init__(self, gaussian_pyramid: GaussianPyramid):
        self.gaussian_pyramid = gaussian_pyramid
        self.octaves: List[List[np.ndarray]] = []
    
    def build(self) -> None:
        """Build DoG pyramid from Gaussian pyramid."""
        self.octaves = []
        
        for gaussian_octave in self.gaussian_pyramid.octaves:
            dog_octave = []
            for i in range(len(gaussian_octave) - 1):
                dog = gaussian_octave[i + 1] - gaussian_octave[i]
                dog_octave.append(dog)
            self.octaves.append(dog_octave)


class KeypointDetector:
    """Detect and refine SIFT keypoints."""
    
    def __init__(self, config: SIFTConfig):
        self.config = config
    
    def detect(
        self,
        dog_pyramid: DoGPyramid,
        sigmas: List[List[float]]
    ) -> List[Keypoint]:
        """Detect keypoints from DoG pyramid."""
        keypoints = []
        
        for octave_idx, dog_octave in enumerate(dog_pyramid.octaves):
            # Search scales excluding first and last
            for scale_idx in range(1, len(dog_octave) - 1):
                keypoints.extend(
                    self._detect_at_scale(
                        octave_idx, scale_idx, dog_octave, sigmas[octave_idx]
                    )
                )
        
        return keypoints
    
    def _detect_at_scale(
        self,
        octave: int,
        scale: int,
        dog_octave: List[np.ndarray],
        octave_sigmas: List[float]
    ) -> List[Keypoint]:
        """Detect keypoints at specific scale level."""
        keypoints = []
        
        prev_img = dog_octave[scale - 1]
        curr_img = dog_octave[scale]
        next_img = dog_octave[scale + 1]
        
        # Find local extrema
        h, w = curr_img.shape
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                value = curr_img[y, x]
                
                # Check if extremum in 3x3x3 neighborhood
                if not self._is_extremum(value, x, y, prev_img, curr_img, next_img):
                    continue
                
                # Refine keypoint location
                kp = self._refine_keypoint(
                    x, y, octave, scale, prev_img, curr_img, next_img, octave_sigmas
                )
                
                if kp is not None:
                    keypoints.append(kp)
        
        return keypoints
    
    def _is_extremum(
        self,
        value: float,
        x: int, y: int,
        prev: np.ndarray,
        curr: np.ndarray,
        next_img: np.ndarray
    ) -> bool:
        """Check if point is local extremum in 3x3x3 neighborhood."""
        # Collect all 26 neighbors
        neighbors = []
        
        # Previous scale (3x3)
        neighbors.extend(prev[y-1:y+2, x-1:x+2].flatten())
        
        # Current scale (8 neighbors, excluding center)
        neighbors.extend([
            curr[y-1, x-1], curr[y-1, x], curr[y-1, x+1],
            curr[y, x-1],                   curr[y, x+1],
            curr[y+1, x-1], curr[y+1, x], curr[y+1, x+1]
        ])
        
        # Next scale (3x3)
        neighbors.extend(next_img[y-1:y+2, x-1:x+2].flatten())
        
        # Check if value is strictly greater or strictly less than all neighbors
        is_max = all(value > n for n in neighbors)
        is_min = all(value < n for n in neighbors)
        
        return is_max or is_min
    
    def _refine_keypoint(
        self,
        x: int, y: int,
        octave: int, scale: int,
        prev: np.ndarray, curr: np.ndarray, next_img: np.ndarray,
        octave_sigmas: List[float]
    ) -> Optional[Keypoint]:
        """Refine keypoint location using interpolation."""
        # Perform up to 5 iterations of refinement
        for _ in range(5):
            # Compute gradient and Hessian
            dx = (curr[y, x+1] - curr[y, x-1]) / 2.0
            dy = (curr[y+1, x] - curr[y-1, x]) / 2.0
            ds = (next_img[y, x] - prev[y, x]) / 2.0
            
            dxx = curr[y, x+1] + curr[y, x-1] - 2 * curr[y, x]
            dyy = curr[y+1, x] + curr[y-1, x] - 2 * curr[y, x]
            dss = next_img[y, x] + prev[y, x] - 2 * curr[y, x]
            
            dxy = (curr[y+1, x+1] - curr[y+1, x-1] - curr[y-1, x+1] + curr[y-1, x-1]) / 4.0
            dxs = (next_img[y, x+1] - next_img[y, x-1] - prev[y, x+1] + prev[y, x-1]) / 4.0
            dys = (next_img[y+1, x] - next_img[y-1, x] - prev[y+1, x] + prev[y-1, x]) / 4.0
            
            gradient = np.array([dx, dy, ds])
            hessian = np.array([
                [dxx, dxy, dxs],
                [dxy, dyy, dys],
                [dxs, dys, dss]
            ])
            
            # Solve for offset
            try:
                offset = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                return None
            
            # Check convergence
            if np.all(np.abs(offset) < 0.5):
                break
            
            # Update position
            x += int(round(offset[0]))
            y += int(round(offset[1]))
            scale_shift = offset[2]
            
            # Check bounds
            if (x < 1 or x >= curr.shape[1] - 1 or 
                y < 1 or y >= curr.shape[0] - 1 or
                scale + int(round(scale_shift)) < 1 or
                scale + int(round(scale_shift)) >= len(octave_sigmas) - 1):
                return None
        else:
            # Did not converge
            return None
        
        # Compute interpolated value
        interpolated_value = curr[y, x] + 0.5 * np.dot(gradient, offset)
        
        # Filter by contrast
        if abs(interpolated_value) < self.config.contrast_threshold:
            return None
        
        # Filter by edge response
        if not self._check_edge_response(curr, x, y):
            return None
        
        # Create keypoint
        sigma = octave_sigmas[scale] * (2 ** octave)
        
        return Keypoint(
            x=float(x) + offset[0],
            y=float(y) + offset[1],
            octave=octave,
            scale_level=scale,
            sigma=sigma
        )
    
    def _check_edge_response(self, image: np.ndarray, x: int, y: int) -> bool:
        """Filter out edge responses using principal curvatures ratio."""
        # Compute Hessian at keypoint
        dxx = image[y, x+1] + image[y, x-1] - 2 * image[y, x]
        dyy = image[y+1, x] + image[y-1, x] - 2 * image[y, x]
        dxy = (image[y+1, x+1] - image[y+1, x-1] - image[y-1, x+1] + image[y-1, x-1]) / 4.0
        
        # Compute trace and determinant
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy
        
        if det <= 0:
            return False
        
        # Check edge threshold
        r = self.config.edge_threshold
        if trace * trace / det > (r + 1) ** 2 / r:
            return False
        
        return True


class OrientationAssigner:
    """Assign orientations to keypoints."""
    
    def __init__(self, config: SIFTConfig):
        self.config = config
    
    def assign(
        self,
        keypoints: List[Keypoint],
        gaussian_pyramid: GaussianPyramid
    ) -> List[Keypoint]:
        """Assign orientations to all keypoints."""
        oriented_keypoints = []
        
        for kp in keypoints:
            oct_kps = self._assign_to_keypoint(kp, gaussian_pyramid)
            oriented_keypoints.extend(oct_kps)
        
        return oriented_keypoints
    
    def _assign_to_keypoint(
        self,
        kp: Keypoint,
        gaussian_pyramid: GaussianPyramid
    ) -> List[Keypoint]:
        """Assign orientation(s) to a single keypoint."""
        octave_images = gaussian_pyramid.octaves[kp.octave]
        image = octave_images[kp.scale_level]
        
        # Scale coordinates to current octave
        x = int(round(kp.x))
        y = int(round(kp.y))
        
        # Gaussian-weighted histogram of gradients
        sigma = self.config.descriptor_radius_factor * kp.sigma / (2 ** kp.octave)
        radius = int(round(3 * sigma))
        
        histogram = np.zeros(self.config.n_bins_orientation)
        
        h, w = image.shape
        
        for dy in range(-radius, radius + 1):
            yy = y + dy
            if yy < 1 or yy >= h - 1:
                continue
            
            for dx in range(-radius, radius + 1):
                xx = x + dx
                if xx < 1 or xx >= w - 1:
                    continue
                
                # Compute gradient
                gx = image[yy, xx+1] - image[yy, xx-1]
                gy = image[yy+1, xx] - image[yy-1, xx]
                
                magnitude = np.sqrt(gx*gx + gy*gy)
                orientation = np.arctan2(gy, gx)
                
                # Gaussian weight
                weight = np.exp(-(dx*dx + dy*dy) / (2 * sigma * sigma))
                
                # Add to histogram
                bin_idx = int(round(self.config.n_bins_orientation * 
                                   (orientation + np.pi) / (2 * np.pi)))
                bin_idx = bin_idx % self.config.n_bins_orientation
                
                histogram[bin_idx] += weight * magnitude
        
        # Smooth histogram
        for _ in range(2):
            temp = histogram.copy()
            for i in range(self.config.n_bins_orientation):
                histogram[i] = (temp[(i-1) % self.config.n_bins_orientation] + 
                               temp[i] + 
                               temp[(i+1) % self.config.n_bins_orientation]) / 3.0
        
        # Find peaks
        max_val = histogram.max()
        threshold = 0.8 * max_val
        
        keypoints = []
        
        for i in range(self.config.n_bins_orientation):
            if histogram[i] > histogram[(i-1) % self.config.n_bins_orientation] and \
               histogram[i] > histogram[(i+1) % self.config.n_bins_orientation] and \
               histogram[i] >= threshold:
                
                # Parabolic interpolation
                left = histogram[(i-1) % self.config.n_bins_orientation]
                right = histogram[(i+1) % self.config.n_bins_orientation]
                peak_offset = 0.5 * (left - right) / (left - 2*histogram[i] + right)
                
                orientation = 2 * np.pi * (i + peak_offset) / self.config.n_bins_orientation - np.pi
                
                new_kp = Keypoint(
                    x=kp.x,
                    y=kp.y,
                    octave=kp.octave,
                    scale_level=kp.scale_level,
                    sigma=kp.sigma,
                    orientation=orientation
                )
                keypoints.append(new_kp)
        
        return keypoints if keypoints else [kp]


class DescriptorComputer:
    """Compute SIFT descriptors for keypoints."""
    
    def __init__(self, config: SIFTConfig):
        self.config = config
    
    def compute(
        self,
        keypoints: List[Keypoint],
        gaussian_pyramid: GaussianPyramid
    ) -> List[Keypoint]:
        """Compute descriptors for all keypoints."""
        for kp in keypoints:
            kp.descriptor = self._compute_descriptor(kp, gaussian_pyramid)
        
        return keypoints
    
    def _compute_descriptor(
        self,
        kp: Keypoint,
        gaussian_pyramid: GaussianPyramid
    ) -> Optional[np.ndarray]:
        """Compute descriptor for a single keypoint."""
        octave_images = gaussian_pyramid.octaves[kp.octave]
        image = octave_images[kp.scale_level]
        
        # Descriptor parameters
        d = self.config.descriptor_window_size
        n_bins = self.config.n_bins_descriptor
        
        # Compute gradient at each point
        h, w = image.shape
        
        # Precompute rotation
        cos_theta = np.cos(kp.orientation)
        sin_theta = np.sin(kp.orientation)
        
        # Descriptor histogram (d x d x n_bins)
        histogram = np.zeros((d, d, n_bins))
        
        # Radius of descriptor window
        radius = d // 2
        sigma = self.config.descriptor_radius_factor * kp.sigma / (2 ** kp.octave)
        
        # Scale factor for sampling
        hist_width = 3 * sigma
        
        for dy in range(-radius * 4, radius * 4 + 1):
            for dx in range(-radius * 4, radius * 4 + 1):
                # Rotate coordinates
                x_rot = (dx * cos_theta - dy * sin_theta) / hist_width
                y_rot = (dx * sin_theta + dy * cos_theta) / hist_width
                
                # Check if within descriptor window
                if abs(x_rot) > radius or abs(y_rot) > radius:
                    continue
                
                # Sample position
                x_sample = int(round(kp.x + dx))
                y_sample = int(round(kp.y + dy))
                
                if x_sample < 1 or x_sample >= w - 1 or y_sample < 1 or y_sample >= h - 1:
                    continue
                
                # Compute gradient
                gx = image[y_sample, x_sample+1] - image[y_sample, x_sample-1]
                gy = image[y_sample+1, x_sample] - image[y_sample-1, x_sample]
                
                magnitude = np.sqrt(gx*gx + gy*gy)
                orientation = np.arctan2(gy, gx) - kp.orientation
                
                # Rotate gradient to keypoint orientation
                while orientation < 0:
                    orientation += 2 * np.pi
                while orientation >= 2 * np.pi:
                    orientation -= 2 * np.pi
                
                # Gaussian weight
                weight = np.exp(-(x_rot*x_rot + y_rot*y_rot) / (2 * (d/2)**2))
                
                # Trilinear interpolation
                bin_x = (x_rot + radius) / (2 * radius) * d - 0.5
                bin_y = (y_rot + radius) / (2 * radius) * d - 0.5
                bin_o = orientation / (2 * np.pi) * n_bins
                
                # Distribute to neighboring bins
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            bx = int(np.floor(bin_x)) + i
                            by = int(np.floor(bin_y)) + j
                            bo = int(np.floor(bin_o)) + k
                            
                            if 0 <= bx < d and 0 <= by < d:
                                bo = bo % n_bins
                                
                                wx = 1 - abs(bin_x - bx)
                                wy = 1 - abs(bin_y - by)
                                wo = 1 - abs(bin_o - bo)
                                
                                histogram[by, bx, bo] += weight * magnitude * wx * wy * wo
        
        # Flatten descriptor
        descriptor = histogram.flatten()
        
        # Normalize
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        # Clamp values to 0.2 (illumination normalization)
        descriptor = np.clip(descriptor, 0, 0.2)
        
        # Renormalize
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor = descriptor / norm
        
        # Convert to uint8 for storage
        descriptor = np.clip(descriptor * 512, 0, 255).astype(np.uint8)
        
        return descriptor


class OpenCVSIFT:
    """
    OpenCV-based SIFT implementation.
    
    This class wraps OpenCV's SIFT implementation (cv2.SIFT_create())
    for better performance when opencv-contrib-python is available.
    
    Requires: pip install opencv-contrib-python
    """
    
    def __init__(self, config: Optional[SIFTConfig] = None):
        if not _has_opencv_sift():
            raise ImportError(
                "OpenCV SIFT not available. Install with: "
                "pip install opencv-contrib-python"
            )
        
        self.config = config or SIFTConfig()
        import cv2
        
        # Map our config to OpenCV parameters
        self._sift = cv2.SIFT_create(
            nfeatures=0,  # No limit
            nOctaveLayers=self.config.n_scales_per_octave,
            contrastThreshold=self.config.contrast_threshold,
            edgeThreshold=self.config.edge_threshold,
            sigma=self.config.sigma_init
        )
    
    def detect(self, image: np.ndarray) -> List[Keypoint]:
        """
        Detect keypoints in an image using OpenCV SIFT.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
        
        Returns:
            List of detected keypoints
        """
        import cv2
        
        # Convert to grayscale uint8 for OpenCV
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float32)
        
        # Normalize to 0-255
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
        
        # Detect keypoints
        cv_kps = self._sift.detect(gray, None)
        
        # Convert to custom format
        keypoints = [_cv_keypoint_to_custom(kp) for kp in cv_kps]
        
        return keypoints
    
    def detect_and_compute(self, image: np.ndarray) -> List[Keypoint]:
        """
        Detect keypoints and compute descriptors using OpenCV SIFT.
        
        Args:
            image: Input image as numpy array (H, W) or (H, W, C)
        
        Returns:
            List of keypoints with descriptors
        """
        import cv2
        
        # Convert to grayscale uint8 for OpenCV
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.float32)
        
        # Normalize to 0-255
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
        
        # Detect and compute
        cv_kps, descriptors = self._sift.detectAndCompute(gray, None)
        
        if cv_kps is None:
            return []
        
        # Convert to custom format
        keypoints = []
        for i, cv_kp in enumerate(cv_kps):
            kp = _cv_keypoint_to_custom(cv_kp)
            if descriptors is not None and i < len(descriptors):
                kp.descriptor = descriptors[i].astype(np.uint8)
            keypoints.append(kp)
        
        return keypoints


class SIFT:
    """Complete SIFT implementation for image feature detection."""
    
    def __init__(self, config: Optional[SIFTConfig] = None):
        self.config = config or SIFTConfig()
        self.gaussian_pyramid = GaussianPyramid(self.config)
        self.keypoint_detector = KeypointDetector(self.config)
        self.orientation_assigner = OrientationAssigner(self.config)
        self.descriptor_computer = DescriptorComputer(self.config)
    
    def detect_and_compute(self, image: np.ndarray) -> List[Keypoint]:
        """Detect keypoints and compute descriptors for an image."""
        # Build Gaussian pyramid
        self.gaussian_pyramid.build(image)
        
        # Build DoG pyramid
        dog_pyramid = DoGPyramid(self.gaussian_pyramid)
        dog_pyramid.build()
        
        # Detect keypoints
        keypoints = self.keypoint_detector.detect(
            dog_pyramid, self.gaussian_pyramid.sigmas
        )
        
        # Assign orientations
        keypoints = self.orientation_assigner.assign(keypoints, self.gaussian_pyramid)
        
        # Compute descriptors
        keypoints = self.descriptor_computer.compute(keypoints, self.gaussian_pyramid)
        
        return keypoints
    
    def detect(self, image: np.ndarray) -> List[Keypoint]:
        """Detect keypoints only (no descriptors)."""
        # Build Gaussian pyramid
        self.gaussian_pyramid.build(image)
        
        # Build DoG pyramid
        dog_pyramid = DoGPyramid(self.gaussian_pyramid)
        dog_pyramid.build()
        
        # Detect keypoints
        keypoints = self.keypoint_detector.detect(
            dog_pyramid, self.gaussian_pyramid.sigmas
        )
        
        # Assign orientations
        keypoints = self.orientation_assigner.assign(keypoints, self.gaussian_pyramid)
        
        return keypoints


class SIFTMatcher:
    """Match SIFT features between images."""
    
    def __init__(self, config: Optional[SIFTConfig] = None):
        self.config = config or SIFTConfig()
    
    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[List[Keypoint], List[Keypoint], List[Match]]:
        """Find matching features between two images."""
        # Validate inputs
        _validate_inputs(image1, image2, allow_different_dtypes=True)
        
        # Detect features in both images using separate SIFT instances
        sift1 = SIFT(self.config)
        sift2 = SIFT(self.config)
        keypoints1 = sift1.detect_and_compute(image1)
        keypoints2 = sift2.detect_and_compute(image2)
        
        # Match features
        matches = self._match_keypoints(keypoints1, keypoints2)
        
        return keypoints1, keypoints2, matches
    
    def _match_keypoints(
        self,
        keypoints1: List[Keypoint],
        keypoints2: List[Keypoint]
    ) -> List[Match]:
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
            
            # Ratio test (Lowe's criterion) - accept if ratio is small enough
            # For identical images, best_dist should be 0 (perfect match)
            if best_dist < 0.001:  # Perfect or near-perfect match
                matches.append(Match(kp1, kp2_with_desc[best_idx], float(best_dist)))
            elif second_best_dist > 0 and best_dist / second_best_dist < self.config.match_ratio_threshold:
                matches.append(Match(kp1, kp2_with_desc[best_idx], float(best_dist)))
        
        return matches


class SIFTDiff:
    """SIFT-based image difference detection."""
    
    def __init__(self, config: Optional[SIFTConfig] = None):
        self.config = config or SIFTConfig()
        self.matcher = SIFTMatcher(self.config)
    
    def compare(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> dict:
        """
        Compare two images using SIFT features.
        
        Args:
            image1: First image as numpy array (H, W) or (H, W, C)
            image2: Second image as numpy array (H, W) or (H, W, C)
        
        Returns:
            Dictionary with comparison results
        
        Raises:
            SIFTInputError: If input arrays have incompatible dimensions, size, or dtype
            TypeError: If inputs are not numpy arrays
        """
        # Validate inputs
        _validate_inputs(image1, image2, allow_different_dtypes=True)
        """
        Compare two images using SIFT features.
        
        Returns a dictionary with:
        - match_ratio: Ratio of matched features to total features
        - avg_distance: Average descriptor distance for matches
        - n_keypoints1: Number of keypoints in image1
        - n_keypoints2: Number of keypoints in image2
        - n_matches: Number of matched features
        - matched_regions: List of matched region coordinates
        - unmatched_regions1: List of unmatched regions in image1
        - unmatched_regions2: List of unmatched regions in image2
        """
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


def sift_diff(
    image1: np.ndarray,
    image2: np.ndarray,
    **kwargs
) -> dict:
    """
    Convenience function for SIFT-based image difference detection.
    
    Args:
        image1: First image (H x W or H x W x C)
        image2: Second image (H x W or H x W x C)
        **kwargs: Additional configuration parameters
    
    Returns:
        Dictionary with comparison results
    """
    config = SIFTConfig(**kwargs)
    diff = SIFTDiff(config)
    return diff.compare(image1, image2)


def compute_sift_diff(
    image1: np.ndarray,
    image2: np.ndarray,
    **kwargs
) -> dict:
    """
    Compute SIFT-based difference metrics between two images.
    
    This is an alias for sift_diff() that provides a more explicit function name
    for computing difference metrics using SIFT feature matching.
    
    Args:
        image1: First image as numpy array (H, W) or (H, W, C)
        image2: Second image as numpy array (H, W) or (H, W, C)
        **kwargs: Additional configuration parameters (e.g., contrast_threshold,
                  edge_threshold, match_ratio_threshold)
    
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
        SIFTInputError: If input arrays have incompatible dimensions, size, or dtype
        TypeError: If inputs are not numpy arrays
    
    Example:
        >>> import numpy as np
        >>> img1 = np.random.rand(256, 256)
        >>> img2 = np.random.rand(256, 256)
        >>> result = compute_sift_diff(img1, img2)
        >>> print(f"Match ratio: {result['match_ratio']:.3f}")
    """
    return sift_diff(image1, image2, **kwargs)


def create_sift(
    backend: str = 'auto',
    config: Optional[SIFTConfig] = None,
    **kwargs
) -> SIFT:
    """
    Factory function to create SIFT instance with specified backend.
    
    Args:
        backend: Backend to use ('auto', 'opencv', or 'numpy')
        config: Optional SIFT configuration
        **kwargs: Additional configuration parameters (merged with config)
    
    Returns:
        SIFT instance (OpenCVSIFT if available, otherwise custom SIFT)
    
    Raises:
        ImportError: If 'opencv' backend is specified but not available
        ValueError: If invalid backend specified
    
    Example:
        >>> sift = create_sift('opencv')  # Use OpenCV if available
        >>> sift = create_sift('numpy')   # Use custom implementation
        >>> sift = create_sift()          # Auto-detect best available
    """
    if config is None:
        config = SIFTConfig(**kwargs)
    else:
        # Update config with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    if backend == 'auto':
        if _has_opencv_sift():
            return OpenCVSIFT(config)
        return SIFT(config)
    
    elif backend == 'opencv':
        if not _has_opencv_sift():
            raise ImportError(
                "OpenCV SIFT not available. Install with: "
                "pip install opencv-contrib-python"
            )
        return OpenCVSIFT(config)
    
    elif backend == 'numpy':
        return SIFT(config)
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'opencv', or 'numpy'")


# For compatibility with OpenCV-style API
def detectAndCompute(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    backend: str = 'auto',
    **kwargs
) -> Tuple[List, np.ndarray]:
    """
    OpenCV-compatible detectAndCompute function.
    
    Args:
        image: Input image
        mask: Optional mask for keypoint detection
        backend: Backend to use ('auto', 'opencv', 'numpy')
        **kwargs: Additional SIFT configuration
    
    Returns:
        Tuple of (keypoints, descriptors)
        - If OpenCV available and used: keypoints are cv2.KeyPoint objects
        - Otherwise: keypoints are custom KeyPoint objects
    
    Example:
        >>> kps, descs = detectAndCompute(image, backend='opencv')
        >>> kps, descs = detectAndCompute(image, backend='numpy')
    """
    sift = create_sift(backend=backend, **kwargs)
    keypoints = sift.detect_and_compute(image)
    
    # Filter by mask if provided
    if mask is not None:
        keypoints = [kp for kp in keypoints 
                    if mask[int(kp.y), int(kp.x)] > 0]
    
    # Convert to arrays
    try:
        import cv2
        cv_keypoints = []
        descriptors = []
        
        for kp in keypoints:
            if kp.descriptor is not None:
                cv_kp = cv2.KeyPoint(
                    float(kp.x),
                    float(kp.y),
                    float(kp.sigma * 2),
                    np.degrees(kp.orientation),
                    1.0,
                    kp.octave,
                    0
                )
                cv_keypoints.append(cv_kp)
                descriptors.append(kp.descriptor)
        
        if descriptors:
            return cv_keypoints, np.array(descriptors)
        return cv_keypoints, np.array([])
    except ImportError:
        descriptors = [kp.descriptor for kp in keypoints if kp.descriptor is not None]
        return keypoints, np.array(descriptors) if descriptors else np.array([])
