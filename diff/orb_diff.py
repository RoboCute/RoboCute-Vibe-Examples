"""
ORB (Oriented FAST and Rotated BRIEF) implementation for image difference detection.

This module provides ORB-based image comparison capabilities for detecting
structural differences between images. ORB is a fast, rotation-invariant
alternative to SIFT and SURF that is free to use.

Reference:
    Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011).
    ORB: An efficient alternative to SIFT or SURF.
    IEEE International Conference on Computer Vision (ICCV), 2564-2571.

Example:
    >>> import numpy as np
    >>> from diff.orb_diff import ORB, orb_diff
    >>> 
    >>> # Create two images
    >>> img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    >>> img2 = img1.copy()
    >>> 
    >>> # Compute ORB difference
    >>> result = orb_diff(img1, img2)
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
    'ORBInputError',
    'ORBConfig',
    'ORBKeypoint',
    'ORBMatch',
    'ORB',
    'ORBMatcher',
    'ORBDiff',
    'orb_diff',
    'compute_orb_diff',
    'create_orb',
    'has_opencv_orb',
]


def has_opencv_orb() -> bool:
    """
    Check if OpenCV ORB is available.
    
    Returns:
        True if cv2.ORB_create() is available
    
    Example:
        >>> from diff.orb_diff import has_opencv_orb
        >>> if has_opencv_orb():
        ...     print("OpenCV ORB is available")
        ... else:
        ...     print("OpenCV ORB is not available")
    """
    try:
        import cv2
        return hasattr(cv2, 'ORB_create')
    except ImportError:
        return False


def _has_opencv_orb() -> bool:
    """Check if OpenCV ORB is available."""
    try:
        import cv2
        return hasattr(cv2, 'ORB_create')
    except ImportError:
        return False


class ORBInputError(ValueError):
    """Raised when ORB input validation fails."""
    pass


def _validate_inputs(
    img1: np.ndarray,
    img2: np.ndarray,
    allow_different_dtypes: bool = False
) -> None:
    """
    Validate that two numpy arrays are compatible for ORB comparison.
    
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
        ORBInputError: If validation fails with detailed error message
        TypeError: If inputs are not numpy arrays
    """
    # Type validation
    if not isinstance(img1, np.ndarray):
        raise TypeError(f"img1 must be a numpy array, got {type(img1).__name__}")
    if not isinstance(img2, np.ndarray):
        raise TypeError(f"img2 must be a numpy array, got {type(img2).__name__}")
    
    # Dimension validation
    if img1.ndim != img2.ndim:
        raise ORBInputError(
            f"Dimension mismatch: img1 has {img1.ndim}D, img2 has {img2.ndim}D. "
            f"Both arrays must have the same number of dimensions."
        )
    
    # Shape validation
    if img1.shape != img2.shape:
        raise ORBInputError(
            f"Shape mismatch: img1 shape {img1.shape} != img2 shape {img2.shape}. "
            f"Both arrays must have identical shapes."
        )
    
    # Size validation (redundant with shape but explicit)
    if img1.size != img2.size:
        raise ORBInputError(
            f"Size mismatch: img1 has {img1.size} elements, img2 has {img2.size} elements. "
            f"Both arrays must have the same number of elements."
        )
    
    # Empty array validation
    if img1.size == 0 or img2.size == 0:
        raise ORBInputError(
            f"Empty input arrays: cannot compute ORB on empty arrays "
            f"(img1 shape: {img1.shape}, img2 shape: {img2.shape})"
        )
    
    # Dtype validation
    if not allow_different_dtypes and img1.dtype != img2.dtype:
        raise ORBInputError(
            f"Dtype mismatch: img1 dtype '{img1.dtype}' != img2 dtype '{img2.dtype}'. "
            f"Both arrays should have the same data type for consistent comparison."
        )
    
    # Valid dimensions check (2D or 3D for images)
    if img1.ndim not in (2, 3):
        raise ORBInputError(
            f"Invalid dimensions: expected 2D or 3D arrays for image comparison, "
            f"got {img1.ndim}D array with shape {img1.shape}"
        )


class ORBConfig:
    """Configuration parameters for ORB algorithm."""
    
    def __init__(
        self,
        n_features: int = 500,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
        first_level: int = 0,
        WTA_K: int = 2,
        score_type: int = 0,  # 0 = HARRIS_SCORE, 1 = FAST_SCORE
        patch_size: int = 31,
        fast_threshold: int = 20,
        match_ratio_threshold: float = 0.75,
        ransac_threshold: float = 3.0,
        min_matches_for_homography: int = 4,
    ):
        self.n_features = n_features
        self.scale_factor = scale_factor
        self.n_levels = n_levels
        self.edge_threshold = edge_threshold
        self.first_level = first_level
        self.WTA_K = WTA_K
        self.score_type = score_type
        self.patch_size = patch_size
        self.fast_threshold = fast_threshold
        self.match_ratio_threshold = match_ratio_threshold
        self.ransac_threshold = ransac_threshold
        self.min_matches_for_homography = min_matches_for_homography


@dataclass
class ORBKeypoint:
    """ORB keypoint with location, scale, and orientation."""
    x: float
    y: float
    size: float
    angle: float
    response: float
    octave: int
    class_id: int
    descriptor: Optional[np.ndarray] = None
    
    @property
    def pt(self) -> Tuple[float, float]:
        """Return point as (x, y) tuple."""
        return (self.x, self.y)


@dataclass
class ORBMatch:
    """Match between two ORB keypoints."""
    kp1: ORBKeypoint
    kp2: ORBKeypoint
    distance: float
    query_idx: int
    train_idx: int


def _cv_keypoint_to_custom(cv_kp, descriptor: Optional[np.ndarray] = None) -> ORBKeypoint:
    """Convert OpenCV KeyPoint to custom ORBKeypoint."""
    return ORBKeypoint(
        x=cv_kp.pt[0],
        y=cv_kp.pt[1],
        size=cv_kp.size,
        angle=cv_kp.angle,
        response=cv_kp.response,
        octave=cv_kp.octave,
        class_id=cv_kp.class_id,
        descriptor=descriptor
    )


class ORB:
    """
    ORB (Oriented FAST and Rotated BRIEF) implementation.
    
    Uses OpenCV's ORB implementation for fast keypoint detection and
    binary descriptor computation.
    
    Example:
        >>> import numpy as np
        >>> from diff.orb_diff import ORB
        >>> 
        >>> img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> orb = ORB(n_features=1000)
        >>> keypoints = orb.detect_and_compute(img)
        >>> print(f"Found {len(keypoints)} keypoints")
    """
    
    def __init__(self, config: Optional[ORBConfig] = None):
        """
        Initialize ORB detector.
        
        Args:
            config: ORB configuration. If None, uses default config.
        
        Raises:
            ImportError: If OpenCV is not available.
        """
        if not _has_opencv_orb():
            raise ImportError(
                "OpenCV ORB not available. Install with: pip install opencv-python"
            )
        
        self.config = config or ORBConfig()
        import cv2
        
        # Map our config to OpenCV parameters
        self._orb = cv2.ORB_create(
            nfeatures=self.config.n_features,
            scaleFactor=self.config.scale_factor,
            nlevels=self.config.n_levels,
            edgeThreshold=self.config.edge_threshold,
            firstLevel=self.config.first_level,
            WTA_K=self.config.WTA_K,
            scoreType=self.config.score_type,
            patchSize=self.config.patch_size,
            fastThreshold=self.config.fast_threshold
        )
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for ORB processing.
        
        Converts to grayscale uint8 format as required by ORB.
        
        Args:
            image: Input image array
        
        Returns:
            uint8 grayscale image
        """
        import cv2
        
        # Handle color images
        if image.ndim == 3:
            if image.shape[2] == 3:
                if image.dtype == np.uint8:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    # Convert float to uint8 first
                    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.0 else np.clip(image, 0, 255).astype(np.uint8)
                    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                if image.dtype == np.uint8:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                else:
                    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1.0 else np.clip(image, 0, 255).astype(np.uint8)
                    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGBA2GRAY)
            else:
                # Take first channel
                gray = image[..., 0]
        else:
            gray = image
        
        # Normalize to uint8 range
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = np.clip(gray, 0, 255).astype(np.uint8)
        
        return gray
    
    def detect(self, image: np.ndarray) -> List[ORBKeypoint]:
        """
        Detect keypoints in an image.
        
        Args:
            image: Input image as numpy array (2D or 3D)
        
        Returns:
            List of ORBKeypoint objects
        """
        # Prepare image
        gray = self._prepare_image(image)
        
        # Detect keypoints
        cv_kps = self._orb.detect(gray, None)
        
        # Convert to custom format
        return [_cv_keypoint_to_custom(kp) for kp in cv_kps]
    
    def detect_and_compute(self, image: np.ndarray) -> List[ORBKeypoint]:
        """
        Detect keypoints and compute descriptors in one call.
        
        Args:
            image: Input image as numpy array (2D or 3D)
        
        Returns:
            List of ORBKeypoint objects with descriptors
        """
        # Prepare image
        gray = self._prepare_image(image)
        
        # Detect and compute
        cv_kps, descriptors = self._orb.detectAndCompute(gray, None)
        
        if cv_kps is None or len(cv_kps) == 0:
            return []
        
        # Convert to custom format
        result = []
        for i, kp in enumerate(cv_kps):
            desc = descriptors[i] if descriptors is not None else None
            result.append(_cv_keypoint_to_custom(kp, desc))
        
        return result
    
    def compute(
        self,
        image: np.ndarray,
        keypoints: List[ORBKeypoint]
    ) -> List[ORBKeypoint]:
        """
        Compute descriptors for given keypoints.
        
        Args:
            image: Input image as numpy array
            keypoints: List of keypoints to compute descriptors for
        
        Returns:
            List of ORBKeypoint objects with descriptors
        """
        import cv2
        
        # Prepare image
        gray = self._prepare_image(image)
        
        # Convert custom keypoints to OpenCV format
        cv_kps = self._custom_to_cv_keypoints(keypoints)
        
        # Compute descriptors
        cv_kps_out, descriptors = self._orb.compute(gray, cv_kps)
        
        # Convert back to custom format with descriptors
        result = []
        for i, kp in enumerate(cv_kps_out):
            desc = descriptors[i] if descriptors is not None and i < len(descriptors) else None
            result.append(_cv_keypoint_to_custom(kp, desc))
        
        return result
    
    def _custom_to_cv_keypoints(
        self,
        keypoints: List[ORBKeypoint]
    ) -> List['cv2.KeyPoint']:
        """
        Convert custom ORBKeypoint objects to OpenCV KeyPoints.
        
        Args:
            keypoints: List of ORBKeypoint objects
        
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
                _angle=kp.angle,
                _response=kp.response,
                _octave=kp.octave,
                _class_id=kp.class_id
            )
            cv_kps.append(cv_kp)
        
        return cv_kps


class ORBMatcher:
    """Match ORB features between images using BFMatcher."""
    
    def __init__(self, config: Optional[ORBConfig] = None):
        self.config = config or ORBConfig()
    
    def match(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> Tuple[List[ORBKeypoint], List[ORBKeypoint], List[ORBMatch], Optional[np.ndarray]]:
        """
        Find matching features between two images and compute homography.
        
        Args:
            image1: First image
            image2: Second image
        
        Returns:
            Tuple of (keypoints1, keypoints2, matches, homography_matrix)
            - keypoints1: List of keypoints from image1
            - keypoints2: List of keypoints from image2
            - matches: List of ORBMatch objects
            - homography_matrix: 3x3 homography matrix or None if insufficient matches
        """
        import cv2
        
        # Validate inputs
        _validate_inputs(image1, image2, allow_different_dtypes=True)
        
        # Detect features in both images
        orb1 = ORB(self.config)
        orb2 = ORB(self.config)
        keypoints1 = orb1.detect_and_compute(image1)
        keypoints2 = orb2.detect_and_compute(image2)
        
        # Match features
        matches = self._match_keypoints(keypoints1, keypoints2)
        
        # Compute homography if enough matches
        homography = self._compute_homography(keypoints1, keypoints2, matches)
        
        return keypoints1, keypoints2, matches, homography
    
    def _match_keypoints(
        self,
        keypoints1: List[ORBKeypoint],
        keypoints2: List[ORBKeypoint]
    ) -> List[ORBMatch]:
        """
        Match keypoints using BFMatcher with Hamming distance.
        
        Uses Lowe's ratio test to filter good matches.
        """
        import cv2
        
        matches = []
        
        # Filter keypoints with descriptors
        kp1_with_desc = [kp for kp in keypoints1 if kp.descriptor is not None]
        kp2_with_desc = [kp for kp in keypoints2 if kp.descriptor is not None]
        
        if not kp1_with_desc or not kp2_with_desc:
            return matches
        
        # Build descriptor arrays
        desc1 = np.array([kp.descriptor for kp in kp1_with_desc], dtype=np.uint8)
        desc2 = np.array([kp.descriptor for kp in kp2_with_desc], dtype=np.uint8)
        
        # Create BFMatcher with Hamming distance (for binary descriptors)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Find k nearest neighbors for each descriptor
        knn_matches = bf.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        for i, match_pair in enumerate(knn_matches):
            if len(match_pair) < 2:
                continue
            
            m, n = match_pair
            
            # Ratio test (Lowe's criterion)
            if m.distance < self.config.match_ratio_threshold * n.distance:
                matches.append(ORBMatch(
                    kp1=kp1_with_desc[m.queryIdx],
                    kp2=kp2_with_desc[m.trainIdx],
                    distance=float(m.distance),
                    query_idx=m.queryIdx,
                    train_idx=m.trainIdx
                ))
        
        return matches
    
    def _compute_homography(
        self,
        keypoints1: List[ORBKeypoint],
        keypoints2: List[ORBKeypoint],
        matches: List[ORBMatch]
    ) -> Optional[np.ndarray]:
        """
        Compute homography matrix from matched keypoints.
        
        Args:
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            matches: List of matches
        
        Returns:
            3x3 homography matrix or None if insufficient matches
        """
        import cv2
        
        if len(matches) < self.config.min_matches_for_homography:
            return None
        
        # Extract matched points
        src_pts = np.float32([match.kp1.pt for match in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([match.kp2.pt for match in matches]).reshape(-1, 1, 2)
        
        # Compute homography using RANSAC
        try:
            H, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                self.config.ransac_threshold
            )
            return H
        except Exception:
            return None


class ORBDiff:
    """ORB-based image difference detection."""
    
    def __init__(self, config: Optional[ORBConfig] = None):
        self.config = config or ORBConfig()
        self.matcher = ORBMatcher(self.config)
    
    def compare(
        self,
        image1: np.ndarray,
        image2: np.ndarray
    ) -> dict:
        """
        Compare two images using ORB features.
        
        Args:
            image1: First image as numpy array (H, W) or (H, W, C)
            image2: Second image as numpy array (H, W) or (H, W, C)
        
        Returns:
            Dictionary with comparison results including:
            - match_ratio: Ratio of matched features to total features
            - avg_distance: Average descriptor distance for matches
            - n_keypoints1: Number of keypoints in image1
            - n_keypoints2: Number of keypoints in image2
            - n_matches: Number of matched features
            - matched_regions: List of matched region coordinates
            - unmatched_regions1: List of unmatched regions in image1
            - unmatched_regions2: List of unmatched regions in image2
            - homography: 3x3 homography matrix or None
            - diff_mask: Binary mask showing different regions
            - diff_map: Grayscale difference map
            - keypoints1, keypoints2: Full keypoint objects
            - matches: Full match objects
        
        Raises:
            ORBInputError: If input arrays have incompatible dimensions, size, or dtype
            TypeError: If inputs are not numpy arrays
        """
        import cv2
        
        # Validate inputs
        _validate_inputs(image1, image2, allow_different_dtypes=True)
        
        # Match features and compute homography
        keypoints1, keypoints2, matches, homography = self.matcher.match(image1, image2)
        
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
                unmatched1.append((int(kp.x), int(kp.y), kp.size))
        
        for kp in keypoints2:
            if (int(kp.x), int(kp.y)) not in matched2:
                unmatched2.append((int(kp.x), int(kp.y), kp.size))
        
        # Compute difference mask and map
        diff_mask, diff_map = self._compute_difference(
            image1, image2, homography, keypoints1, keypoints2, matches
        )
        
        return {
            'match_ratio': match_ratio,
            'avg_distance': avg_distance,
            'n_keypoints1': n_kp1,
            'n_keypoints2': n_kp2,
            'n_matches': n_matches,
            'matched_regions': list(matched1),
            'unmatched_regions1': unmatched1,
            'unmatched_regions2': unmatched2,
            'homography': homography,
            'diff_mask': diff_mask,
            'diff_map': diff_map,
            'keypoints1': keypoints1,
            'keypoints2': keypoints2,
            'matches': matches
        }
    
    def _compute_difference(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        homography: Optional[np.ndarray],
        keypoints1: List[ORBKeypoint],
        keypoints2: List[ORBKeypoint],
        matches: List[ORBMatch]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute difference mask and map between two images.
        
        Args:
            image1: First image
            image2: Second image
            homography: Homography matrix or None
            keypoints1: Keypoints from image1
            keypoints2: Keypoints from image2
            matches: List of matches
        
        Returns:
            Tuple of (diff_mask, diff_map)
            - diff_mask: Binary mask (uint8) showing different regions
            - diff_map: Grayscale difference map (float32)
        """
        import cv2
        
        # Convert to grayscale for comparison
        if image1.ndim == 3:
            if image1.dtype == np.uint8:
                gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
            else:
                img1_uint8 = (np.clip(image1, 0, 1) * 255).astype(np.uint8) if image1.max() <= 1.0 else np.clip(image1, 0, 255).astype(np.uint8)
                gray1 = cv2.cvtColor(img1_uint8, cv2.COLOR_RGB2GRAY)
        else:
            if image1.dtype == np.uint8:
                gray1 = image1.copy()
            else:
                gray1 = (np.clip(image1, 0, 1) * 255).astype(np.uint8) if image1.max() <= 1.0 else np.clip(image1, 0, 255).astype(np.uint8)
        
        if image2.ndim == 3:
            if image2.dtype == np.uint8:
                gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
            else:
                img2_uint8 = (np.clip(image2, 0, 1) * 255).astype(np.uint8) if image2.max() <= 1.0 else np.clip(image2, 0, 255).astype(np.uint8)
                gray2 = cv2.cvtColor(img2_uint8, cv2.COLOR_RGB2GRAY)
        else:
            if image2.dtype == np.uint8:
                gray2 = image2.copy()
            else:
                gray2 = (np.clip(image2, 0, 1) * 255).astype(np.uint8) if image2.max() <= 1.0 else np.clip(image2, 0, 255).astype(np.uint8)
        
        h, w = gray1.shape[:2]
        
        # If homography exists, warp image1 to align with image2
        if homography is not None:
            try:
                warped1 = cv2.warpPerspective(gray1, homography, (w, h))
                diff = cv2.absdiff(warped1.astype(np.float32), gray2.astype(np.float32))
            except Exception:
                # Fallback to direct comparison if warping fails
                diff = cv2.absdiff(gray1.astype(np.float32), gray2.astype(np.float32))
        else:
            diff = cv2.absdiff(gray1.astype(np.float32), gray2.astype(np.float32))
        
        # Create difference map
        diff_map = diff / 255.0  # Normalize to [0, 1]
        
        # Create binary mask from thresholded difference
        _, diff_mask = cv2.threshold(diff.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
        
        # Add unmatched keypoint regions to mask
        for kp in keypoints1:
            if (int(kp.x), int(kp.y)) not in set((int(m.kp1.x), int(m.kp1.y)) for m in matches):
                cv2.circle(diff_mask, (int(kp.x), int(kp.y)), int(kp.size), 255, -1)
        
        for kp in keypoints2:
            if (int(kp.x), int(kp.y)) not in set((int(m.kp2.x), int(m.kp2.y)) for m in matches):
                cv2.circle(diff_mask, (int(kp.x), int(kp.y)), int(kp.size), 255, -1)
        
        return diff_mask, diff_map
    
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


def orb_diff(
    image1: np.ndarray,
    image2: np.ndarray,
    **kwargs
) -> dict:
    """
    Convenience function for ORB-based image difference detection.
    
    Args:
        image1: First image (H x W or H x W x C)
        image2: Second image (H x W or H x W x C)
        **kwargs: Additional configuration parameters
    
    Returns:
        Dictionary with comparison results including:
        - match_ratio: Ratio of matched features to total features
        - avg_distance: Average descriptor distance for matches
        - n_keypoints1: Number of keypoints in image1
        - n_keypoints2: Number of keypoints in image2
        - n_matches: Number of matched features
        - matched_regions: List of matched region coordinates
        - unmatched_regions1: List of unmatched regions in image1
        - unmatched_regions2: List of unmatched regions in image2
        - homography: 3x3 homography matrix or None
        - diff_mask: Binary mask showing different regions
        - diff_map: Grayscale difference map
        - keypoints1: List of keypoint objects from image1
        - keypoints2: List of keypoint objects from image2
        - matches: List of match objects
    
    Example:
        >>> import numpy as np
        >>> from diff.orb_diff import orb_diff
        >>> 
        >>> img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> img2 = img1.copy()
        >>> result = orb_diff(img1, img2)
        >>> print(f"Match ratio: {result['match_ratio']:.3f}")
        >>> print(f"Homography computed: {result['homography'] is not None}")
    """
    config = ORBConfig(**kwargs)
    diff = ORBDiff(config)
    return diff.compare(image1, image2)


def compute_orb_diff(
    image1: np.ndarray,
    image2: np.ndarray,
    **kwargs
) -> dict:
    """
    Compute ORB-based difference metrics between two images.
    
    This is an alias for orb_diff() that provides a more explicit function name
    for computing difference metrics using ORB feature matching.
    
    Args:
        image1: First image as numpy array (H, W) or (H, W, C)
        image2: Second image as numpy array (H, W) or (H, W, C)
        **kwargs: Additional configuration parameters
    
    Returns:
        Dictionary with difference metrics
    
    Raises:
        ORBInputError: If input arrays have incompatible dimensions, size, or dtype
        TypeError: If inputs are not numpy arrays
    
    Example:
        >>> import numpy as np
        >>> img1 = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        >>> img2 = img1.copy()
        >>> result = compute_orb_diff(img1, img2)
        >>> print(f"Match ratio: {result['match_ratio']:.3f}")
    """
    return orb_diff(image1, image2, **kwargs)


def create_orb(
    config: Optional[ORBConfig] = None,
    **kwargs
) -> ORB:
    """
    Factory function to create ORB instance.
    
    Args:
        config: Optional ORB configuration
        **kwargs: Additional configuration parameters (merged with config)
    
    Returns:
        ORB instance
    
    Example:
        >>> from diff.orb_diff import create_orb
        >>> 
        >>> # Create with default config
        >>> orb = create_orb()
        >>> 
        >>> # Create with custom parameters
        >>> orb = create_orb(n_features=1000, match_ratio_threshold=0.8)
    """
    if config is None:
        config = ORBConfig(**kwargs)
    else:
        # Update config with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return ORB(config)
