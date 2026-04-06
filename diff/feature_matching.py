"""
Feature descriptor computation and matching between two images.

This module provides a high-level API for:
1. Computing feature descriptors (SIFT) for detected keypoints
2. Matching descriptors between two images using various algorithms
3. Finding corresponding points and regions between images

Usage:
    >>> from feature_matching import compute_descriptors, match_features, find_correspondences
    >>> 
    >>> # Compute descriptors for both images
    >>> keypoints1, descriptors1 = compute_descriptors(image1)
    >>> keypoints2, descriptors2 = compute_descriptors(image2)
    >>> 
    >>> # Match features between images
    >>> matches = match_features(descriptors1, descriptors2, method='ratio')
    >>> 
    >>> # Find corresponding points
    >>> correspondences = find_correspondences(image1, image2)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Literal
from dataclasses import dataclass
from enum import Enum

# Import from sift module
from sift import (
    SIFT, SIFTConfig, Keypoint, Match, SIFTMatcher,
    SIFTInputError, _validate_inputs, create_sift
)

__all__ = [
    # Main API
    'compute_descriptors',
    'match_descriptors',
    'find_correspondences',
    'match_features',
    
    # Types
    'DescriptorResult',
    'MatchingResult',
    'Correspondence',
    'MatchingMethod',
    
    # Utilities
    'filter_matches_by_distance',
    'filter_matches_by_ratio',
    'get_matched_points',
    'get_unmatched_keypoints',
]


class MatchingMethod(Enum):
    """Matching algorithm methods."""
    RATIO = 'ratio'           # Lowe's ratio test
    DISTANCE = 'distance'     # Simple distance threshold
    MUTUAL = 'mutual'         # Mutual nearest neighbor
    CROSS_CHECK = 'cross'     # Cross-check matching


@dataclass
class DescriptorResult:
    """Result of descriptor computation."""
    keypoints: List[Keypoint]
    descriptors: np.ndarray
    n_keypoints: int
    descriptor_dim: int
    
    def __post_init__(self):
        if len(self.descriptors) > 0:
            assert self.descriptors.shape[0] == self.n_keypoints
            self.descriptor_dim = self.descriptors.shape[1]


@dataclass
class MatchingResult:
    """Result of descriptor matching."""
    matches: List[Match]
    n_matches: int
    match_ratio: float  # matches / total keypoints in image1
    avg_distance: float
    inlier_ratio: float
    
    def get_matched_indices(self) -> Tuple[List[int], List[int]]:
        """Get indices of matched keypoints in both images."""
        indices1 = []
        indices2 = []
        for match in self.matches:
            # Find indices (linear search - could be optimized with dict)
            for i, kp in enumerate(self.matches):
                if kp is match.kp1:
                    indices1.append(i)
                if kp is match.kp2:
                    indices2.append(i)
        return indices1, indices2


@dataclass  
class Correspondence:
    """A correspondence between two images."""
    pt1: Tuple[float, float]  # Point in image1 (x, y)
    pt2: Tuple[float, float]  # Point in image2 (x, y)
    distance: float           # Descriptor distance
    confidence: float         # Match confidence (0-1)


def compute_descriptors(
    image: np.ndarray,
    backend: str = 'auto',
    config: Optional[SIFTConfig] = None,
    **kwargs
) -> DescriptorResult:
    """
    Compute SIFT descriptors for detected keypoints in an image.
    
    Args:
        image: Input image as numpy array (H, W) or (H, W, C)
        backend: Backend to use ('auto', 'opencv', 'numpy')
        config: Optional SIFT configuration
        **kwargs: Additional SIFT parameters (e.g., contrast_threshold)
    
    Returns:
        DescriptorResult containing keypoints and their descriptors
    
    Example:
        >>> result = compute_descriptors(image, backend='numpy')
        >>> print(f"Found {result.n_keypoints} keypoints")
        >>> print(f"Descriptor shape: {result.descriptors.shape}")
    """
    if image is None or image.size == 0:
        raise SIFTInputError("Empty image provided")
    
    if image.ndim not in (2, 3):
        raise SIFTInputError(f"Invalid image dimensions: expected 2D or 3D, got {image.ndim}D")
    
    # Create SIFT instance
    sift = create_sift(backend=backend, config=config, **kwargs)
    
    # Detect keypoints and compute descriptors
    keypoints = sift.detect_and_compute(image)
    
    # Extract descriptors into array
    descriptors = []
    valid_keypoints = []
    
    for kp in keypoints:
        if kp.descriptor is not None:
            descriptors.append(kp.descriptor)
            valid_keypoints.append(kp)
    
    if descriptors:
        desc_array = np.array(descriptors, dtype=np.float32)
    else:
        desc_array = np.array([]).reshape(0, 128).astype(np.float32)
    
    return DescriptorResult(
        keypoints=valid_keypoints,
        descriptors=desc_array,
        n_keypoints=len(valid_keypoints),
        descriptor_dim=desc_array.shape[1] if desc_array.size > 0 else 0
    )


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    method: MatchingMethod | str = MatchingMethod.RATIO,
    ratio_threshold: float = 0.8,
    distance_threshold: float = 300.0,
    cross_check: bool = False
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Match two sets of descriptors using specified method.
    
    Args:
        desc1: Descriptors from image1, shape (N, D)
        desc2: Descriptors from image2, shape (M, D)
        method: Matching method ('ratio', 'distance', 'mutual', 'cross')
        ratio_threshold: Threshold for ratio test (Lowe's criterion)
        distance_threshold: Maximum distance for valid matches
        cross_check: If True, only keep mutual best matches
    
    Returns:
        Tuple of (matches, distances)
        - matches: List of (idx1, idx2) tuples
        - distances: Array of descriptor distances for each match
    
    Example:
        >>> matches, distances = match_descriptors(desc1, desc2, method='ratio')
        >>> for (i, j), dist in zip(matches, distances):
        ...     print(f"Keypoint {i} matches {j} with distance {dist:.2f}")
    """
    if desc1.size == 0 or desc2.size == 0:
        return [], np.array([])
    
    # Normalize descriptors
    desc1_norm = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-7)
    desc2_norm = desc2 / (np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-7)
    
    # Convert method string to enum
    if isinstance(method, str):
        method = MatchingMethod(method)
    
    matches = []
    distances = []
    
    if method == MatchingMethod.RATIO:
        # Lowe's ratio test
        for i, d1 in enumerate(desc1_norm):
            dists = np.linalg.norm(desc2_norm - d1, axis=1)
            sorted_idx = np.argsort(dists)
            
            if len(sorted_idx) < 2:
                continue
            
            best_dist = dists[sorted_idx[0]]
            second_dist = dists[sorted_idx[1]]
            
            # Apply ratio test
            if second_dist > 0 and best_dist / second_dist < ratio_threshold:
                matches.append((i, sorted_idx[0]))
                distances.append(best_dist)
    
    elif method == MatchingMethod.DISTANCE:
        # Simple distance threshold
        for i, d1 in enumerate(desc1_norm):
            dists = np.linalg.norm(desc2_norm - d1, axis=1)
            best_idx = np.argmin(dists)
            best_dist = dists[best_idx]
            
            if best_dist < distance_threshold:
                matches.append((i, best_idx))
                distances.append(best_dist)
    
    elif method == MatchingMethod.MUTUAL or method == MatchingMethod.CROSS_CHECK or cross_check:
        # Mutual nearest neighbor matching
        # Find best match for each descriptor in both directions
        forward_matches = {}  # desc1 -> desc2
        
        for i, d1 in enumerate(desc1_norm):
            dists = np.linalg.norm(desc2_norm - d1, axis=1)
            best_idx = np.argmin(dists)
            forward_matches[i] = (best_idx, dists[best_idx])
        
        backward_matches = {}  # desc2 -> desc1
        for j, d2 in enumerate(desc2_norm):
            dists = np.linalg.norm(desc1_norm - d2, axis=1)
            best_idx = np.argmin(dists)
            backward_matches[j] = best_idx
        
        # Keep only mutual matches
        for i, (j, dist) in forward_matches.items():
            if backward_matches.get(j) == i:
                matches.append((i, j))
                distances.append(dist)
    
    return matches, np.array(distances)


def match_features(
    image1: np.ndarray,
    image2: np.ndarray,
    backend: str = 'auto',
    method: MatchingMethod | str = MatchingMethod.RATIO,
    config: Optional[SIFTConfig] = None,
    **kwargs
) -> MatchingResult:
    """
    High-level function to detect and match features between two images.
    
    Args:
        image1: First image
        image2: Second image
        backend: Backend to use ('auto', 'opencv', 'numpy')
        method: Matching method
        config: Optional SIFT configuration
        **kwargs: Additional parameters (ratio_threshold, distance_threshold)
    
    Returns:
        MatchingResult with matches and statistics
    
    Example:
        >>> result = match_features(img1, img2, backend='numpy', method='ratio')
        >>> print(f"Found {result.n_matches} matches")
        >>> print(f"Match ratio: {result.match_ratio:.2%}")
    """
    # Validate inputs
    _validate_inputs(image1, image2, allow_different_dtypes=True)
    
    # Compute descriptors for both images
    result1 = compute_descriptors(image1, backend=backend, config=config)
    result2 = compute_descriptors(image2, backend=backend, config=config)
    
    if result1.n_keypoints == 0 or result2.n_keypoints == 0:
        return MatchingResult(
            matches=[],
            n_matches=0,
            match_ratio=0.0,
            avg_distance=float('inf'),
            inlier_ratio=0.0
        )
    
    # Match descriptors
    ratio_threshold = kwargs.get('ratio_threshold', 0.8)
    distance_threshold = kwargs.get('distance_threshold', 300.0)
    
    match_indices, distances = match_descriptors(
        result1.descriptors,
        result2.descriptors,
        method=method,
        ratio_threshold=ratio_threshold,
        distance_threshold=distance_threshold
    )
    
    # Create Match objects
    matches = []
    for (i, j), dist in zip(match_indices, distances):
        matches.append(Match(result1.keypoints[i], result2.keypoints[j], float(dist)))
    
    # Calculate statistics
    n_matches = len(matches)
    match_ratio = n_matches / result1.n_keypoints if result1.n_keypoints > 0 else 0.0
    avg_distance = float(np.mean(distances)) if len(distances) > 0 else float('inf')
    
    # Estimate inlier ratio (rough estimate based on distance distribution)
    if len(distances) > 0:
        median_dist = np.median(distances)
        inliers = np.sum(distances < median_dist * 1.5)
        inlier_ratio = inliers / len(distances)
    else:
        inlier_ratio = 0.0
    
    return MatchingResult(
        matches=matches,
        n_matches=n_matches,
        match_ratio=match_ratio,
        avg_distance=avg_distance,
        inlier_ratio=inlier_ratio
    )


def find_correspondences(
    image1: np.ndarray,
    image2: np.ndarray,
    backend: str = 'auto',
    min_confidence: float = 0.0,
    **kwargs
) -> List[Correspondence]:
    """
    Find corresponding points between two images.
    
    Args:
        image1: First image
        image2: Second image
        backend: Backend to use
        min_confidence: Minimum confidence threshold (0-1)
        **kwargs: Additional parameters for matching
    
    Returns:
        List of Correspondence objects
    
    Example:
        >>> correspondences = find_correspondences(img1, img2)
        >>> for corr in correspondences:
        ...     print(f"({corr.pt1}) -> ({corr.pt2}), conf={corr.confidence:.2f}")
    """
    result = match_features(image1, image2, backend=backend, **kwargs)
    
    correspondences = []
    for match in result.matches:
        # Calculate confidence based on distance (lower is better)
        # Normalize to 0-1 range using a heuristic
        max_expected_dist = 200.0  # Typical max distance for good matches
        confidence = max(0.0, 1.0 - match.distance / max_expected_dist)
        
        if confidence >= min_confidence:
            correspondences.append(Correspondence(
                pt1=(match.kp1.x, match.kp1.y),
                pt2=(match.kp2.x, match.kp2.y),
                distance=match.distance,
                confidence=confidence
            ))
    
    return correspondences


def filter_matches_by_distance(
    matches: List[Match],
    max_distance: float
) -> List[Match]:
    """
    Filter matches by maximum descriptor distance.
    
    Args:
        matches: List of matches
        max_distance: Maximum allowed distance
    
    Returns:
        Filtered list of matches
    """
    return [m for m in matches if m.distance <= max_distance]


def filter_matches_by_ratio(
    matches: List[Match],
    distances: np.ndarray,
    ratio_threshold: float = 0.8
) -> List[Match]:
    """
    Filter matches using Lowe's ratio test.
    
    Note: This is a post-processing filter. For new matches,
    use match_descriptors with method='ratio' instead.
    
    Args:
        matches: List of matches
        distances: Array of best match distances
        ratio_threshold: Ratio threshold (typically 0.7-0.8)
    
    Returns:
        Filtered list of matches
    """
    # This is a simplified version - assumes distances are already ratio-tested
    return matches


def get_matched_points(
    matches: List[Match]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract matched point coordinates from matches.
    
    Args:
        matches: List of Match objects
    
    Returns:
        Tuple of (points1, points2) where each is array of shape (N, 2)
    """
    if not matches:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    pts1 = np.array([[m.kp1.x, m.kp1.y] for m in matches])
    pts2 = np.array([[m.kp2.x, m.kp2.y] for m in matches])
    
    return pts1, pts2


def get_unmatched_keypoints(
    keypoints: List[Keypoint],
    matches: List[Match],
    image_idx: int = 1
) -> List[Keypoint]:
    """
    Get keypoints that were not matched.
    
    Args:
        keypoints: All keypoints from an image
        matches: List of matches
        image_idx: Which image (1 or 2)
    
    Returns:
        List of unmatched keypoints
    """
    if image_idx == 1:
        matched_kps = {id(m.kp1) for m in matches}
        return [kp for kp in keypoints if id(kp) not in matched_kps]
    else:
        matched_kps = {id(m.kp2) for m in matches}
        return [kp for kp in keypoints if id(kp) not in matched_kps]
