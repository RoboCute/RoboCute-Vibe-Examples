"""
Difference detection logic that identifies regions with significant feature mismatches.

This module provides algorithms to:
1. Analyze feature matches and identify mismatched regions
2. Cluster unmatched/poorly-matched features into difference regions
3. Compute difference metrics and confidence scores
4. Provide region-based change detection

Usage:
    >>> from diff_regions import detect_difference_regions, DiffRegionDetector
    >>> 
    >>> # Detect difference regions between two images
    >>> regions = detect_difference_regions(image1, image2)
    >>> 
    >>> for region in regions:
    ...     print(f"Change at ({region.x}, {region.y}): size={region.width}x{region.height}, "
    ...           f"severity={region.severity:.2f}")
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from sift import Keypoint, Match, SIFTInputError, _validate_inputs
from feature_matching import (
    match_features, 
    get_matched_points,
    MatchingResult,
    MatchingMethod
)

__all__ = [
    # Main API
    'detect_difference_regions',
    'DiffRegionDetector',
    'compute_difference_map',
    
    # Types
    'DiffRegion',
    'DifferenceSeverity',
    'RegionDiffResult',
    
    # Analysis
    'analyze_match_quality',
    'find_mismatched_regions',
    'cluster_keypoints_into_regions',
]


class DifferenceSeverity(Enum):
    """Severity levels for detected differences."""
    NONE = 0      # No significant difference
    LOW = 1       # Minor differences
    MEDIUM = 2    # Moderate differences  
    HIGH = 3      # Significant differences
    CRITICAL = 4  # Major structural changes


@dataclass
class DiffRegion:
    """
    A detected difference region between two images.
    
    Attributes:
        x, y: Top-left corner coordinates
        width, height: Region dimensions
        severity: Difference severity level
        confidence: Detection confidence (0-1)
        unmatched_count: Number of unmatched keypoints in this region
        avg_match_distance: Average descriptor distance for matches
        keypoints_img1: Keypoints from image 1 in this region
        keypoints_img2: Keypoints from image 2 in this region
    """
    x: int
    y: int
    width: int
    height: int
    severity: DifferenceSeverity
    confidence: float
    unmatched_count: int
    avg_match_distance: float
    keypoints_img1: List[Keypoint]
    keypoints_img2: List[Keypoint]
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get region center coordinates."""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> int:
        """Get region area in pixels."""
        return self.width * self.height


@dataclass
class RegionDiffResult:
    """Complete difference detection result."""
    regions: List[DiffRegion]
    n_regions: int
    total_diff_area: int
    diff_coverage: float  # Percentage of image covered by differences
    global_severity: DifferenceSeverity
    match_result: MatchingResult
    
    def get_regions_by_severity(self, min_severity: DifferenceSeverity) -> List[DiffRegion]:
        """Get regions with at least the specified severity."""
        return [r for r in self.regions if r.severity.value >= min_severity.value]


@dataclass
class MatchQuality:
    """Quality metrics for a feature match."""
    match: Match
    is_good: bool
    distance_score: float  # 0-1, higher is better
    geometric_consistency: float  # 0-1, consistency with neighbors


def analyze_match_quality(
    matches: List[Match],
    keypoints1: List[Keypoint],
    keypoints2: List[Keypoint],
    distance_threshold: float = 100.0,
    neighbor_radius: float = 50.0
) -> List[MatchQuality]:
    """
    Analyze quality of feature matches.
    
    Evaluates each match based on:
    - Descriptor distance (lower is better)
    - Geometric consistency with neighboring matches
    
    Args:
        matches: List of matches to analyze
        keypoints1: All keypoints from image 1
        keypoints2: All keypoints from image 2
        distance_threshold: Max distance for a "good" match
        neighbor_radius: Radius for neighbor consistency check
    
    Returns:
        List of MatchQuality objects with quality metrics
    """
    if not matches:
        return []
    
    # Build spatial index for keypoints
    def get_neighbors(kp: Keypoint, keypoints: List[Keypoint], radius: float) -> List[Keypoint]:
        """Find neighboring keypoints within radius."""
        neighbors = []
        for other in keypoints:
            if other is kp:
                continue
            dist = np.sqrt((kp.x - other.x)**2 + (kp.y - other.y)**2)
            if dist <= radius:
                neighbors.append(other)
        return neighbors
    
    # Create lookup for matches
    match_lookup = {}
    for m in matches:
        match_lookup[(id(m.kp1), id(m.kp2))] = m
    
    qualities = []
    
    for match in matches:
        # Distance score (normalize to 0-1)
        max_expected_dist = 300.0
        distance_score = max(0.0, 1.0 - match.distance / max_expected_dist)
        is_good = match.distance < distance_threshold
        
        # Geometric consistency check
        # In a good match, neighboring keypoints should have similar displacement
        neighbors1 = get_neighbors(match.kp1, keypoints1, neighbor_radius)
        
        if len(neighbors1) >= 2:
            # Compute displacement vector for this match
            displacement = np.array([match.kp2.x - match.kp1.x, match.kp2.y - match.kp1.y])
            
            # Check displacement consistency with neighbors
            consistent_count = 0
            for n1 in neighbors1:
                # Find if this neighbor has a match
                found_match = None
                for m in matches:
                    if m.kp1 is n1:
                        found_match = m
                        break
                
                if found_match:
                    neighbor_disp = np.array([
                        found_match.kp2.x - found_match.kp1.x,
                        found_match.kp2.y - found_match.kp1.y
                    ])
                    # Check if displacement is similar
                    diff = np.linalg.norm(displacement - neighbor_disp)
                    if diff < neighbor_radius * 0.5:  # Within half radius
                        consistent_count += 1
            
            geometric_consistency = consistent_count / len(neighbors1) if neighbors1 else 0.0
        else:
            geometric_consistency = 0.5  # Neutral if no neighbors
        
        qualities.append(MatchQuality(
            match=match,
            is_good=is_good,
            distance_score=distance_score,
            geometric_consistency=geometric_consistency
        ))
    
    return qualities


def cluster_keypoints_into_regions(
    keypoints: List[Keypoint],
    image_shape: Tuple[int, ...],
    cluster_radius: float = 30.0,
    min_cluster_size: int = 3
) -> List[Tuple[int, int, int, int]]:
    """
    Cluster keypoints into rectangular regions using simple density-based clustering.
    
    Args:
        keypoints: List of keypoints to cluster
        image_shape: Shape of the image (H, W) or (H, W, C)
        cluster_radius: Maximum distance between keypoints in same cluster
        min_cluster_size: Minimum keypoints to form a region
    
    Returns:
        List of (x, y, width, height) rectangles
    """
    if not keypoints:
        return []
    
    if len(image_shape) >= 2:
        h, w = image_shape[0], image_shape[1]
    else:
        return []
    
    # Simple clustering: group nearby keypoints
    visited = set()
    clusters = []
    
    def get_neighbors(kp_idx: int) -> List[int]:
        """Find indices of neighboring keypoints."""
        kp = keypoints[kp_idx]
        neighbors = []
        for i, other in enumerate(keypoints):
            if i == kp_idx:
                continue
            dist = np.sqrt((kp.x - other.x)**2 + (kp.y - other.y)**2)
            if dist <= cluster_radius:
                neighbors.append(i)
        return neighbors
    
    for i, kp in enumerate(keypoints):
        if i in visited:
            continue
        
        # Start new cluster with BFS
        cluster = [i]
        visited.add(i)
        queue = [i]
        
        while queue:
            current = queue.pop(0)
            for neighbor_idx in get_neighbors(current):
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    cluster.append(neighbor_idx)
                    queue.append(neighbor_idx)
        
        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)
    
    # Convert clusters to bounding boxes
    regions = []
    for cluster in clusters:
        xs = [keypoints[i].x for i in cluster]
        ys = [keypoints[i].y for i in cluster]
        
        # Add padding
        padding = cluster_radius
        x1 = max(0, int(min(xs) - padding))
        y1 = max(0, int(min(ys) - padding))
        x2 = min(w, int(max(xs) + padding))
        y2 = min(h, int(max(ys) + padding))
        
        regions.append((x1, y1, x2 - x1, y2 - y1))
    
    return regions


def find_mismatched_regions(
    match_qualities: List[MatchQuality],
    all_keypoints1: List[Keypoint],
    all_keypoints2: List[Keypoint],
    image_shape: Tuple[int, ...],
    poor_match_threshold: float = 0.3,
    min_region_size: int = 3
) -> List[DiffRegion]:
    """
    Identify regions with poor feature matching.
    
    Args:
        match_qualities: Quality analysis for all matches
        all_keypoints1: All keypoints from image 1
        all_keypoints2: All keypoints from image 2
        image_shape: Shape of images
        poor_match_threshold: Score threshold for "poor" matches
        min_region_size: Minimum keypoints to form a region
    
    Returns:
        List of difference regions
    """
    if len(image_shape) >= 2:
        h, w = image_shape[0], image_shape[1]
    else:
        return []
    
    # Find poorly matched keypoints in image 1
    poorly_matched_kp1 = set()
    good_matches = []
    
    for mq in match_qualities:
        score = (mq.distance_score + mq.geometric_consistency) / 2
        if score < poor_match_threshold:
            poorly_matched_kp1.add(id(mq.match.kp1))
        else:
            good_matches.append(mq.match)
    
    # Find unmatched keypoints
    matched_ids1 = {id(mq.match.kp1) for mq in match_qualities}
    matched_ids2 = {id(mq.match.kp2) for mq in match_qualities}
    
    unmatched_kp1 = [kp for kp in all_keypoints1 if id(kp) not in matched_ids1]
    unmatched_kp2 = [kp for kp in all_keypoints2 if id(kp) not in matched_ids2]
    
    # Combine poorly matched and unmatched for region detection
    problematic_kp1 = unmatched_kp1 + [kp for kp in all_keypoints1 if id(kp) in poorly_matched_kp1]
    
    if not problematic_kp1:
        return []
    
    # Cluster problematic keypoints into regions
    region_boxes = cluster_keypoints_into_regions(
        problematic_kp1, 
        image_shape,
        cluster_radius=40.0,
        min_cluster_size=min_region_size
    )
    
    # Create DiffRegion objects
    regions = []
    for x, y, width, height in region_boxes:
        # Find keypoints in this region
        region_kp1 = [kp for kp in all_keypoints1 
                      if x <= kp.x < x + width and y <= kp.y < y + height]
        region_kp2 = [kp for kp in all_keypoints2 
                      if x <= kp.x < x + width and y <= kp.y < y + height]
        
        # Count unmatched in region
        unmatched_in_region = sum(1 for kp in region_kp1 if id(kp) not in matched_ids1)
        
        # Calculate avg distance for matches in region
        region_matches = [mq for mq in match_qualities
                         if x <= mq.match.kp1.x < x + width 
                         and y <= mq.match.kp1.y < y + height]
        
        if region_matches:
            avg_dist = np.mean([mq.match.distance for mq in region_matches])
        else:
            avg_dist = float('inf')
        
        # Determine severity
        unmatched_ratio = unmatched_in_region / len(region_kp1) if region_kp1 else 0
        
        if unmatched_ratio > 0.8:
            severity = DifferenceSeverity.CRITICAL
        elif unmatched_ratio > 0.5:
            severity = DifferenceSeverity.HIGH
        elif unmatched_ratio > 0.3:
            severity = DifferenceSeverity.MEDIUM
        elif unmatched_ratio > 0.1:
            severity = DifferenceSeverity.LOW
        else:
            severity = DifferenceSeverity.NONE
        
        # Confidence based on number of keypoints
        confidence = min(1.0, len(region_kp1) / 10.0)
        
        regions.append(DiffRegion(
            x=x, y=y, width=width, height=height,
            severity=severity,
            confidence=confidence,
            unmatched_count=unmatched_in_region,
            avg_match_distance=avg_dist,
            keypoints_img1=region_kp1,
            keypoints_img2=region_kp2
        ))
    
    return regions


def compute_difference_map(
    image1: np.ndarray,
    image2: np.ndarray,
    match_result: Optional[MatchingResult] = None,
    blur_sigma: float = 5.0
) -> np.ndarray:
    """
    Compute a pixel-level difference map based on feature mismatches.
    
    Creates a heatmap showing regions of significant difference.
    
    Args:
        image1: First image
        image2: Second image
        match_result: Pre-computed matching result (optional)
        blur_sigma: Gaussian blur sigma for smoothing the map
    
    Returns:
        Difference map as 2D array (0-1 range, higher = more different)
    """
    _validate_inputs(image1, image2, allow_different_dtypes=True)
    
    h, w = image1.shape[:2]
    
    # Get matches if not provided
    if match_result is None:
        match_result = match_features(image1, image2, backend='numpy')
    
    # Initialize difference map
    diff_map = np.zeros((h, w), dtype=np.float32)
    
    # Mark unmatched regions
    if match_result.matches:
        from sift import SIFT, SIFTConfig
        sift = SIFT(SIFTConfig())
        all_kp1 = sift.detect_and_compute(image1)
        all_kp2 = sift.detect_and_compute(image2)
        
        matched_ids1 = {id(m.kp1) for m in match_result.matches}
        matched_ids2 = {id(m.kp2) for m in match_result.matches}
        
        # Mark unmatched keypoints
        for kp in all_kp1:
            if id(kp) not in matched_ids1:
                y, x = int(kp.y), int(kp.x)
                if 0 <= y < h and 0 <= x < w:
                    # Draw circle around unmatched keypoint
                    radius = int(kp.scale * 3)
                    y_range = slice(max(0, y-radius), min(h, y+radius+1))
                    x_range = slice(max(0, x-radius), min(w, x+radius+1))
                    diff_map[y_range, x_range] = 1.0
        
        for kp in all_kp2:
            if id(kp) not in matched_ids2:
                y, x = int(kp.y), int(kp.x)
                if 0 <= y < h and 0 <= x < w:
                    radius = int(kp.scale * 3)
                    y_range = slice(max(0, y-radius), min(h, y+radius+1))
                    x_range = slice(max(0, x-radius), min(w, x+radius+1))
                    diff_map[y_range, x_range] = 1.0
    
    # Apply Gaussian blur to smooth
    if blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        diff_map = gaussian_filter(diff_map, sigma=blur_sigma)
    
    # Normalize to 0-1
    if diff_map.max() > 0:
        diff_map = diff_map / diff_map.max()
    
    return diff_map


class DiffRegionDetector:
    """
    Detect difference regions between two images using feature analysis.
    
    This class provides a configurable interface for detecting and analyzing
    regions with significant feature mismatches.
    
    Example:
        >>> detector = DiffRegionDetector(
        ...     sensitivity='medium',
        ...     min_region_size=5,
        ...     cluster_radius=40.0
        ... )
        >>> result = detector.detect(image1, image2)
        >>> for region in result.regions:
        ...     print(f"Change: {region}")
    """
    
    SENSITIVITY_SETTINGS = {
        'low': {
            'poor_match_threshold': 0.2,
            'min_region_size': 5,
            'cluster_radius': 50.0,
            'distance_threshold': 150.0
        },
        'medium': {
            'poor_match_threshold': 0.3,
            'min_region_size': 3,
            'cluster_radius': 40.0,
            'distance_threshold': 100.0
        },
        'high': {
            'poor_match_threshold': 0.4,
            'min_region_size': 2,
            'cluster_radius': 30.0,
            'distance_threshold': 75.0
        }
    }
    
    def __init__(
        self,
        sensitivity: str = 'medium',
        min_region_size: Optional[int] = None,
        cluster_radius: Optional[float] = None,
        poor_match_threshold: Optional[float] = None,
        distance_threshold: Optional[float] = None,
        backend: str = 'numpy'
    ):
        """
        Initialize detector with settings.
        
        Args:
            sensitivity: Preset sensitivity ('low', 'medium', 'high')
            min_region_size: Override minimum region size
            cluster_radius: Override cluster radius
            poor_match_threshold: Override poor match threshold
            distance_threshold: Override distance threshold
            backend: Feature detection backend ('numpy', 'opencv', 'auto')
        """
        if sensitivity not in self.SENSITIVITY_SETTINGS:
            raise ValueError(f"Sensitivity must be one of: {list(self.SENSITIVITY_SETTINGS.keys())}")
        
        settings = self.SENSITIVITY_SETTINGS[sensitivity].copy()
        
        # Apply overrides
        if min_region_size is not None:
            settings['min_region_size'] = min_region_size
        if cluster_radius is not None:
            settings['cluster_radius'] = cluster_radius
        if poor_match_threshold is not None:
            settings['poor_match_threshold'] = poor_match_threshold
        if distance_threshold is not None:
            settings['distance_threshold'] = distance_threshold
        
        self.settings = settings
        self.backend = backend
    
    def detect(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        return_diff_map: bool = False
    ) -> RegionDiffResult | Tuple[RegionDiffResult, np.ndarray]:
        """
        Detect difference regions between two images.
        
        Args:
            image1: First image
            image2: Second image
            return_diff_map: If True, also return the difference map
        
        Returns:
            RegionDiffResult, optionally with difference map
        """
        _validate_inputs(image1, image2, allow_different_dtypes=True)
        
        h, w = image1.shape[:2]
        
        # Match features
        match_result = match_features(
            image1, image2, 
            backend=self.backend,
            method=MatchingMethod.RATIO
        )
        
        # Get all keypoints
        from sift import SIFT, SIFTConfig
        sift = SIFT(SIFTConfig())
        all_kp1 = sift.detect_and_compute(image1)
        all_kp2 = sift.detect_and_compute(image2)
        
        # Analyze match quality
        qualities = analyze_match_quality(
            match_result.matches,
            all_kp1,
            all_kp2,
            distance_threshold=self.settings['distance_threshold']
        )
        
        # Find mismatched regions
        regions = find_mismatched_regions(
            qualities,
            all_kp1,
            all_kp2,
            image1.shape,
            poor_match_threshold=self.settings['poor_match_threshold'],
            min_region_size=self.settings['min_region_size']
        )
        
        # Calculate global metrics
        total_diff_area = sum(r.area for r in regions)
        diff_coverage = total_diff_area / (h * w) if h * w > 0 else 0
        
        # Determine global severity
        if regions:
            max_severity = max(r.severity.value for r in regions)
            global_severity = DifferenceSeverity(max_severity)
        else:
            global_severity = DifferenceSeverity.NONE
        
        result = RegionDiffResult(
            regions=regions,
            n_regions=len(regions),
            total_diff_area=total_diff_area,
            diff_coverage=diff_coverage,
            global_severity=global_severity,
            match_result=match_result
        )
        
        if return_diff_map:
            diff_map = compute_difference_map(image1, image2, match_result)
            return result, diff_map
        
        return result


def detect_difference_regions(
    image1: np.ndarray,
    image2: np.ndarray,
    sensitivity: str = 'medium',
    return_diff_map: bool = False,
    backend: str = 'numpy'
) -> RegionDiffResult | Tuple[RegionDiffResult, np.ndarray]:
    """
    Convenience function to detect difference regions between two images.
    
    Args:
        image1: First image
        image2: Second image  
        sensitivity: Detection sensitivity ('low', 'medium', 'high')
        return_diff_map: If True, also return difference heatmap
        backend: Feature detection backend
    
    Returns:
        RegionDiffResult with detected regions, optionally with diff map
    
    Example:
        >>> result = detect_difference_regions(img1, img2, sensitivity='high')
        >>> print(f"Found {result.n_regions} difference regions")
        >>> print(f"Coverage: {result.diff_coverage:.1%}")
        >>> for region in result.regions:
        ...     print(f"  Region at ({region.x}, {region.y}): {region.severity.name}")
    """
    detector = DiffRegionDetector(sensitivity=sensitivity, backend=backend)
    return detector.detect(image1, image2, return_diff_map=return_diff_map)
