"""
Main comparison function for image difference detection.

This module provides a high-level API for comparing two images and returning
comprehensive difference metrics including match ratios, difference masks,
region-based changes, and statistical summaries.

Usage:
    >>> from compare import compare_images, CompareResult
    >>> 
    >>> # Compare two images
    >>> result = compare_images(image1, image2)
    >>> 
    >>> # Access metrics
    >>> print(f"Similarity: {result.similarity_score:.2%}")
    >>> print(f"Match ratio: {result.match_ratio:.2%}")
    >>> print(f"Changed regions: {result.n_regions}")
    >>> 
    >>> # Visualize differences
    >>> diff_mask = result.diff_mask  # Binary or graded mask
    >>> diff_map = result.diff_map    # Heatmap of changes
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Import from existing modules
from sift import (
    SIFT, SIFTConfig, Keypoint, Match, 
    SIFTInputError, _validate_inputs, create_sift
)

# Default SIFT config optimized for comparison (more permissive)
DEFAULT_SIFT_CONFIG = SIFTConfig(
    contrast_threshold=0.01,  # Lower threshold to detect more features
    edge_threshold=20.0,      # Higher threshold to keep more edge features
)
from feature_matching import (
    match_features,
    compute_descriptors,
    get_matched_points,
    MatchingResult,
    MatchingMethod
)
from diff_regions import (
    detect_difference_regions,
    DiffRegionDetector,
    compute_difference_map,
    DiffRegion,
    DifferenceSeverity,
    RegionDiffResult
)

__all__ = [
    # Main API
    'compare_images',
    'batch_compare',
    'CompareResult',
    'ComparisonLevel',
    
    # Utilities
    'compute_similarity_score',
    'create_diff_visualization',
    'summarize_differences',
]


class ComparisonLevel(Enum):
    """Level of comparison detail."""
    BASIC = 'basic'       # Basic metrics only (fast)
    STANDARD = 'standard' # Standard metrics + regions
    FULL = 'full'         # Full analysis including diff maps


@dataclass
class CompareResult:
    """
    Complete comparison result between two images.
    
    Attributes:
        similarity_score: Overall similarity (0-1, higher = more similar)
        match_ratio: Ratio of matched features to total features
        n_matches: Number of feature matches
        n_keypoints1: Number of keypoints in image 1
        n_keypoints2: Number of keypoints in image 2
        avg_match_distance: Average descriptor distance for matches
        
        n_regions: Number of difference regions detected
        regions: List of difference regions
        diff_coverage: Percentage of image covered by differences
        max_severity: Maximum severity of detected differences
        
        diff_mask: Binary mask of changed pixels
        diff_map: Graded difference heatmap (0-1)
        diff_overlay: RGB visualization overlay
        
        is_similar: Whether images are considered similar
        is_identical: Whether images are nearly identical
        confidence: Confidence in the comparison result
        
        metrics: Dictionary of all raw metrics
    """
    # Core metrics
    similarity_score: float
    match_ratio: float
    n_matches: int
    n_keypoints1: int
    n_keypoints2: int
    avg_match_distance: float
    
    # Region analysis
    n_regions: int
    regions: List[DiffRegion]
    diff_coverage: float
    max_severity: DifferenceSeverity
    
    # Masks and visualizations
    diff_mask: Optional[np.ndarray] = None
    diff_map: Optional[np.ndarray] = None
    diff_overlay: Optional[np.ndarray] = None
    
    # Classification
    is_similar: bool = False
    is_identical: bool = False
    confidence: float = 0.0
    
    # Raw data
    metrics: Dict[str, Any] = field(default_factory=dict)
    match_result: Optional[MatchingResult] = None
    region_result: Optional[RegionDiffResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary (excluding arrays)."""
        return {
            'similarity_score': float(self.similarity_score),
            'match_ratio': float(self.match_ratio),
            'n_matches': int(self.n_matches),
            'n_keypoints1': int(self.n_keypoints1),
            'n_keypoints2': int(self.n_keypoints2),
            'avg_match_distance': float(self.avg_match_distance),
            'n_regions': int(self.n_regions),
            'diff_coverage': float(self.diff_coverage),
            'max_severity': self.max_severity.name,
            'is_similar': bool(self.is_similar),
            'is_identical': bool(self.is_identical),
            'confidence': float(self.confidence),
            'has_diff_mask': self.diff_mask is not None,
            'has_diff_map': self.diff_map is not None,
        }
    
    def get_changed_regions(self, min_severity: DifferenceSeverity = DifferenceSeverity.LOW) -> List[DiffRegion]:
        """Get regions with at least the specified severity."""
        return [r for r in self.regions if r.severity.value >= min_severity.value]
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Image Comparison Result:",
            f"  Similarity: {self.similarity_score:.1%}",
            f"  Feature match ratio: {self.match_ratio:.1%} ({self.n_matches}/{self.n_keypoints1})",
            f"  Difference regions: {self.n_regions}",
            f"  Coverage: {self.diff_coverage:.1%}",
            f"  Max severity: {self.max_severity.name}",
            f"  Classification: {'Identical' if self.is_identical else 'Similar' if self.is_similar else 'Different'}",
        ]
        return '\n'.join(lines)


def compute_similarity_score(
    match_ratio: float,
    avg_distance: float,
    diff_coverage: float,
    n_matches: int,
    min_matches: int = 10
) -> float:
    """
    Compute overall similarity score from various metrics.
    
    The score combines:
    - Feature match ratio (how many features matched)
    - Match quality (descriptor distance)
    - Spatial coverage (how much of image is different)
    - Match confidence (number of matches)
    
    Args:
        match_ratio: Ratio of matched features (0-1)
        avg_distance: Average match distance (lower is better)
        diff_coverage: Percentage of image with differences (0-1)
        n_matches: Total number of matches
        min_matches: Minimum matches for confident result
    
    Returns:
        Similarity score (0-1, higher = more similar)
    """
    # Normalize distance to 0-1 (assuming max expected distance of 300)
    distance_score = max(0.0, 1.0 - avg_distance / 300.0)
    
    # Coverage score (inverse of coverage)
    coverage_score = 1.0 - diff_coverage
    
    # Confidence based on number of matches
    confidence = min(1.0, n_matches / min_matches)
    
    # Weighted combination
    # Feature matching is most important, then coverage, then distance
    similarity = (
        0.4 * match_ratio +
        0.3 * coverage_score +
        0.2 * distance_score +
        0.1 * confidence
    )
    
    return float(np.clip(similarity, 0.0, 1.0))


def create_diff_visualization(
    image1: np.ndarray,
    result: CompareResult,
    mask_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create RGB visualization of differences overlaid on image.
    
    Args:
        image1: Base image
        result: Comparison result with diff_mask
        mask_color: RGB color for difference highlighting
        alpha: Transparency of overlay
    
    Returns:
        RGB image with difference overlay
    """
    # Normalize image to uint8 RGB
    if image1.ndim == 2:
        img_rgb = np.stack([image1] * 3, axis=-1)
    else:
        img_rgb = image1.copy()
    
    # Normalize to 0-255
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)
    else:
        img_rgb = img_rgb.astype(np.uint8)
    
    # Ensure RGB
    if img_rgb.shape[-1] != 3:
        img_rgb = img_rgb[..., :3]
    
    # Create overlay
    if result.diff_mask is not None:
        overlay = img_rgb.copy()
        mask = result.diff_mask > 0
        overlay[mask] = mask_color
        
        # Blend
        img_rgb = (img_rgb * (1 - alpha) + overlay * alpha).astype(np.uint8)
    
    # Draw region boxes
    for region in result.regions:
        x1, y1 = region.x, region.y
        x2, y2 = x1 + region.width, y1 + region.height
        
        # Color based on severity
        if region.severity == DifferenceSeverity.CRITICAL:
            color = (0, 0, 255)  # Red
        elif region.severity == DifferenceSeverity.HIGH:
            color = (0, 128, 255)  # Orange
        elif region.severity == DifferenceSeverity.MEDIUM:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 0)  # Green
        
        # Draw rectangle
        img_rgb[y1:y2, x1:x1+2] = color  # Left
        img_rgb[y1:y2, x2-2:x2] = color  # Right
        img_rgb[y1:y1+2, x1:x2] = color  # Top
        img_rgb[y2-2:y2, x1:x2] = color  # Bottom
    
    return img_rgb


def compare_images(
    image1: np.ndarray,
    image2: np.ndarray,
    level: ComparisonLevel | str = ComparisonLevel.STANDARD,
    similarity_threshold: float = 0.7,
    identical_threshold: float = 0.95,
    backend: str = 'numpy',
    sensitivity: str = 'medium',
    return_visualization: bool = False,
    **kwargs
) -> CompareResult:
    """
    Main comparison function for two images.
    
    This is the primary API for image comparison, providing comprehensive
    metrics about differences between two images.
    
    Args:
        image1: First image (H, W) or (H, W, C)
        image2: Second image (H, W) or (H, W, C)
        level: Detail level ('basic', 'standard', 'full')
        similarity_threshold: Threshold for considering images similar
        identical_threshold: Threshold for considering images identical
        backend: Feature detection backend ('numpy', 'opencv', 'auto')
        sensitivity: Region detection sensitivity ('low', 'medium', 'high')
        return_visualization: If True, include diff_overlay in result
        **kwargs: Additional parameters for fine-tuning
    
    Returns:
        CompareResult with all comparison metrics
    
    Example:
        >>> result = compare_images(img1, img2)
        >>> print(result.get_summary())
        >>> 
        >>> if not result.is_similar:
        ...     print(f"Found {result.n_regions} difference regions")
        ...     for region in result.regions:
        ...         print(f"  - {region}")
    """
    # Validate inputs - catch TypeError and convert to SIFTInputError
    try:
        _validate_inputs(image1, image2, allow_different_dtypes=True)
    except TypeError as e:
        raise SIFTInputError(str(e))
    
    # Convert level string to enum
    if isinstance(level, str):
        level = ComparisonLevel(level)
    
    h, w = image1.shape[:2]
    
    # Step 1: Feature matching
    sift_kwargs = {'backend': backend, 'method': MatchingMethod.RATIO}
    if 'ratio_threshold' in kwargs:
        sift_kwargs['ratio_threshold'] = kwargs['ratio_threshold']
    if 'distance_threshold' in kwargs:
        sift_kwargs['distance_threshold'] = kwargs['distance_threshold']
    # Pass SIFT config if provided
    if 'config' in kwargs:
        sift_kwargs['config'] = kwargs['config']
    else:
        sift_kwargs['config'] = DEFAULT_SIFT_CONFIG
    
    match_result = match_features(image1, image2, **sift_kwargs)
    
    # Extract keypoint counts using same config
    sift = SIFT(kwargs.get('config', DEFAULT_SIFT_CONFIG))
    all_kp1 = sift.detect_and_compute(image1)
    all_kp2 = sift.detect_and_compute(image2)
    n_kp1 = len([kp for kp in all_kp1 if kp.descriptor is not None])
    n_kp2 = len([kp for kp in all_kp2 if kp.descriptor is not None])
    
    # Step 2: Region detection (if standard or full)
    if level in (ComparisonLevel.STANDARD, ComparisonLevel.FULL):
        if level == ComparisonLevel.FULL:
            region_result, diff_map = detect_difference_regions(
                image1, image2,
                sensitivity=sensitivity,
                return_diff_map=True,
                backend=backend
            )
        else:
            region_result = detect_difference_regions(
                image1, image2,
                sensitivity=sensitivity,
                return_diff_map=False,
                backend=backend
            )
            diff_map = None
    else:
        region_result = None
        diff_map = None
    
    # Step 3: Compute masks and visualizations (if full)
    diff_mask = None
    diff_overlay = None
    
    if level == ComparisonLevel.FULL:
        # Create binary mask from regions
        diff_mask = np.zeros((h, w), dtype=np.uint8)
        if region_result:
            for region in region_result.regions:
                x1, y1 = max(0, region.x), max(0, region.y)
                x2, y2 = min(w, region.x + region.width), min(h, region.y + region.height)
                diff_mask[y1:y2, x1:x2] = 1
        
        # Create visualization if requested
        if return_visualization:
            # Temp result for visualization function
            temp_result = CompareResult(
                similarity_score=0.0,
                match_ratio=match_result.match_ratio,
                n_matches=match_result.n_matches,
                n_keypoints1=n_kp1,
                n_keypoints2=n_kp2,
                avg_match_distance=match_result.avg_distance,
                n_regions=region_result.n_regions if region_result else 0,
                regions=region_result.regions if region_result else [],
                diff_coverage=region_result.diff_coverage if region_result else 0.0,
                max_severity=region_result.global_severity if region_result else DifferenceSeverity.NONE,
                diff_mask=diff_mask,
            )
            diff_overlay = create_diff_visualization(image1, temp_result)
    
    # Step 4: Compute similarity and classification
    diff_coverage = region_result.diff_coverage if region_result else 0.0
    max_severity = region_result.global_severity if region_result else DifferenceSeverity.NONE
    
    similarity = compute_similarity_score(
        match_ratio=match_result.match_ratio,
        avg_distance=match_result.avg_distance,
        diff_coverage=diff_coverage,
        n_matches=match_result.n_matches
    )
    
    is_similar = similarity >= similarity_threshold
    is_identical = similarity >= identical_threshold and match_result.n_matches > 10
    
    # Confidence based on number of features and matches
    confidence = min(1.0, (n_kp1 + n_kp2) / 100.0) * min(1.0, match_result.n_matches / 20.0)
    
    # Build metrics dictionary
    metrics = {
        'image_shape': (h, w),
        'level': level.value,
        'similarity_threshold': similarity_threshold,
        'identical_threshold': identical_threshold,
        'match_details': {
            'n_keypoints1': n_kp1,
            'n_keypoints2': n_kp2,
            'n_matches': match_result.n_matches,
            'match_ratio': match_result.match_ratio,
            'avg_distance': match_result.avg_distance,
            'inlier_ratio': match_result.inlier_ratio,
        },
        'region_details': {
            'n_regions': region_result.n_regions if region_result else 0,
            'diff_coverage': diff_coverage,
            'total_diff_area': region_result.total_diff_area if region_result else 0,
            'max_severity': max_severity.name,
        } if region_result else None,
    }
    
    return CompareResult(
        similarity_score=similarity,
        match_ratio=match_result.match_ratio,
        n_matches=match_result.n_matches,
        n_keypoints1=n_kp1,
        n_keypoints2=n_kp2,
        avg_match_distance=match_result.avg_distance,
        n_regions=region_result.n_regions if region_result else 0,
        regions=region_result.regions if region_result else [],
        diff_coverage=diff_coverage,
        max_severity=max_severity,
        diff_mask=diff_mask,
        diff_map=diff_map,
        diff_overlay=diff_overlay,
        is_similar=is_similar,
        is_identical=is_identical,
        confidence=confidence,
        metrics=metrics,
        match_result=match_result,
        region_result=region_result
    )


def batch_compare(
    reference_image: np.ndarray,
    test_images: List[np.ndarray],
    **kwargs
) -> List[CompareResult]:
    """
    Compare a reference image against multiple test images.
    
    Args:
        reference_image: The reference/base image
        test_images: List of images to compare against reference
        **kwargs: Arguments passed to compare_images()
    
    Returns:
        List of CompareResult objects (one per test image)
    
    Example:
        >>> results = batch_compare(ref_img, [img1, img2, img3])
        >>> for i, result in enumerate(results):
        ...     print(f"Image {i}: similarity={result.similarity_score:.2%}")
    """
    results = []
    for i, test_img in enumerate(test_images):
        try:
            result = compare_images(reference_image, test_img, **kwargs)
            results.append(result)
        except Exception as e:
            # Create error result
            result = CompareResult(
                similarity_score=0.0,
                match_ratio=0.0,
                n_matches=0,
                n_keypoints1=0,
                n_keypoints2=0,
                avg_match_distance=float('inf'),
                n_regions=0,
                regions=[],
                diff_coverage=0.0,
                max_severity=DifferenceSeverity.NONE,
                is_similar=False,
                is_identical=False,
                confidence=0.0,
                metrics={'error': str(e)}
            )
            results.append(result)
    
    return results


def summarize_differences(results: List[CompareResult]) -> Dict[str, Any]:
    """
    Summarize multiple comparison results.
    
    Args:
        results: List of CompareResult objects
    
    Returns:
        Dictionary with aggregate statistics
    """
    if not results:
        return {'n_comparisons': 0}
    
    similarities = [r.similarity_score for r in results]
    match_ratios = [r.match_ratio for r in results]
    coverages = [r.diff_coverage for r in results]
    
    similar_count = sum(1 for r in results if r.is_similar)
    identical_count = sum(1 for r in results if r.is_identical)
    
    total_regions = sum(r.n_regions for r in results)
    
    # Severity distribution
    severity_counts = {s.name: 0 for s in DifferenceSeverity}
    for r in results:
        severity_counts[r.max_severity.name] += 1
    
    return {
        'n_comparisons': len(results),
        'similarity': {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
        },
        'match_ratio': {
            'mean': float(np.mean(match_ratios)),
            'std': float(np.std(match_ratios)),
        },
        'diff_coverage': {
            'mean': float(np.mean(coverages)),
            'std': float(np.std(coverages)),
        },
        'classification': {
            'identical': identical_count,
            'similar': similar_count - identical_count,
            'different': len(results) - similar_count,
        },
        'total_regions': total_regions,
        'severity_distribution': severity_counts,
    }
