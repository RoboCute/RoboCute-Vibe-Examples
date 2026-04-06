"""
Quick validation tests for diff_regions module.
Run without pytest: python test_diff_regions.py
"""

import sys
import numpy as np

# Check dependencies
try:
    from diff_regions import (
        detect_difference_regions,
        DiffRegionDetector,
        compute_difference_map,
        DiffRegion,
        DifferenceSeverity,
        RegionDiffResult,
        analyze_match_quality,
        find_mismatched_regions,
        cluster_keypoints_into_regions,
    )
    from sift import SIFT, SIFTConfig
    from feature_matching import match_features
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_test_image(size=128, pattern='checkerboard', offset=(0, 0)):
    """Create test image with features."""
    img = np.zeros((size, size), dtype=np.float32)
    block = size // 8
    
    if pattern == 'checkerboard':
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    y1 = max(0, i*block + offset[1])
                    y2 = min(size, (i+1)*block + offset[1])
                    x1 = max(0, j*block + offset[0])
                    x2 = min(size, (j+1)*block + offset[0])
                    if y2 > y1 and x2 > x1:
                        img[y1:y2, x1:x2] = 1.0
    elif pattern == 'gradient':
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        img = (xx + yy) / 2
    elif pattern == 'modified':
        # Similar to checkerboard but with a modification
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    img[i*block:(i+1)*block, j*block:(j+1)*block] = 1.0
        # Add a modification in the center
        img[size//3:2*size//3, size//3:2*size//3] = 0.5
    
    return img


def test_cluster_keypoints():
    """Test keypoint clustering."""
    print("\n1. Testing keypoint clustering...")
    
    # Create keypoints in two clusters
    sift = SIFT(SIFTConfig())
    img = create_test_image(128, 'checkerboard')
    keypoints = sift.detect_and_compute(img)
    
    regions = cluster_keypoints_into_regions(
        keypoints, 
        img.shape,
        cluster_radius=50.0,
        min_cluster_size=3
    )
    
    assert isinstance(regions, list)
    print(f"   Found {len(regions)} clusters")
    for i, (x, y, w, h) in enumerate(regions[:3]):
        print(f"   Region {i}: ({x}, {y}, {w}, {h})")
    print("   [OK] Clustering works")


def test_analyze_match_quality():
    """Test match quality analysis."""
    print("\n2. Testing match quality analysis...")
    
    img1 = create_test_image(128, 'checkerboard')
    img2 = img1.copy()  # Identical
    
    match_result = match_features(img1, img2)
    
    sift = SIFT(SIFTConfig())
    all_kp1 = sift.detect_and_compute(img1)
    all_kp2 = sift.detect_and_compute(img2)
    
    qualities = analyze_match_quality(
        match_result.matches,
        all_kp1,
        all_kp2
    )
    
    assert len(qualities) == len(match_result.matches)
    
    good_matches = sum(1 for q in qualities if q.is_good)
    print(f"   Total matches: {len(qualities)}")
    print(f"   Good matches: {good_matches}")
    print("   [OK] Quality analysis works")


def test_detect_difference_regions():
    """Test difference region detection."""
    print("\n3. Testing difference region detection...")
    
    # Identical images should have no differences
    img1 = create_test_image(128, 'checkerboard')
    img2 = img1.copy()
    
    result = detect_difference_regions(img1, img2, sensitivity='medium')
    
    assert isinstance(result, RegionDiffResult)
    print(f"   Identical images: {result.n_regions} regions, coverage={result.diff_coverage:.1%}")
    
    # Different images should have some differences
    img3 = create_test_image(128, 'modified')
    result2 = detect_difference_regions(img1, img3, sensitivity='medium')
    
    print(f"   Different images: {result2.n_regions} regions, coverage={result2.diff_coverage:.1%}")
    print(f"   Global severity: {result2.global_severity.name}")
    
    for i, region in enumerate(result2.regions[:3]):
        print(f"   Region {i}: ({region.x}, {region.y}) {region.width}x{region.height}, "
              f"severity={region.severity.name}, conf={region.confidence:.2f}")
    
    print("   [OK] Difference detection works")


def test_diff_region_detector():
    """Test DiffRegionDetector class."""
    print("\n4. Testing DiffRegionDetector...")
    
    detector = DiffRegionDetector(sensitivity='high')
    
    img1 = create_test_image(128, 'checkerboard')
    img2 = create_test_image(128, 'modified')
    
    result = detector.detect(img1, img2)
    
    assert isinstance(result, RegionDiffResult)
    print(f"   High sensitivity: {result.n_regions} regions")
    
    # Test with different sensitivity
    detector_low = DiffRegionDetector(sensitivity='low')
    result_low = detector_low.detect(img1, img2)
    print(f"   Low sensitivity: {result_low.n_regions} regions")
    
    print("   [OK] Detector class works")


def test_compute_difference_map():
    """Test difference map computation."""
    print("\n5. Testing difference map...")
    
    img1 = create_test_image(128, 'checkerboard')
    img2 = create_test_image(128, 'modified')
    
    diff_map = compute_difference_map(img1, img2)
    
    assert diff_map.shape == (128, 128)
    assert diff_map.min() >= 0 and diff_map.max() <= 1
    
    print(f"   Diff map shape: {diff_map.shape}")
    print(f"   Diff map range: [{diff_map.min():.3f}, {diff_map.max():.3f}]")
    print(f"   Mean difference: {diff_map.mean():.3f}")
    
    # Identical images should have near-zero diff map
    diff_map_same = compute_difference_map(img1, img1)
    print(f"   Identical images mean diff: {diff_map_same.mean():.3f}")
    
    print("   [OK] Difference map works")


def test_with_return_diff_map():
    """Test returning both result and diff map."""
    print("\n6. Testing return with diff map...")
    
    img1 = create_test_image(128, 'checkerboard')
    img2 = create_test_image(128, 'modified')
    
    result, diff_map = detect_difference_regions(
        img1, img2, 
        sensitivity='medium',
        return_diff_map=True
    )
    
    assert isinstance(result, RegionDiffResult)
    assert isinstance(diff_map, np.ndarray)
    assert diff_map.shape == (128, 128)
    
    print(f"   Got result with {result.n_regions} regions and diff map")
    print("   [OK] Return with diff map works")


def test_region_properties():
    """Test DiffRegion properties."""
    print("\n7. Testing region properties...")
    
    img1 = create_test_image(128, 'modified')
    img2 = create_test_image(128, 'checkerboard')
    
    result = detect_difference_regions(img1, img2, sensitivity='medium')
    
    for region in result.regions:
        # Test properties
        center = region.center
        area = region.area
        assert area == region.width * region.height
        assert isinstance(center, tuple) and len(center) == 2
        assert 0 <= region.confidence <= 1
        assert isinstance(region.severity, DifferenceSeverity)
    
    print(f"   Tested {len(result.regions)} regions")
    print("   [OK] Region properties work")


def test_filter_by_severity():
    """Test filtering regions by severity."""
    print("\n8. Testing severity filtering...")
    
    img1 = create_test_image(128, 'checkerboard')
    img2 = create_test_image(128, 'modified')
    
    result = detect_difference_regions(img1, img2, sensitivity='high')
    
    high_severity = result.get_regions_by_severity(DifferenceSeverity.HIGH)
    medium_severity = result.get_regions_by_severity(DifferenceSeverity.MEDIUM)
    
    print(f"   Total regions: {result.n_regions}")
    print(f"   High+ severity: {len(high_severity)}")
    print(f"   Medium+ severity: {len(medium_severity)}")
    
    print("   [OK] Severity filtering works")


def main():
    print("Testing diff_regions module...")
    print("=" * 60)
    
    try:
        test_cluster_keypoints()
        test_analyze_match_quality()
        test_detect_difference_regions()
        test_diff_region_detector()
        test_compute_difference_map()
        test_with_return_diff_map()
        test_region_properties()
        test_filter_by_severity()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] All tests passed!")
        return 0
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
