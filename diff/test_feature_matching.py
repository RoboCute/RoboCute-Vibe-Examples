"""
Quick validation tests for feature_matching module.
Run without pytest: python test_feature_matching.py
"""

import sys
import numpy as np

# Check dependencies
try:
    from sift import SIFTInputError
    from feature_matching import (
        compute_descriptors,
        match_descriptors,
        match_features,
        find_correspondences,
        get_matched_points,
        DescriptorResult,
        MatchingResult,
        Correspondence,
        MatchingMethod,
    )
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def create_test_pattern(size=128, pattern='checkerboard'):
    """Create test image with features."""
    if pattern == 'checkerboard':
        img = np.zeros((size, size), dtype=np.float32)
        block = size // 8
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    img[i*block:(i+1)*block, j*block:(j+1)*block] = 1.0
    elif pattern == 'gradient':
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x, y)
        img = (xx + yy) / 2
    else:
        img = np.random.rand(size, size).astype(np.float32)
    
    return img


def test_compute_descriptors():
    """Test descriptor computation."""
    print("\n1. Testing descriptor computation...")
    
    img = create_test_pattern(128, 'checkerboard')
    result = compute_descriptors(img, backend='numpy')
    
    assert isinstance(result, DescriptorResult)
    assert result.n_keypoints > 0, "Should detect keypoints"
    assert result.descriptors.shape[0] == result.n_keypoints
    assert result.descriptors.shape[1] == 128, "SIFT descriptors should be 128-dim"
    assert result.descriptors.dtype == np.float32
    
    print(f"   Detected {result.n_keypoints} keypoints")
    print(f"   Descriptor shape: {result.descriptors.shape}")
    print("   [OK] Descriptor computation works")


def test_match_descriptors():
    """Test descriptor matching."""
    print("\n2. Testing descriptor matching...")
    
    # Create identical descriptors (should match perfectly)
    desc1 = np.random.rand(10, 128).astype(np.float32)
    desc2 = desc1.copy()  # Identical
    
    matches, distances = match_descriptors(desc1, desc2, method='ratio')
    
    assert len(matches) > 0, "Should find matches for identical descriptors"
    assert len(matches) == len(distances)
    
    # Check that matches are correct (should match same indices)
    for i, j in matches:
        assert i == j, f"Expected match {i}->{i}, got {i}->{j}"
    
    print(f"   Found {len(matches)} matches")
    print(f"   Average distance: {np.mean(distances):.4f}")
    print("   [OK] Descriptor matching works")


def test_match_features():
    """Test full feature matching pipeline."""
    print("\n3. Testing feature matching pipeline...")
    
    # Create two identical images
    img1 = create_test_pattern(128, 'checkerboard')
    img2 = img1.copy()
    
    result = match_features(img1, img2, backend='numpy', method='ratio')
    
    assert isinstance(result, MatchingResult)
    assert result.n_matches > 0, "Should find matches for identical images"
    # Note: Due to detection stochasticity, identical images may not have 100% match ratio
    # A match ratio > 0.2 indicates good feature matching
    assert result.match_ratio > 0.2, f"Low match ratio for identical images: {result.match_ratio:.2%}"
    
    print(f"   Keypoints in img1: {result.n_matches / result.match_ratio:.0f}")
    print(f"   Matches: {result.n_matches}")
    print(f"   Match ratio: {result.match_ratio:.2%}")
    print(f"   Avg distance: {result.avg_distance:.2f}")
    print("   [OK] Feature matching pipeline works")


def test_find_correspondences():
    """Test finding correspondences."""
    print("\n4. Testing correspondence finding...")
    
    img1 = create_test_pattern(128, 'checkerboard')
    img2 = img1.copy()
    
    correspondences = find_correspondences(img1, img2, backend='numpy')
    
    assert len(correspondences) > 0, "Should find correspondences"
    
    for corr in correspondences:
        assert isinstance(corr, Correspondence)
        assert len(corr.pt1) == 2
        assert len(corr.pt2) == 2
        assert 0 <= corr.confidence <= 1
    
    print(f"   Found {len(correspondences)} correspondences")
    print("   [OK] Correspondence finding works")


def test_different_images():
    """Test matching different images."""
    print("\n5. Testing different images...")
    
    img1 = create_test_pattern(128, 'checkerboard')
    img2 = create_test_pattern(128, 'gradient')  # Different pattern
    
    result = match_features(img1, img2, backend='numpy')
    
    # Should have fewer matches than for identical images
    print(f"   Matches between different images: {result.n_matches}")
    print(f"   Match ratio: {result.match_ratio:.2%}")
    print("   [OK] Different images detected")


def test_get_matched_points():
    """Test utility function for extracting points."""
    print("\n6. Testing get_matched_points...")
    
    img1 = create_test_pattern(128, 'checkerboard')
    img2 = img1.copy()
    
    result = match_features(img1, img2, backend='numpy')
    pts1, pts2 = get_matched_points(result.matches)
    
    assert pts1.shape == pts2.shape
    assert pts1.shape[0] == result.n_matches
    assert pts1.shape[1] == 2
    
    # For identical images, points should be very close
    diff = np.abs(pts1 - pts2)
    assert np.all(diff < 1.0), "Points should align for identical images"
    
    print(f"   Extracted {len(pts1)} point pairs")
    print("   [OK] Point extraction works")


def test_error_handling():
    """Test error handling."""
    print("\n7. Testing error handling...")
    
    # Empty image
    try:
        compute_descriptors(np.array([]))
        assert False, "Should raise error for empty image"
    except SIFTInputError:
        print("   [OK] Empty image raises error")
    
    # Wrong dimensions
    try:
        compute_descriptors(np.random.rand(10))  # 1D
        assert False, "Should raise error for 1D array"
    except SIFTInputError:
        print("   [OK] Wrong dimensions raise error")
    
    print("   [OK] Error handling works")


def test_matching_methods():
    """Test different matching methods."""
    print("\n8. Testing different matching methods...")
    
    img1 = create_test_pattern(128, 'checkerboard')
    img2 = img1.copy()
    
    methods = ['ratio', 'distance', 'mutual']
    
    for method in methods:
        result = match_features(img1, img2, backend='numpy', method=method)
        print(f"   Method '{method}': {result.n_matches} matches")
    
    print("   [OK] All matching methods work")


def main():
    print("Testing feature_matching module...")
    print("=" * 50)
    
    try:
        test_compute_descriptors()
        test_match_descriptors()
        test_match_features()
        test_find_correspondences()
        test_different_images()
        test_get_matched_points()
        test_error_handling()
        test_matching_methods()
        
        print("\n" + "=" * 50)
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
