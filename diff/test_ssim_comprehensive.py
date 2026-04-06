"""
Comprehensive Jumbo Test Suite for SSIM (Structural Similarity Index) Implementation.

This module contains exhaustive tests to verify the correctness of the SSIM
implementation with various input scenarios including:
- Identical arrays (should return 1.0)
- Completely different arrays (should return low value)
- Slightly modified arrays (should detect similarity correctly)
- Edge cases and boundary conditions
- Multi-channel images
- Different data types

Usage:
    python test_ssim_comprehensive.py          # Run comprehensive tests
    pytest test_ssim_comprehensive.py          # Run with pytest
"""

import numpy as np
import sys
from ssim import compute_ssim, compute_ssim_map, compare_images, SSIMInputError

# Try to import pytest for full test suite
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


# =============================================================================
# Test Constants
# =============================================================================
TEST_IMAGE_SIZE = 64
TEST_TOLERANCE = 1e-10
SSIM_HIGH_THRESHOLD = 0.9
SSIM_LOW_THRESHOLD = 0.5


# =============================================================================
# Helper Functions
# =============================================================================
def create_test_image(size=TEST_IMAGE_SIZE, channels=1, dtype=np.float64, seed=None):
    """Create a test image with given specifications."""
    if seed is not None:
        np.random.seed(seed)
    
    if channels == 1:
        shape = (size, size)
    else:
        shape = (size, size, channels)
    
    if dtype in (np.uint8, np.uint16):
        max_val = 255 if dtype == np.uint8 else 65535
        return (np.random.rand(*shape) * max_val).astype(dtype)
    else:
        return np.random.rand(*shape).astype(dtype)


def add_noise(image, noise_level=0.01, seed=None):
    """Add Gaussian noise to an image."""
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.randn(*image.shape) * noise_level
    
    if image.dtype == np.uint8:
        return np.clip(image.astype(np.float64) + noise * 255, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        return np.clip(image.astype(np.float64) + noise * 65535, 0, 65535).astype(np.uint16)
    elif image.dtype == np.float32:
        return np.clip(image.astype(np.float32) + noise.astype(np.float32), 0, 1).astype(np.float32)
    else:
        return np.clip(image + noise, 0, 1)


def print_test_header(title):
    """Print a formatted test header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subtest(description):
    """Print a subtest description."""
    print(f"\n▶ {description}")


def assert_ssim_range(ssim_value, msg=""):
    """Assert SSIM value is in valid range [-1, 1]."""
    assert -1 <= ssim_value <= 1, f"SSIM {ssim_value} out of range [-1, 1]. {msg}"


# =============================================================================
# Core SSIM Test Scenarios
# =============================================================================
def test_identical_arrays():
    """
    Test 1: Identical arrays should return SSIM = 1.0
    
    This is the most basic test - two identical images should have
    perfect SSIM score of exactly 1.0.
    """
    print_subtest("Testing identical 2D arrays (grayscale)")
    
    img = create_test_image(size=64, channels=1, seed=42)
    score = compute_ssim(img, img)
    
    print(f"   SSIM score: {score:.10f}")
    assert score == 1.0, f"Expected 1.0 for identical arrays, got {score}"
    print("   ✓ PASS")


def test_identical_color_arrays():
    """
    Test 2: Identical color images should return SSIM = 1.0
    """
    print_subtest("Testing identical 3D arrays (color images)")
    
    img = create_test_image(size=64, channels=3, seed=42)
    score = compute_ssim(img, img)
    
    print(f"   SSIM score: {score:.10f}")
    assert score == 1.0, f"Expected 1.0 for identical color arrays, got {score}"
    print("   ✓ PASS")


def test_completely_different_arrays():
    """
    Test 3: Completely different arrays should return a low SSIM value
    
    Two randomly generated images should have a low SSIM score,
    typically close to 0 (though exact value depends on random seed).
    """
    print_subtest("Testing completely different 2D arrays")
    
    np.random.seed(42)
    img1 = np.random.rand(64, 64)
    np.random.seed(123)
    img2 = np.random.rand(64, 64)
    
    score = compute_ssim(img1, img2)
    
    print(f"   SSIM score: {score:.6f} (expected: low, < 0.5)")
    assert score < SSIM_LOW_THRESHOLD, f"Expected SSIM < {SSIM_LOW_THRESHOLD} for different arrays, got {score}"
    print("   ✓ PASS")


def test_completely_different_color_arrays():
    """
    Test 4: Completely different color arrays should return low SSIM
    """
    print_subtest("Testing completely different 3D arrays (color)")
    
    np.random.seed(42)
    img1 = np.random.rand(64, 64, 3)
    np.random.seed(999)
    img2 = np.random.rand(64, 64, 3)
    
    score = compute_ssim(img1, img2)
    
    print(f"   SSIM score: {score:.6f} (expected: low, < 0.5)")
    assert score < SSIM_LOW_THRESHOLD, f"Expected SSIM < {SSIM_LOW_THRESHOLD} for different color arrays, got {score}"
    print("   ✓ PASS")


def test_slightly_modified_arrays():
    """
    Test 5: Slightly modified arrays should detect similarity correctly
    
    This test adds small Gaussian noise to an image and verifies:
    - SSIM is high (close to 1.0) for small modifications
    - SSIM decreases as noise increases
    """
    print_subtest("Testing similarity detection with small modifications")
    
    np.random.seed(42)
    img1 = np.random.rand(64, 64)
    
    # Test with very small noise
    img2_small_noise = add_noise(img1, noise_level=0.005, seed=100)
    score_small = compute_ssim(img1, img2_small_noise)
    print(f"   SSIM with 0.5% noise: {score_small:.6f}")
    assert score_small > 0.95, f"Expected SSIM > 0.95 for small noise, got {score_small}"
    
    # Test with small noise
    img2_med_noise = add_noise(img1, noise_level=0.01, seed=101)
    score_med = compute_ssim(img1, img2_med_noise)
    print(f"   SSIM with 1% noise: {score_med:.6f}")
    assert score_med > SSIM_HIGH_THRESHOLD, f"Expected SSIM > {SSIM_HIGH_THRESHOLD} for 1% noise, got {score_med}"
    
    # Test with larger noise
    img2_large_noise = add_noise(img1, noise_level=0.05, seed=102)
    score_large = compute_ssim(img1, img2_large_noise)
    print(f"   SSIM with 5% noise: {score_large:.6f}")
    
    # Verify monotonic decrease
    assert score_small > score_med > score_large, \
        f"SSIM should decrease with noise: {score_small} > {score_med} > {score_large}"
    
    print("   ✓ PASS - SSIM correctly detects similarity levels")


def test_slightly_modified_color_arrays():
    """
    Test 6: Slightly modified color arrays - similarity detection
    """
    print_subtest("Testing similarity detection with color images")
    
    np.random.seed(42)
    img1 = np.random.rand(64, 64, 3)
    
    # Test with small noise
    img2 = add_noise(img1, noise_level=0.01, seed=103)
    score = compute_ssim(img1, img2)
    
    print(f"   SSIM with 1% noise (color): {score:.6f}")
    assert score > SSIM_HIGH_THRESHOLD, f"Expected SSIM > {SSIM_HIGH_THRESHOLD}, got {score}"
    print("   ✓ PASS")


# =============================================================================
# Edge Case Tests
# =============================================================================
def test_ssim_range_bounds():
    """
    Test 7: Verify SSIM always returns values in [-1, 1] range
    """
    print_subtest("Testing SSIM value bounds")
    
    test_cases = [
        ("identical", lambda: (create_test_image(seed=1), create_test_image(seed=1))),
        ("different", lambda: (np.random.rand(64, 64), np.random.rand(64, 64))),
        ("constant", lambda: (np.ones((64, 64)), np.zeros((64, 64)))),
    ]
    
    for name, img_gen in test_cases:
        np.random.seed(42)
        img1, img2 = img_gen()
        score = compute_ssim(img1, img2)
        print(f"   {name}: SSIM = {score:.6f}")
        assert_ssim_range(score, f"Test case: {name}")
    
    print("   ✓ PASS - All SSIM values in valid range")


def test_different_sizes():
    """
    Test 8: Arrays with different sizes should raise SSIMInputError
    """
    print_subtest("Testing input validation for different sizes")
    
    img1 = np.random.rand(64, 64)
    img2 = np.random.rand(32, 32)
    
    try:
        compute_ssim(img1, img2)
        assert False, "Should have raised SSIMInputError for size mismatch"
    except SSIMInputError as e:
        print(f"   ✓ Correctly raised SSIMInputError: {e}")
    
    print("   ✓ PASS")


def test_different_dimensions():
    """
    Test 9: Arrays with different dimensions should raise SSIMInputError
    """
    print_subtest("Testing input validation for different dimensions")
    
    img1 = np.random.rand(64, 64)
    img2 = np.random.rand(64, 64, 3)
    
    try:
        compute_ssim(img1, img2)
        assert False, "Should have raised SSIMInputError for dimension mismatch"
    except SSIMInputError as e:
        print(f"   ✓ Correctly raised SSIMInputError: {e}")
    
    print("   ✓ PASS")


def test_different_dtypes():
    """
    Test 10: Arrays with different dtypes should raise SSIMInputError
    """
    print_subtest("Testing input validation for different dtypes")
    
    img1 = np.random.rand(64, 64).astype(np.float32)
    img2 = np.random.rand(64, 64).astype(np.float64)
    
    try:
        compute_ssim(img1, img2)
        assert False, "Should have raised SSIMInputError for dtype mismatch"
    except SSIMInputError as e:
        print(f"   ✓ Correctly raised SSIMInputError: {e}")
    
    print("   ✓ PASS")


def test_empty_arrays():
    """
    Test 11: Empty arrays should raise SSIMInputError
    """
    print_subtest("Testing input validation for empty arrays")
    
    img1 = np.array([])
    img2 = np.array([])
    
    try:
        compute_ssim(img1, img2)
        assert False, "Should have raised SSIMInputError for empty arrays"
    except SSIMInputError as e:
        print(f"   ✓ Correctly raised SSIMInputError: {e}")
    
    print("   ✓ PASS")


def test_non_numpy_inputs():
    """
    Test 12: Non-numpy array inputs should raise TypeError
    """
    print_subtest("Testing input validation for non-numpy inputs")
    
    img1 = [[1, 2], [3, 4]]  # List instead of numpy array
    img2 = np.random.rand(2, 2)
    
    try:
        compute_ssim(img1, img2)
        assert False, "Should have raised TypeError for non-numpy input"
    except TypeError as e:
        print(f"   ✓ Correctly raised TypeError: {e}")
    
    print("   ✓ PASS")


def test_invalid_dimensions():
    """
    Test 13: 1D arrays should raise SSIMInputError (invalid for images)
    """
    print_subtest("Testing input validation for 1D arrays")
    
    img1 = np.random.rand(100)
    img2 = np.random.rand(100)
    
    try:
        compute_ssim(img1, img2)
        assert False, "Should have raised SSIMInputError for 1D arrays"
    except SSIMInputError as e:
        print(f"   ✓ Correctly raised SSIMInputError: {e}")
    
    print("   ✓ PASS")


# =============================================================================
# Data Type Tests
# =============================================================================
def test_uint8_images():
    """
    Test 14: Test SSIM with uint8 images
    """
    print_subtest("Testing uint8 images")
    
    np.random.seed(42)
    img1 = create_test_image(size=64, channels=1, dtype=np.uint8, seed=42)
    
    # Identical images
    score_identical = compute_ssim(img1, img1)
    print(f"   Identical uint8 SSIM: {score_identical:.10f}")
    assert score_identical == 1.0, f"Expected 1.0, got {score_identical}"
    
    # With noise
    img2 = add_noise(img1, noise_level=0.01, seed=50)
    score_noise = compute_ssim(img1, img2)
    print(f"   With noise SSIM: {score_noise:.6f}")
    assert score_noise > 0.9, f"Expected SSIM > 0.9, got {score_noise}"
    
    print("   ✓ PASS")


def test_uint16_images():
    """
    Test 15: Test SSIM with uint16 images
    """
    print_subtest("Testing uint16 images")
    
    img1 = create_test_image(size=64, channels=1, dtype=np.uint16, seed=42)
    
    # Identical images
    score_identical = compute_ssim(img1, img1)
    print(f"   Identical uint16 SSIM: {score_identical:.10f}")
    assert score_identical == 1.0, f"Expected 1.0, got {score_identical}"
    
    print("   ✓ PASS")


def test_float32_images():
    """
    Test 16: Test SSIM with float32 images
    """
    print_subtest("Testing float32 images")
    
    img1 = create_test_image(size=64, channels=1, dtype=np.float32, seed=42)
    img2 = add_noise(img1, noise_level=0.01, seed=60)
    
    score = compute_ssim(img1, img2)
    print(f"   float32 SSIM: {score:.6f}")
    assert score > 0.9, f"Expected SSIM > 0.9, got {score}"
    
    print("   ✓ PASS")


def test_float64_images():
    """
    Test 17: Test SSIM with float64 images (default)
    """
    print_subtest("Testing float64 images")
    
    img1 = create_test_image(size=64, channels=1, dtype=np.float64, seed=42)
    img2 = add_noise(img1, noise_level=0.01, seed=70)
    
    score = compute_ssim(img1, img2)
    print(f"   float64 SSIM: {score:.6f}")
    assert score > 0.9, f"Expected SSIM > 0.9, got {score}"
    
    print("   ✓ PASS")


# =============================================================================
# SSIM Map Tests
# =============================================================================
def test_ssim_map_identical():
    """
    Test 18: SSIM map of identical images should be all ones
    """
    print_subtest("Testing SSIM map for identical images")
    
    img = np.random.rand(32, 32)
    ssim_map = compute_ssim_map(img, img)
    
    print(f"   Map shape: {ssim_map.shape}")
    print(f"   Map mean: {ssim_map.mean():.10f}")
    print(f"   Map min: {ssim_map.min():.10f}, max: {ssim_map.max():.10f}")
    
    assert ssim_map.shape == (32, 32), f"Expected shape (32, 32), got {ssim_map.shape}"
    np.testing.assert_array_almost_equal(ssim_map, np.ones((32, 32)), decimal=6)
    
    print("   ✓ PASS")


def test_ssim_map_different():
    """
    Test 19: SSIM map of different images should have lower values
    """
    print_subtest("Testing SSIM map for different images")
    
    np.random.seed(42)
    img1 = np.random.rand(32, 32)
    np.random.seed(999)
    img2 = np.random.rand(32, 32)
    
    ssim_map = compute_ssim_map(img1, img2)
    
    print(f"   Map shape: {ssim_map.shape}")
    print(f"   Map mean: {ssim_map.mean():.6f}")
    print(f"   Map min: {ssim_map.min():.6f}, max: {ssim_map.max():.6f}")
    
    assert ssim_map.mean() < 0.5, f"Expected mean SSIM map < 0.5, got {ssim_map.mean()}"
    
    print("   ✓ PASS")


def test_ssim_map_return_type():
    """
    Test 20: compute_ssim_map should return numpy array
    """
    print_subtest("Testing SSIM map return type")
    
    img1 = np.random.rand(32, 32)
    img2 = np.random.rand(32, 32)
    ssim_map = compute_ssim_map(img1, img2)
    
    assert isinstance(ssim_map, np.ndarray), f"Expected ndarray, got {type(ssim_map)}"
    print(f"   ✓ PASS - Returns {type(ssim_map)}")


# =============================================================================
# Advanced Similarity Detection Tests
# =============================================================================
def test_gradual_degradation():
    """
    Test 21: Test SSIM responds correctly to gradual image degradation
    
    As we add more noise, SSIM should decrease monotonically.
    """
    print_subtest("Testing SSIM response to gradual degradation")
    
    np.random.seed(42)
    img_base = np.random.rand(64, 64)
    
    noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    scores = []
    
    for noise in noise_levels:
        img_noisy = add_noise(img_base, noise_level=noise, seed=200)
        score = compute_ssim(img_base, img_noisy)
        scores.append(score)
        print(f"   Noise {noise:6.3f}: SSIM = {score:.6f}")
    
    # Check monotonic decrease
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1], \
            f"SSIM should decrease with noise: {scores[i]} !> {scores[i+1]}"
    
    print("   ✓ PASS - SSIM decreases monotonically with noise")


def test_structural_vs_noise():
    """
    Test 22: SSIM should be sensitive to structural changes
    
    Compare: adding noise vs shifting the entire image
    Shifting should have more impact on structure than noise.
    """
    print_subtest("Testing structural sensitivity vs noise")
    
    np.random.seed(42)
    img = np.random.rand(64, 64)
    
    # Add noise
    img_noisy = add_noise(img, noise_level=0.05, seed=300)
    ssim_noise = compute_ssim(img, img_noisy)
    
    # Shift image (circular shift)
    img_shifted = np.roll(img, shift=5, axis=0)
    ssim_shifted = compute_ssim(img, img_shifted)
    
    print(f"   SSIM with 5% noise:    {ssim_noise:.6f}")
    print(f"   SSIM with shift by 5:  {ssim_shifted:.6f}")
    
    # Structural shift should have lower SSIM than noise
    assert ssim_shifted < ssim_noise, \
        f"Structural shift should lower SSIM more than noise"
    
    print("   ✓ PASS - SSIM is more sensitive to structural changes")


def test_local_variations():
    """
    Test 23: Test that SSIM detects local variations correctly
    """
    print_subtest("Testing local variation detection")
    
    np.random.seed(42)
    img1 = np.random.rand(64, 64)
    
    # Create image with local modifications
    img2 = img1.copy()
    img2[20:30, 20:30] += 0.1  # Modify a small region
    img2 = np.clip(img2, 0, 1)
    
    ssim_map = compute_ssim_map(img1, img2)
    
    print(f"   SSIM map mean: {ssim_map.mean():.6f}")
    print(f"   SSIM map min in modified region: {ssim_map[20:30, 20:30].mean():.6f}")
    print(f"   SSIM map max elsewhere: {ssim_map[:20, :].mean():.6f}")
    
    # Modified region should have lower SSIM
    modified_ssim = ssim_map[20:30, 20:30].mean()
    unmodified_ssim = ssim_map[:20, :].mean()
    
    assert modified_ssim < unmodified_ssim, \
        f"Modified region should have lower SSIM: {modified_ssim} !< {unmodified_ssim}"
    
    print("   ✓ PASS - SSIM detects local variations")


# =============================================================================
# Multi-Channel Tests
# =============================================================================
def test_rgb_channels():
    """
    Test 24: Test SSIM with RGB images
    """
    print_subtest("Testing RGB images")
    
    np.random.seed(42)
    img1 = np.random.rand(64, 64, 3)
    img2 = add_noise(img1, noise_level=0.01, seed=400)
    
    score = compute_ssim(img1, img2)
    print(f"   RGB SSIM with 1% noise: {score:.6f}")
    
    assert score > 0.9, f"Expected SSIM > 0.9, got {score}"
    print("   ✓ PASS")


def test_rgba_channels():
    """
    Test 25: Test SSIM with RGBA images
    """
    print_subtest("Testing RGBA images")
    
    np.random.seed(42)
    img1 = np.random.rand(64, 64, 4)
    img2 = add_noise(img1, noise_level=0.01, seed=500)
    
    score = compute_ssim(img1, img2)
    print(f"   RGBA SSIM with 1% noise: {score:.6f}")
    
    assert score > 0.9, f"Expected SSIM > 0.9, got {score}"
    print("   ✓ PASS")


# =============================================================================
# Window Size Tests
# =============================================================================
def test_different_window_sizes():
    """
    Test 26: Test SSIM with different window sizes
    """
    print_subtest("Testing different window sizes")
    
    np.random.seed(42)
    img1 = np.random.rand(64, 64)
    img2 = add_noise(img1, noise_level=0.02, seed=600)
    
    window_sizes = [5, 7, 11, 15]
    scores = []
    
    for ws in window_sizes:
        score = compute_ssim(img1, img2, window_size=ws)
        scores.append(score)
        print(f"   Window size {ws}: SSIM = {score:.6f}")
    
    print("   ✓ PASS - All window sizes computed successfully")


# =============================================================================
# Main Test Runner
# =============================================================================
def run_all_tests():
    """Run all comprehensive tests."""
    
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE JUMBO SSIM TEST SUITE")
    print("=" * 70)
    print(f"\nRunning {26} comprehensive tests...\n")
    
    test_functions = [
        # Core SSIM Tests
        test_identical_arrays,
        test_identical_color_arrays,
        test_completely_different_arrays,
        test_completely_different_color_arrays,
        test_slightly_modified_arrays,
        test_slightly_modified_color_arrays,
        
        # Edge Case Tests
        test_ssim_range_bounds,
        test_different_sizes,
        test_different_dimensions,
        test_different_dtypes,
        test_empty_arrays,
        test_non_numpy_inputs,
        test_invalid_dimensions,
        
        # Data Type Tests
        test_uint8_images,
        test_uint16_images,
        test_float32_images,
        test_float64_images,
        
        # SSIM Map Tests
        test_ssim_map_identical,
        test_ssim_map_different,
        test_ssim_map_return_type,
        
        # Advanced Tests
        test_gradual_degradation,
        test_structural_vs_noise,
        test_local_variations,
        
        # Multi-Channel Tests
        test_rgb_channels,
        test_rgba_channels,
        
        # Window Size Tests
        test_different_window_sizes,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n   ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"\n   ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"  TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n  🎉 ALL TESTS PASSED! 🎉\n")
        return True
    else:
        print(f"\n  ⚠️  {failed} test(s) failed\n")
        return False


# =============================================================================
# Pytest Test Classes (for pytest integration)
# =============================================================================
if HAS_PYTEST:
    class TestSSIMIdentical:
        """Tests for identical arrays - should return 1.0"""
        
        def test_identical_grayscale(self):
            img = np.random.rand(64, 64)
            assert compute_ssim(img, img) == 1.0
        
        def test_identical_rgb(self):
            img = np.random.rand(64, 64, 3)
            assert compute_ssim(img, img) == 1.0
        
        def test_identical_rgba(self):
            img = np.random.rand(64, 64, 4)
            assert compute_ssim(img, img) == 1.0
        
        def test_identical_uint8(self):
            img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            assert compute_ssim(img, img) == 1.0
        
        def test_identical_float32(self):
            img = np.random.rand(64, 64).astype(np.float32)
            assert compute_ssim(img, img) == 1.0
    
    class TestSSIMDifferent:
        """Tests for completely different arrays - should return low values"""
        
        def test_different_grayscale(self):
            np.random.seed(42)
            img1 = np.random.rand(64, 64)
            np.random.seed(123)
            img2 = np.random.rand(64, 64)
            assert compute_ssim(img1, img2) < 0.5
        
        def test_different_rgb(self):
            np.random.seed(42)
            img1 = np.random.rand(64, 64, 3)
            np.random.seed(999)
            img2 = np.random.rand(64, 64, 3)
            assert compute_ssim(img1, img2) < 0.5
        
        def test_different_uniform(self):
            img1 = np.zeros((64, 64))
            img2 = np.ones((64, 64))
            score = compute_ssim(img1, img2)
            assert -1 <= score < 0.5
    
    class TestSSIMSimilar:
        """Tests for slightly modified arrays - similarity detection"""
        
        def test_small_noise_high_ssim(self):
            np.random.seed(42)
            img1 = np.random.rand(64, 64)
            img2 = np.clip(img1 + np.random.randn(64, 64) * 0.005, 0, 1)
            assert compute_ssim(img1, img2) > 0.95
        
        def test_medium_noise_good_ssim(self):
            np.random.seed(42)
            img1 = np.random.rand(64, 64)
            img2 = np.clip(img1 + np.random.randn(64, 64) * 0.01, 0, 1)
            assert compute_ssim(img1, img2) > 0.9
        
        def test_large_noise_lower_ssim(self):
            np.random.seed(42)
            img1 = np.random.rand(64, 64)
            img2 = np.clip(img1 + np.random.randn(64, 64) * 0.05, 0, 1)
            score = compute_ssim(img1, img2)
            assert 0 < score < 0.99
        
        def test_monotonic_noise_response(self):
            np.random.seed(42)
            img = np.random.rand(64, 64)
            
            scores = []
            for noise in [0.001, 0.01, 0.05, 0.1]:
                img_noisy = np.clip(img + np.random.randn(64, 64) * noise, 0, 1)
                scores.append(compute_ssim(img, img_noisy))
            
            for i in range(len(scores) - 1):
                assert scores[i] > scores[i + 1]
        
        def test_color_noise(self):
            np.random.seed(42)
            img1 = np.random.rand(64, 64, 3)
            img2 = np.clip(img1 + np.random.randn(64, 64, 3) * 0.01, 0, 1)
            assert compute_ssim(img1, img2) > 0.9
    
    class TestSSIMValidation:
        """Input validation tests"""
        
        def test_shape_mismatch(self):
            with pytest.raises(SSIMInputError):
                compute_ssim(np.random.rand(64, 64), np.random.rand(32, 32))
        
        def test_dimension_mismatch(self):
            with pytest.raises(SSIMInputError):
                compute_ssim(np.random.rand(64, 64), np.random.rand(64, 64, 3))
        
        def test_dtype_mismatch(self):
            with pytest.raises(SSIMInputError):
                compute_ssim(
                    np.random.rand(64, 64).astype(np.float32),
                    np.random.rand(64, 64).astype(np.float64)
                )
        
        def test_empty_array(self):
            with pytest.raises(SSIMInputError):
                compute_ssim(np.array([]), np.array([]))
        
        def test_non_numpy_input(self):
            with pytest.raises(TypeError):
                compute_ssim([[1, 2], [3, 4]], np.random.rand(2, 2))
        
        def test_1d_array(self):
            with pytest.raises(SSIMInputError):
                compute_ssim(np.random.rand(100), np.random.rand(100))
    
    class TestSSIMMap:
        """SSIM map tests"""
        
        def test_map_shape(self):
            img = np.random.rand(64, 64)
            ssim_map = compute_ssim_map(img, img)
            assert ssim_map.shape == (64, 64)
        
        def test_map_identical_all_ones(self):
            img = np.random.rand(32, 32)
            ssim_map = compute_ssim_map(img, img)
            np.testing.assert_array_almost_equal(ssim_map, np.ones((32, 32)), decimal=6)
        
        def test_map_returns_array(self):
            img1 = np.random.rand(32, 32)
            img2 = np.random.rand(32, 32)
            ssim_map = compute_ssim_map(img1, img2)
            assert isinstance(ssim_map, np.ndarray)
    
    class TestSSIMRange:
        """SSIM value range tests"""
        
        def test_ssim_in_valid_range_identical(self):
            img = np.random.rand(64, 64)
            score = compute_ssim(img, img)
            assert -1 <= score <= 1
        
        def test_ssim_in_valid_range_different(self):
            img1 = np.random.rand(64, 64)
            img2 = np.random.rand(64, 64)
            score = compute_ssim(img1, img2)
            assert -1 <= score <= 1
        
        def test_ssim_in_valid_range_constant(self):
            img1 = np.zeros((64, 64))
            img2 = np.ones((64, 64))
            score = compute_ssim(img1, img2)
            assert -1 <= score <= 1
    
    class TestSSIMDataTypes:
        """Tests for different data types"""
        
        def test_uint8(self):
            img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
            assert compute_ssim(img, img) == 1.0
        
        def test_uint16(self):
            img = np.random.randint(0, 65536, (64, 64), dtype=np.uint16)
            assert compute_ssim(img, img) == 1.0
        
        def test_float32(self):
            img = np.random.rand(64, 64).astype(np.float32)
            assert compute_ssim(img, img) == 1.0
        
        def test_float64(self):
            img = np.random.rand(64, 64).astype(np.float64)
            assert compute_ssim(img, img) == 1.0


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
