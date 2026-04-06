"""
Test suite for SSIM (Structural Similarity Index) implementation.

This module contains unit tests to verify the correctness of the SSIM
implementation with various input scenarios.

Usage:
    python test_ssim.py          # Run simple tests
    pytest test_ssim.py          # Run full test suite (requires pytest)
"""

import numpy as np
from ssim import compute_ssim, compute_ssim_map, SSIMInputError

# Try to import pytest for full test suite
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


def run_simple_test():
    """
    Run a simple test to demonstrate SSIM functionality.
    
    This function can be called directly to verify the implementation works.
    """
    print("=" * 60)
    print("SSIM Implementation Simple Test")
    print("=" * 60)
    
    # Test 1: Identical images
    print("\n1. Testing identical images...")
    img = np.random.rand(64, 64)
    score = compute_ssim(img, img)
    print(f"   SSIM score: {score:.10f}")
    assert score == 1.0
    print("   ✓ PASS")
    
    # Test 2: Similar images with noise
    print("\n2. Testing similar images with small noise...")
    np.random.seed(42)
    img1 = np.random.rand(64, 64)
    noise = np.random.randn(64, 64) * 0.01
    img2 = np.clip(img1 + noise, 0, 1)
    score = compute_ssim(img1, img2)
    print(f"   SSIM score: {score:.6f} (high, close to 1.0)")
    assert score > 0.9
    print("   ✓ PASS")
    
    # Test 3: Different images
    print("\n3. Testing different random images...")
    np.random.seed(42)
    img1 = np.random.rand(64, 64)
    np.random.seed(123)
    img2 = np.random.rand(64, 64)
    score = compute_ssim(img1, img2)
    print(f"   SSIM score: {score:.6f} (low)")
    assert score < 0.5
    print("   ✓ PASS")
    
    # Test 4: Color images
    print("\n4. Testing color images...")
    color_img = np.random.rand(64, 64, 3)
    score = compute_ssim(color_img, color_img)
    print(f"   SSIM score: {score:.10f}")
    assert score == 1.0
    print("   ✓ PASS")
    
    # Test 5: SSIM map
    print("\n5. Testing SSIM map...")
    img = np.random.rand(32, 32)
    ssim_map = compute_ssim_map(img, img)
    print(f"   Map shape: {ssim_map.shape}")
    print(f"   Map mean: {ssim_map.mean():.6f}")
    assert ssim_map.shape == (32, 32)
    assert np.allclose(ssim_map, 1.0)
    print("   ✓ PASS")
    
    # Test 6: Input validation
    print("\n6. Testing input validation...")
    try:
        compute_ssim(np.random.rand(32, 32), np.random.rand(16, 16))
        print("   ✗ FAIL - Should have raised SSIMInputError")
    except SSIMInputError:
        print("   ✓ PASS - Correctly raised SSIMInputError for shape mismatch")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if HAS_PYTEST:
    # Full test suite using pytest
    
    class TestSSIMBasic:
        """Basic functionality tests for SSIM computation."""
        
        def test_identical_images_return_one(self):
            """Identical images should have SSIM = 1.0."""
            img = np.random.rand(64, 64)
            score = compute_ssim(img, img)
            assert score == 1.0, f"Expected 1.0, got {score}"
        
        def test_identical_color_images(self):
            """Identical color images should have SSIM = 1.0."""
            img = np.random.rand(64, 64, 3)
            score = compute_ssim(img, img)
            assert score == 1.0, f"Expected 1.0, got {score}"
        
        def test_ssim_range(self):
            """SSIM score should be in range [-1, 1]."""
            img1 = np.random.rand(64, 64)
            img2 = np.random.rand(64, 64)
            score = compute_ssim(img1, img2)
            assert -1 <= score <= 1, f"Score {score} out of range [-1, 1]"
        
        def test_similar_images_high_ssim(self):
            """Similar images with small noise should have high SSIM."""
            np.random.seed(42)
            img1 = np.random.rand(64, 64)
            noise = np.random.randn(64, 64) * 0.01
            img2 = np.clip(img1 + noise, 0, 1)
            score = compute_ssim(img1, img2)
            assert score > 0.9, f"Expected high SSIM for similar images, got {score}"
        
        def test_different_images_low_ssim(self):
            """Completely different images should have low SSIM."""
            np.random.seed(42)
            img1 = np.random.rand(64, 64)
            np.random.seed(123)
            img2 = np.random.rand(64, 64)
            score = compute_ssim(img1, img2)
            assert score < 0.5, f"Expected low SSIM for different images, got {score}"
        
        def test_return_type_is_float(self):
            """SSIM function should return a float."""
            img1 = np.random.rand(32, 32)
            img2 = np.random.rand(32, 32)
            score = compute_ssim(img1, img2)
            assert isinstance(score, float), f"Expected float, got {type(score)}"


    class TestSSIMInputValidation:
        """Input validation tests for SSIM computation."""
        
        def test_shape_mismatch_raises_error(self):
            """Arrays with different shapes should raise SSIMInputError."""
            img1 = np.random.rand(64, 64)
            img2 = np.random.rand(32, 32)
            with pytest.raises(SSIMInputError):
                compute_ssim(img1, img2)
        
        def test_dimension_mismatch_raises_error(self):
            """Arrays with different dimensions should raise SSIMInputError."""
            img1 = np.random.rand(64, 64)
            img2 = np.random.rand(64, 64, 3)
            with pytest.raises(SSIMInputError):
                compute_ssim(img1, img2)
        
        def test_dtype_mismatch_raises_error(self):
            """Arrays with different dtypes should raise SSIMInputError."""
            img1 = np.random.rand(64, 64).astype(np.float32)
            img2 = np.random.rand(64, 64).astype(np.float64)
            with pytest.raises(SSIMInputError):
                compute_ssim(img1, img2)
        
        def test_empty_array_raises_error(self):
            """Empty arrays should raise SSIMInputError."""
            img1 = np.array([])
            img2 = np.array([])
            with pytest.raises(SSIMInputError):
                compute_ssim(img1, img2)
        
        def test_non_numpy_array_raises_type_error(self):
            """Non-numpy array inputs should raise TypeError."""
            img1 = [[1, 2], [3, 4]]  # List instead of numpy array
            img2 = np.random.rand(2, 2)
            with pytest.raises(TypeError):
                compute_ssim(img1, img2)
        
        def test_invalid_dimensions_raises_error(self):
            """1D arrays should raise SSIMInputError."""
            img1 = np.random.rand(100)
            img2 = np.random.rand(100)
            with pytest.raises(SSIMInputError):
                compute_ssim(img1, img2)


    class TestSSIMMap:
        """Tests for SSIM map computation."""
        
        def test_ssim_map_shape(self):
            """SSIM map should have same shape as input."""
            img = np.random.rand(64, 64)
            ssim_map = compute_ssim_map(img, img)
            assert ssim_map.shape == (64, 64), f"Expected shape (64, 64), got {ssim_map.shape}"
        
        def test_ssim_map_identical_images(self):
            """SSIM map of identical images should be all ones."""
            img = np.random.rand(32, 32)
            ssim_map = compute_ssim_map(img, img)
            np.testing.assert_array_almost_equal(ssim_map, np.ones((32, 32)), decimal=6)
        
        def test_ssim_map_returns_numpy_array(self):
            """compute_ssim_map should return a numpy array."""
            img1 = np.random.rand(32, 32)
            img2 = np.random.rand(32, 32)
            ssim_map = compute_ssim_map(img1, img2)
            assert isinstance(ssim_map, np.ndarray), f"Expected ndarray, got {type(ssim_map)}"
        
        def test_ssim_map_input_validation(self):
            """SSIM map should validate inputs."""
            img1 = np.random.rand(64, 64)
            img2 = np.random.rand(32, 32)
            with pytest.raises(SSIMInputError):
                compute_ssim_map(img1, img2)


if __name__ == '__main__':
    run_simple_test()
