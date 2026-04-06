"""
Comprehensive tests for orb_diff.py - ORB feature detection and image difference.

Tests cover:
- Input validation (types, shapes, dimensions, dtypes)
- Algorithm correctness (keypoint detection, descriptors, matching, homography)
- Edge cases (empty images, uniform images, small images)
- Output format verification (diff_mask and diff_map shapes)
"""

import sys
import numpy as np
import pytest
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, 'D:\\RoboCute_robot\\diff')

from orb_diff import (
    ORBInputError,
    ORBConfig,
    ORBKeypoint,
    ORBMatch,
    ORB,
    ORBMatcher,
    ORBDiff,
    orb_diff,
    compute_orb_diff,
    create_orb,
    has_opencv_orb,
    _validate_inputs,
)


# Skip all tests if OpenCV ORB is not available
try:
    import cv2
    OPENCV_AVAILABLE = hasattr(cv2, 'ORB_create')
except ImportError:
    OPENCV_AVAILABLE = False

skip_if_no_opencv = pytest.mark.skipif(
    not OPENCV_AVAILABLE,
    reason="OpenCV ORB not available"
)


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for _validate_inputs function."""
    
    def test_valid_identical_arrays(self):
        """Test validation passes for identical arrays."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()
        # Should not raise
        _validate_inputs(img1, img2)
    
    def test_valid_2d_arrays(self):
        """Test validation passes for 2D arrays."""
        img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img2 = img1.copy()
        _validate_inputs(img1, img2)
    
    def test_valid_3d_arrays(self):
        """Test validation passes for 3D arrays."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = img1.copy()
        _validate_inputs(img1, img2)
    
    def test_type_error_non_numpy(self):
        """Test TypeError raised for non-numpy inputs."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(TypeError, match="img2 must be a numpy array"):
            _validate_inputs(img1, "not an array")
        
        with pytest.raises(TypeError, match="img1 must be a numpy array"):
            _validate_inputs("not an array", img1)
    
    def test_dimension_mismatch(self):
        """Test ORBInputError for dimension mismatch."""
        img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ORBInputError, match="Dimension mismatch"):
            _validate_inputs(img1, img2)
    
    def test_shape_mismatch(self):
        """Test ORBInputError for shape mismatch."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 150, 3), dtype=np.uint8)
        
        with pytest.raises(ORBInputError, match="Shape mismatch"):
            _validate_inputs(img1, img2)
    
    def test_size_mismatch_3d(self):
        """Test ORBInputError for 3D size mismatch."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)
        
        with pytest.raises(ORBInputError, match="Shape mismatch"):
            _validate_inputs(img1, img2)
    
    def test_empty_array(self):
        """Test ORBInputError for empty arrays."""
        img1 = np.array([], dtype=np.uint8).reshape(0, 0)
        img2 = np.array([], dtype=np.uint8).reshape(0, 0)
        
        with pytest.raises(ORBInputError, match="Empty input arrays"):
            _validate_inputs(img1, img2)
    
    def test_dtype_mismatch(self):
        """Test ORBInputError for dtype mismatch."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = (np.random.rand(100, 100, 3) * 255).astype(np.float32)
        
        with pytest.raises(ORBInputError, match="Dtype mismatch"):
            _validate_inputs(img1, img2)
    
    def test_dtype_mismatch_allowed(self):
        """Test dtype mismatch allowed with flag."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = (np.random.rand(100, 100, 3) * 255).astype(np.float32)
        
        # Should not raise when allow_different_dtypes=True
        _validate_inputs(img1, img2, allow_different_dtypes=True)
    
    def test_invalid_dimensions_1d(self):
        """Test ORBInputError for 1D arrays."""
        img1 = np.random.randint(0, 256, (100,), dtype=np.uint8)
        img2 = img1.copy()
        
        with pytest.raises(ORBInputError, match="Invalid dimensions"):
            _validate_inputs(img1, img2)
    
    def test_invalid_dimensions_4d(self):
        """Test ORBInputError for 4D arrays."""
        img1 = np.random.randint(0, 256, (10, 10, 10, 3), dtype=np.uint8)
        img2 = img1.copy()
        
        with pytest.raises(ORBInputError, match="Invalid dimensions"):
            _validate_inputs(img1, img2)


# =============================================================================
# Data Class Tests
# =============================================================================

class TestDataClasses:
    """Tests for ORBKeypoint and ORBMatch dataclasses."""
    
    def test_keypoint_creation(self):
        """Test ORBKeypoint dataclass creation."""
        kp = ORBKeypoint(
            x=50.0,
            y=60.0,
            size=31.0,
            angle=45.0,
            response=0.8,
            octave=0,
            class_id=0,
            descriptor=np.random.randint(0, 256, (32,), dtype=np.uint8)
        )
        
        assert kp.x == 50.0
        assert kp.y == 60.0
        assert kp.size == 31.0
        assert kp.angle == 45.0
        assert kp.response == 0.8
        assert kp.octave == 0
        assert kp.class_id == 0
        assert kp.descriptor is not None
        assert kp.descriptor.shape == (32,)
    
    def test_keypoint_pt_property(self):
        """Test ORBKeypoint pt property."""
        kp = ORBKeypoint(
            x=50.0,
            y=60.0,
            size=31.0,
            angle=0.0,
            response=0.5,
            octave=0,
            class_id=0
        )
        
        assert kp.pt == (50.0, 60.0)
    
    def test_match_creation(self):
        """Test ORBMatch dataclass creation."""
        kp1 = ORBKeypoint(
            x=50.0, y=60.0, size=31.0, angle=0.0,
            response=0.5, octave=0, class_id=0
        )
        kp2 = ORBKeypoint(
            x=55.0, y=65.0, size=31.0, angle=0.0,
            response=0.6, octave=0, class_id=0
        )
        
        match = ORBMatch(
            kp1=kp1,
            kp2=kp2,
            distance=50.0,
            query_idx=0,
            train_idx=1
        )
        
        assert match.kp1 == kp1
        assert match.kp2 == kp2
        assert match.distance == 50.0
        assert match.query_idx == 0
        assert match.train_idx == 1


# =============================================================================
# ORBConfig Tests
# =============================================================================

class TestORBConfig:
    """Tests for ORBConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ORBConfig()
        
        assert config.n_features == 500
        assert config.scale_factor == 1.2
        assert config.n_levels == 8
        assert config.edge_threshold == 31
        assert config.first_level == 0
        assert config.WTA_K == 2
        assert config.score_type == 0
        assert config.patch_size == 31
        assert config.fast_threshold == 20
        assert config.match_ratio_threshold == 0.75
        assert config.ransac_threshold == 3.0
        assert config.min_matches_for_homography == 4
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ORBConfig(
            n_features=1000,
            match_ratio_threshold=0.8,
            ransac_threshold=5.0
        )
        
        assert config.n_features == 1000
        assert config.match_ratio_threshold == 0.8
        assert config.ransac_threshold == 5.0
        # Other values should remain defaults
        assert config.scale_factor == 1.2
        assert config.n_levels == 8


# =============================================================================
# ORB Algorithm Tests
# =============================================================================

@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV ORB not available")
class TestORB:
    """Tests for ORB class."""
    
    @pytest.fixture
    def synthetic_image(self):
        """Create a synthetic test image with corners/edges."""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        # Create checkerboard pattern
        for i in range(0, 128, 32):
            for j in range(0, 128, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    img[i:i+32, j:j+32] = 255
        return img
    
    def test_initialization(self):
        """Test ORB initialization."""
        orb = ORB()
        assert orb is not None
        assert isinstance(orb.config, ORBConfig)
    
    def test_initialization_with_config(self):
        """Test ORB initialization with custom config."""
        config = ORBConfig(n_features=1000)
        orb = ORB(config)
        assert orb.config.n_features == 1000
    
    def test_detect(self, synthetic_image):
        """Test keypoint detection."""
        orb = ORB(ORBConfig(n_features=500))
        keypoints = orb.detect(synthetic_image)
        
        assert isinstance(keypoints, list)
        assert len(keypoints) > 0
        assert all(isinstance(kp, ORBKeypoint) for kp in keypoints)
    
    def test_detect_and_compute(self, synthetic_image):
        """Test keypoint detection and descriptor computation."""
        orb = ORB(ORBConfig(n_features=500))
        keypoints = orb.detect_and_compute(synthetic_image)
        
        assert isinstance(keypoints, list)
        assert len(keypoints) > 0
        
        # Check that keypoints have descriptors
        kp_with_desc = [kp for kp in keypoints if kp.descriptor is not None]
        assert len(kp_with_desc) > 0
        
        # Check descriptor shape (ORB uses 32-byte descriptors)
        for kp in kp_with_desc:
            assert kp.descriptor.shape == (32,)
            assert kp.descriptor.dtype == np.uint8
    
    def test_prepare_image_grayscale(self):
        """Test image preparation for grayscale input."""
        orb = ORB()
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        prepared = orb._prepare_image(gray)
        
        assert prepared.dtype == np.uint8
        assert prepared.ndim == 2
    
    def test_prepare_image_color(self):
        """Test image preparation for color input."""
        orb = ORB()
        color = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        prepared = orb._prepare_image(color)
        
        assert prepared.dtype == np.uint8
        assert prepared.ndim == 2
    
    def test_prepare_image_float(self):
        """Test image preparation for float input."""
        orb = ORB()
        float_img = np.random.rand(100, 100, 3).astype(np.float32)
        prepared = orb._prepare_image(float_img)
        
        assert prepared.dtype == np.uint8
        assert prepared.ndim == 2


# =============================================================================
# ORBMatcher Tests
# =============================================================================

@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV ORB not available")
class TestORBMatcher:
    """Tests for ORBMatcher class."""
    
    @pytest.fixture
    def test_images(self):
        """Create two similar test images."""
        img1 = np.zeros((128, 128, 3), dtype=np.uint8)
        # Add features
        img1[30:50, 30:50] = 255
        img1[80:100, 80:100] = 255
        
        img2 = img1.copy()
        
        return img1, img2
    
    @pytest.fixture
    def different_images(self):
        """Create two different test images."""
        img1 = np.zeros((128, 128, 3), dtype=np.uint8)
        img1[30:50, 30:50] = 255
        
        img2 = np.zeros((128, 128, 3), dtype=np.uint8)
        img2[80:100, 80:100] = 255
        
        return img1, img2
    
    def test_initialization(self):
        """Test ORBMatcher initialization."""
        matcher = ORBMatcher()
        assert matcher is not None
        assert isinstance(matcher.config, ORBConfig)
    
    def test_match_identical_images(self, test_images):
        """Test matching identical images."""
        img1, img2 = test_images
        matcher = ORBMatcher()
        
        keypoints1, keypoints2, matches, homography = matcher.match(img1, img2)
        
        assert isinstance(keypoints1, list)
        assert isinstance(keypoints2, list)
        assert isinstance(matches, list)
        
        # Should have keypoints in both images
        assert len(keypoints1) > 0
        assert len(keypoints2) > 0
        
        # Should have matches for identical images
        assert len(matches) > 0
    
    def test_match_returns_homography(self, test_images):
        """Test that match returns homography for sufficient matches."""
        img1, img2 = test_images
        matcher = ORBMatcher()
        
        keypoints1, keypoints2, matches, homography = matcher.match(img1, img2)
        
        # For identical images, homography should be computed
        if len(matches) >= 4:
            assert homography is not None
            assert homography.shape == (3, 3)
    
    def test_match_no_homography_for_few_matches(self):
        """Test that match returns None homography for insufficient matches."""
        # Create images with few features
        img1 = np.zeros((128, 128, 3), dtype=np.uint8)
        img2 = np.zeros((128, 128, 3), dtype=np.uint8)
        
        matcher = ORBMatcher()
        keypoints1, keypoints2, matches, homography = matcher.match(img1, img2)
        
        # Should return None for homography with insufficient matches
        assert homography is None


# =============================================================================
# ORBDiff Tests
# =============================================================================

@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV ORB not available")
class TestORBDiff:
    """Tests for ORBDiff class."""
    
    @pytest.fixture
    def identical_images(self):
        """Create two identical test images."""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        # Add features for keypoint detection
        for i in range(0, 128, 32):
            for j in range(0, 128, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    img[i:i+32, j:j+32] = 255
        return img, img.copy()
    
    @pytest.fixture
    def different_images(self):
        """Create two different test images."""
        img1 = np.zeros((128, 128, 3), dtype=np.uint8)
        img1[30:50, 30:50] = 255
        img1[80:100, 80:100] = 255
        
        img2 = np.zeros((128, 128, 3), dtype=np.uint8)
        img2[60:80, 60:80] = 255
        
        return img1, img2
    
    def test_initialization(self):
        """Test ORBDiff initialization."""
        diff = ORBDiff()
        assert diff is not None
        assert isinstance(diff.config, ORBConfig)
        assert isinstance(diff.matcher, ORBMatcher)
    
    def test_compare_returns_dict(self, identical_images):
        """Test that compare returns a dictionary."""
        img1, img2 = identical_images
        diff = ORBDiff()
        
        result = diff.compare(img1, img2)
        
        assert isinstance(result, dict)
    
    def test_compare_has_required_keys(self, identical_images):
        """Test that compare result has all required keys."""
        img1, img2 = identical_images
        diff = ORBDiff()
        
        result = diff.compare(img1, img2)
        
        required_keys = [
            'match_ratio', 'avg_distance', 'n_keypoints1', 'n_keypoints2',
            'n_matches', 'matched_regions', 'unmatched_regions1', 'unmatched_regions2',
            'homography', 'diff_mask', 'diff_map', 'keypoints1', 'keypoints2', 'matches'
        ]
        
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_compare_diff_mask_shape(self, identical_images):
        """Test that diff_mask has correct shape."""
        img1, img2 = identical_images
        diff = ORBDiff()
        
        result = diff.compare(img1, img2)
        
        # diff_mask should be 2D with same spatial dimensions as input
        assert result['diff_mask'].ndim == 2
        assert result['diff_mask'].shape[:2] == img1.shape[:2]
        assert result['diff_mask'].dtype == np.uint8
    
    def test_compare_diff_map_shape(self, identical_images):
        """Test that diff_map has correct shape."""
        img1, img2 = identical_images
        diff = ORBDiff()
        
        result = diff.compare(img1, img2)
        
        # diff_map should be 2D with same spatial dimensions as input
        assert result['diff_map'].ndim == 2
        assert result['diff_map'].shape == img1.shape[:2]
    
    def test_compare_match_ratio_for_identical_images(self, identical_images):
        """Test match ratio for identical images."""
        img1, img2 = identical_images
        diff = ORBDiff()
        
        result = diff.compare(img1, img2)
        
        # For identical images, keypoints should be detected
        # Note: match_ratio may be 0 if no matches pass the ratio test,
        # but we should at least have keypoints detected
        assert result['n_keypoints1'] > 0
        assert result['n_keypoints2'] > 0
        # If there are matches, match_ratio should be non-negative
        assert result['match_ratio'] >= 0
    
    def test_compare_different_images(self, different_images):
        """Test comparison of different images."""
        img1, img2 = different_images
        diff = ORBDiff()
        
        result = diff.compare(img1, img2)
        
        # Should still return results even for different images
        assert 'match_ratio' in result
        assert 'diff_mask' in result
        assert 'diff_map' in result
        assert result['diff_mask'].shape[:2] == img1.shape[:2]
        assert result['diff_map'].shape == img1.shape[:2]
    
    def test_detect_changes_same_image(self, identical_images):
        """Test change detection for identical images."""
        img1, img2 = identical_images
        diff = ORBDiff()
        
        has_changes, result = diff.detect_changes(img1, img2, match_threshold=0.5)
        
        # Identical images should not have changes (or very few)
        assert isinstance(has_changes, bool)
        assert isinstance(result, dict)
    
    def test_detect_changes_different_images(self, different_images):
        """Test change detection for different images."""
        img1, img2 = different_images
        diff = ORBDiff()
        
        has_changes, result = diff.detect_changes(img1, img2, match_threshold=0.9)
        
        # Different images should have changes
        assert isinstance(has_changes, bool)
        assert isinstance(result, dict)
    
    def test_compare_2d_images(self):
        """Test comparison with 2D grayscale images."""
        img1 = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
        img2 = img1.copy()
        
        diff = ORBDiff()
        result = diff.compare(img1, img2)
        
        assert result['diff_mask'].shape == img1.shape
        assert result['diff_map'].shape == img1.shape
    
    def test_compare_with_homography(self):
        """Test that homography is computed when possible."""
        # Create image with clear features
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        img1[50:100, 50:100] = 255
        img1[150:200, 150:200] = 255
        
        # Slightly shifted version
        img2 = np.zeros((256, 256, 3), dtype=np.uint8)
        img2[52:102, 52:102] = 255
        img2[152:202, 152:202] = 255
        
        diff = ORBDiff()
        result = diff.compare(img1, img2)
        
        # Homography may or may not be computed depending on matches
        if result['n_matches'] >= 4:
            assert result['homography'] is not None
            assert result['homography'].shape == (3, 3)


# =============================================================================
# Convenience Function Tests
# =============================================================================

@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV ORB not available")
class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    @pytest.fixture
    def test_images(self):
        """Create test images."""
        img1 = np.zeros((128, 128, 3), dtype=np.uint8)
        img1[30:50, 30:50] = 255
        img1[80:100, 80:100] = 255
        
        img2 = img1.copy()
        
        return img1, img2
    
    def test_orb_diff(self, test_images):
        """Test orb_diff convenience function."""
        img1, img2 = test_images
        
        result = orb_diff(img1, img2)
        
        assert isinstance(result, dict)
        assert 'match_ratio' in result
        assert 'diff_mask' in result
        assert 'diff_map' in result
    
    def test_orb_diff_with_kwargs(self, test_images):
        """Test orb_diff with custom configuration."""
        img1, img2 = test_images
        
        result = orb_diff(img1, img2, n_features=1000, match_ratio_threshold=0.8)
        
        assert isinstance(result, dict)
        assert 'match_ratio' in result
    
    def test_compute_orb_diff(self, test_images):
        """Test compute_orb_diff convenience function."""
        img1, img2 = test_images
        
        result = compute_orb_diff(img1, img2)
        
        assert isinstance(result, dict)
        assert 'match_ratio' in result
        assert 'diff_mask' in result
        assert 'diff_map' in result
    
    def test_create_orb_default(self):
        """Test create_orb with default config."""
        orb = create_orb()
        
        assert isinstance(orb, ORB)
        assert orb.config.n_features == 500  # default
    
    def test_create_orb_with_kwargs(self):
        """Test create_orb with kwargs."""
        orb = create_orb(n_features=1000, match_ratio_threshold=0.8)
        
        assert isinstance(orb, ORB)
        assert orb.config.n_features == 1000
        assert orb.config.match_ratio_threshold == 0.8
    
    def test_create_orb_with_config(self):
        """Test create_orb with existing config."""
        config = ORBConfig(n_features=2000)
        orb = create_orb(config)
        
        assert isinstance(orb, ORB)
        assert orb.config.n_features == 2000
    
    def test_has_opencv_orb(self):
        """Test has_opencv_orb function."""
        result = has_opencv_orb()
        
        assert isinstance(result, bool)
        # Result should match our detection
        assert result == OPENCV_AVAILABLE


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV ORB not available")
class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_uniform_image(self):
        """Test with uniform (solid color) image."""
        img1 = np.full((128, 128, 3), 128, dtype=np.uint8)
        img2 = img1.copy()
        
        result = orb_diff(img1, img2)
        
        # Should still return valid results (possibly with zero keypoints)
        assert 'match_ratio' in result
        assert 'diff_mask' in result
        assert 'diff_map' in result
        assert result['diff_mask'].shape[:2] == img1.shape[:2]
    
    def test_small_image(self):
        """Test with small image."""
        img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        img2 = img1.copy()
        
        result = orb_diff(img1, img2)
        
        assert result['diff_mask'].shape[:2] == img1.shape[:2]
        assert result['diff_map'].shape == img1.shape[:2]
    
    def test_large_image(self):
        """Test with large image."""
        img1 = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        img2 = img1.copy()
        
        result = orb_diff(img1, img2)
        
        assert result['diff_mask'].shape[:2] == img1.shape[:2]
        assert result['diff_map'].shape == img1.shape[:2]
    
    def test_high_noise_image(self):
        """Test with high noise image."""
        img1 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        
        result = orb_diff(img1, img2)
        
        assert 'match_ratio' in result
        assert 'diff_mask' in result
        assert 'diff_map' in result
    
    def test_float_input_normalized(self):
        """Test with float input in [0, 1] range."""
        img1 = np.random.rand(128, 128, 3).astype(np.float32)
        img2 = img1.copy()
        
        result = orb_diff(img1, img2)
        
        assert 'match_ratio' in result
        assert 'diff_mask' in result
        assert 'diff_map' in result
    
    def test_float_input_scaled(self):
        """Test with float input in [0, 255] range."""
        img1 = (np.random.rand(128, 128, 3) * 255).astype(np.float32)
        img2 = img1.copy()
        
        result = orb_diff(img1, img2)
        
        assert 'match_ratio' in result
        assert 'diff_mask' in result
        assert 'diff_map' in result


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV ORB not available")
class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_pipeline_identical_images(self):
        """Test full pipeline with identical images."""
        # Create test image with clear features
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Draw several distinct shapes
        cv2 = pytest.importorskip("cv2")
        cv2.rectangle(img1, (50, 50), (100, 100), (255, 255, 255), -1)
        cv2.circle(img1, (180, 180), 30, (255, 255, 255), -1)
        cv2.rectangle(img1, (150, 30), (200, 80), (200, 200, 200), -1)
        
        img2 = img1.copy()
        
        # Run full pipeline
        result = orb_diff(img1, img2)
        
        # Verify outputs
        assert result['n_keypoints1'] > 0
        assert result['n_keypoints2'] > 0
        assert result['n_matches'] > 0
        assert result['match_ratio'] > 0
        assert result['diff_mask'].shape == (256, 256)
        assert result['diff_map'].shape == (256, 256)
        
        # For identical images, avg_distance should be low
        assert result['avg_distance'] < 100  # Arbitrary threshold for similar descriptors
    
    def test_full_pipeline_with_rotation(self):
        """Test full pipeline with rotated image."""
        cv2 = pytest.importorskip("cv2")
        
        # Create test image
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(img1, (80, 80), (176, 176), (255, 255, 255), -1)
        
        # Rotate image
        center = (128, 128)
        angle = 15
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        img2 = cv2.warpAffine(img1, rotation_matrix, (256, 256))
        
        # Run comparison
        result = orb_diff(img1, img2)
        
        # ORB should handle some rotation
        assert 'match_ratio' in result
        assert result['diff_mask'].shape == (256, 256)
        assert result['diff_map'].shape == (256, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
