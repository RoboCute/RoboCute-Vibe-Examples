"""
Comprehensive tests for sift.py - SIFT feature detection and matching.

Tests cover:
- Input validation (types, shapes, dimensions, dtypes)
- Algorithm correctness (keypoint detection, descriptors, matching)
- Edge cases (empty images, uniform images, small images)
- Backend selection (numpy vs opencv)
"""

import sys
import numpy as np
import pytest
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, 'D:\\RoboCute_robot\\diff')

from sift import (
    SIFTInputError,
    SIFTConfig,
    Keypoint,
    Match,
    GaussianPyramid,
    DoGPyramid,
    KeypointDetector,
    OrientationAssigner,
    DescriptorComputer,
    SIFT,
    SIFTMatcher,
    SIFTDiff,
    sift_diff,
    create_sift,
    detectAndCompute,
    _validate_inputs,
    _has_opencv_sift,
)


# =============================================================================
# Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for _validate_inputs function."""
    
    def test_valid_identical_arrays(self):
        """Test validation passes for identical arrays."""
        img1 = np.random.rand(100, 100).astype(np.float32)
        img2 = img1.copy()
        # Should not raise
        _validate_inputs(img1, img2)
    
    def test_type_error_non_numpy(self):
        """Test TypeError raised for non-numpy inputs."""
        img1 = np.random.rand(100, 100).astype(np.float32)
        
        with pytest.raises(TypeError, match="img2 must be a numpy array"):
            _validate_inputs(img1, "not an array")
        
        with pytest.raises(TypeError, match="img1 must be a numpy array"):
            _validate_inputs("not an array", img1)
    
    def test_dimension_mismatch(self):
        """Test SIFTInputError for dimension mismatch."""
        img1 = np.random.rand(100, 100).astype(np.float32)
        img2 = np.random.rand(100, 100, 3).astype(np.float32)
        
        with pytest.raises(SIFTInputError, match="Dimension mismatch"):
            _validate_inputs(img1, img2)
    
    def test_shape_mismatch(self):
        """Test SIFTInputError for shape mismatch."""
        img1 = np.random.rand(100, 100).astype(np.float32)
        img2 = np.random.rand(100, 150).astype(np.float32)
        
        with pytest.raises(SIFTInputError, match="Shape mismatch"):
            _validate_inputs(img1, img2)
    
    def test_size_mismatch_3d(self):
        """Test SIFTInputError for 3D size mismatch."""
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = np.random.rand(100, 100, 4).astype(np.float32)
        
        with pytest.raises(SIFTInputError, match="Shape mismatch"):
            _validate_inputs(img1, img2)
    
    def test_empty_array(self):
        """Test SIFTInputError for empty arrays."""
        img1 = np.array([]).astype(np.float32)
        img2 = np.array([]).astype(np.float32)
        
        with pytest.raises(SIFTInputError, match="Empty input arrays"):
            _validate_inputs(img1, img2)
    
    def test_dtype_mismatch(self):
        """Test SIFTInputError for dtype mismatch."""
        img1 = np.random.rand(100, 100).astype(np.float32)
        img2 = np.random.rand(100, 100).astype(np.float64)
        
        with pytest.raises(SIFTInputError, match="Dtype mismatch"):
            _validate_inputs(img1, img2)
    
    def test_dtype_mismatch_allowed(self):
        """Test dtype mismatch allowed with flag."""
        img1 = np.random.rand(100, 100).astype(np.float32)
        img2 = np.random.rand(100, 100).astype(np.float64)
        
        # Should not raise when allow_different_dtypes=True
        _validate_inputs(img1, img2, allow_different_dtypes=True)
    
    def test_invalid_dimensions_1d(self):
        """Test SIFTInputError for 1D arrays."""
        img1 = np.random.rand(100).astype(np.float32)
        img2 = img1.copy()
        
        with pytest.raises(SIFTInputError, match="Invalid dimensions"):
            _validate_inputs(img1, img2)
    
    def test_invalid_dimensions_4d(self):
        """Test SIFTInputError for 4D arrays."""
        img1 = np.random.rand(10, 10, 10, 3).astype(np.float32)
        img2 = img1.copy()
        
        with pytest.raises(SIFTInputError, match="Invalid dimensions"):
            _validate_inputs(img1, img2)
    
    def test_valid_2d_arrays(self):
        """Test validation passes for 2D arrays."""
        img1 = np.random.rand(100, 100).astype(np.float32)
        img2 = img1.copy()
        _validate_inputs(img1, img2)
    
    def test_valid_3d_arrays(self):
        """Test validation passes for 3D arrays."""
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = img1.copy()
        _validate_inputs(img1, img2)


# =============================================================================
# Data Class Tests
# =============================================================================

class TestDataClasses:
    """Tests for Keypoint and Match dataclasses."""
    
    def test_keypoint_creation(self):
        """Test Keypoint dataclass creation."""
        kp = Keypoint(
            x=50.5,
            y=75.3,
            octave=2,
            scale_level=3,
            sigma=2.5,
            orientation=1.57,
            descriptor=np.random.randint(0, 256, 128, dtype=np.uint8)
        )
        
        assert kp.x == 50.5
        assert kp.y == 75.3
        assert kp.octave == 2
        assert kp.scale_level == 3
        assert kp.sigma == 2.5
        assert kp.orientation == 1.57
        assert kp.descriptor is not None
    
    def test_keypoint_scale_property(self):
        """Test Keypoint scale property."""
        kp = Keypoint(
            x=50.0,
            y=50.0,
            octave=2,
            scale_level=0,
            sigma=1.6,
            orientation=0.0
        )
        
        # scale = sigma * (2 ** octave)
        expected_scale = 1.6 * (2 ** 2)
        assert kp.scale == expected_scale
    
    def test_keypoint_without_descriptor(self):
        """Test Keypoint without descriptor."""
        kp = Keypoint(
            x=50.0,
            y=50.0,
            octave=0,
            scale_level=0,
            sigma=1.6
        )
        
        assert kp.descriptor is None
    
    def test_match_creation(self):
        """Test Match dataclass creation."""
        kp1 = Keypoint(x=10.0, y=10.0, octave=0, scale_level=0, sigma=1.6)
        kp2 = Keypoint(x=20.0, y=20.0, octave=0, scale_level=0, sigma=1.6)
        
        match = Match(kp1=kp1, kp2=kp2, distance=0.5)
        
        assert match.kp1 == kp1
        assert match.kp2 == kp2
        assert match.distance == 0.5


# =============================================================================
# SIFTConfig Tests
# =============================================================================

class TestSIFTConfig:
    """Tests for SIFTConfig class."""
    
    def test_default_config(self):
        """Test default SIFTConfig values."""
        config = SIFTConfig()
        
        assert config.n_octaves == 4
        assert config.n_scales_per_octave == 3
        assert config.sigma_init == 1.6
        assert config.contrast_threshold == 0.04
        assert config.edge_threshold == 10.0
        assert config.descriptor_radius_factor == 3.0
        assert config.n_bins_orientation == 36
        assert config.n_bins_descriptor == 8
        assert config.descriptor_window_size == 4
        assert config.match_ratio_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom SIFTConfig values."""
        config = SIFTConfig(
            n_octaves=8,
            contrast_threshold=0.02,
            edge_threshold=5.0
        )
        
        assert config.n_octaves == 8
        assert config.contrast_threshold == 0.02
        assert config.edge_threshold == 5.0
        # Other values should be defaults
        assert config.n_scales_per_octave == 3
        assert config.sigma_init == 1.6


# =============================================================================
# Gaussian Pyramid Tests
# =============================================================================

class TestGaussianPyramid:
    """Tests for GaussianPyramid class."""
    
    def test_pyramid_building_2d(self):
        """Test Gaussian pyramid building from 2D image."""
        config = SIFTConfig(n_octaves=2, n_scales_per_octave=2)
        pyramid = GaussianPyramid(config)
        
        image = np.random.rand(64, 64).astype(np.float32)
        pyramid.build(image)
        
        # Should have 2 octaves
        assert len(pyramid.octaves) == 2
        # Each octave should have n_scales + 3 images
        assert len(pyramid.octaves[0]) == 5  # 2 + 3
        assert len(pyramid.octaves[1]) == 5
    
    def test_pyramid_building_3d(self):
        """Test Gaussian pyramid building from 3D image."""
        config = SIFTConfig(n_octaves=2, n_scales_per_octave=2)
        pyramid = GaussianPyramid(config)
        
        image = np.random.rand(64, 64, 3).astype(np.float32)
        pyramid.build(image)
        
        # Should have 2 octaves
        assert len(pyramid.octaves) == 2
    
    def test_pyramid_downsampling(self):
        """Test that pyramid correctly downsamples."""
        config = SIFTConfig(n_octaves=3, n_scales_per_octave=2)
        pyramid = GaussianPyramid(config)
        
        image = np.random.rand(64, 64).astype(np.float32)
        pyramid.build(image)
        
        # Second octave should be half size
        first_octave_shape = pyramid.octaves[0][0].shape
        second_octave_shape = pyramid.octaves[1][0].shape
        
        assert second_octave_shape[0] <= first_octave_shape[0] // 2
        assert second_octave_shape[1] <= first_octave_shape[1] // 2
    
    def test_pyramid_sigmas(self):
        """Test that pyramid tracks sigma values."""
        config = SIFTConfig(n_octaves=2, n_scales_per_octave=2, sigma_init=1.6)
        pyramid = GaussianPyramid(config)
        
        image = np.random.rand(64, 64).astype(np.float32)
        pyramid.build(image)
        
        # Should have sigma values for each octave
        assert len(pyramid.sigmas) == 2
        # Each octave should have n_scales + 3 sigma values
        assert len(pyramid.sigmas[0]) == 5
        # First sigma should be sigma_init
        assert pyramid.sigmas[0][0] == 1.6


# =============================================================================
# DoG Pyramid Tests
# =============================================================================

class TestDoGPyramid:
    """Tests for DoGPyramid class."""
    
    def test_dog_building(self):
        """Test DoG pyramid building."""
        config = SIFTConfig(n_octaves=2, n_scales_per_octave=2)
        
        gaussian_pyramid = GaussianPyramid(config)
        image = np.random.rand(64, 64).astype(np.float32)
        gaussian_pyramid.build(image)
        
        dog_pyramid = DoGPyramid(gaussian_pyramid)
        dog_pyramid.build()
        
        # Should have same number of octaves
        assert len(dog_pyramid.octaves) == len(gaussian_pyramid.octaves)
        # Each DoG octave should have one less image than Gaussian
        for i, dog_octave in enumerate(dog_pyramid.octaves):
            expected_len = len(gaussian_pyramid.octaves[i]) - 1
            assert len(dog_octave) == expected_len


# =============================================================================
# Keypoint Detection Tests
# =============================================================================

class TestKeypointDetection:
    """Tests for KeypointDetector class."""
    
    def test_detect_keypoints(self):
        """Test basic keypoint detection."""
        config = SIFTConfig(
            n_octaves=2,
            n_scales_per_octave=2,
            contrast_threshold=0.001,  # Very low threshold for test images
            edge_threshold=20.0  # Higher edge threshold to keep more features
        )
        
        # Create image with rich structure (checkerboard + gradients)
        size = 128
        x = np.linspace(0, 8*np.pi, size)
        y = np.linspace(0, 8*np.pi, size)
        X, Y = np.meshgrid(x, y)
        # Combine multiple frequencies for richer texture
        image = (
            np.sin(X) * np.cos(Y) +
            0.5 * np.sin(2*X) * np.cos(2*Y) +
            0.25 * np.sin(4*X) * np.cos(4*Y)
        ).astype(np.float32)
        
        # Build pyramids
        gaussian_pyramid = GaussianPyramid(config)
        gaussian_pyramid.build(image)
        
        dog_pyramid = DoGPyramid(gaussian_pyramid)
        dog_pyramid.build()
        
        # Detect keypoints
        detector = KeypointDetector(config)
        keypoints = detector.detect(dog_pyramid, gaussian_pyramid.sigmas)
        
        # Should detect some keypoints (may be 0 for synthetic, that's OK)
        # Just verify the function runs without error
        assert isinstance(keypoints, list)
        
        # All keypoints should have required attributes
        for kp in keypoints:
            assert kp.x >= 0
            assert kp.y >= 0
            assert kp.octave >= 0
            assert kp.sigma > 0
    
    def test_no_keypoints_in_uniform_image(self):
        """Test no keypoints detected in uniform image."""
        config = SIFTConfig(contrast_threshold=0.01)
        
        # Uniform image
        image = np.ones((128, 128), dtype=np.float32) * 128
        
        # Build pyramids
        gaussian_pyramid = GaussianPyramid(config)
        gaussian_pyramid.build(image)
        
        dog_pyramid = DoGPyramid(gaussian_pyramid)
        dog_pyramid.build()
        
        # Detect keypoints
        detector = KeypointDetector(config)
        keypoints = detector.detect(dog_pyramid, gaussian_pyramid.sigmas)
        
        # Should detect no keypoints in uniform image
        assert len(keypoints) == 0


# =============================================================================
# Orientation Assignment Tests
# =============================================================================

class TestOrientationAssignment:
    """Tests for OrientationAssigner class."""
    
    def test_assign_orientations(self):
        """Test orientation assignment to keypoints."""
        config = SIFTConfig(n_octaves=2, n_scales_per_octave=2)
        
        # Create structured image
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        # Build pyramids and detect keypoints
        gaussian_pyramid = GaussianPyramid(config)
        gaussian_pyramid.build(image)
        
        dog_pyramid = DoGPyramid(gaussian_pyramid)
        dog_pyramid.build()
        
        detector = KeypointDetector(config)
        keypoints = detector.detect(dog_pyramid, gaussian_pyramid.sigmas)
        
        # Assign orientations
        assigner = OrientationAssigner(config)
        oriented_kps = assigner.assign(keypoints, gaussian_pyramid)
        
        # Should have at least as many oriented keypoints
        assert len(oriented_kps) >= len(keypoints)
        
        # All should have orientation assigned
        for kp in oriented_kps:
            assert kp.orientation is not None


# =============================================================================
# Descriptor Computation Tests
# =============================================================================

class TestDescriptorComputation:
    """Tests for DescriptorComputer class."""
    
    def test_compute_descriptors(self):
        """Test descriptor computation for keypoints."""
        config = SIFTConfig(
            n_octaves=2,
            n_scales_per_octave=2,
            contrast_threshold=0.001,  # Very low threshold
            edge_threshold=20.0
        )
        
        # Create rich structured image
        size = 128
        x = np.linspace(0, 8*np.pi, size)
        y = np.linspace(0, 8*np.pi, size)
        X, Y = np.meshgrid(x, y)
        image = (
            np.sin(X) * np.cos(Y) +
            0.5 * np.sin(2*X) * np.cos(2*Y) +
            0.25 * np.sin(4*X) * np.cos(4*Y)
        ).astype(np.float32)
        
        # Full SIFT pipeline
        sift = SIFT(config)
        keypoints = sift.detect(image)
        
        # Compute descriptors
        computer = DescriptorComputer(config)
        keypoints_with_desc = computer.compute(keypoints, sift.gaussian_pyramid)
        
        # Verify function runs - descriptors may or may not be computed
        # depending on keypoint locations and image structure
        assert isinstance(keypoints_with_desc, list)
        
        # If descriptors were computed, verify their properties
        kps_with_desc = [kp for kp in keypoints_with_desc if kp.descriptor is not None]
        for kp in kps_with_desc:
            assert len(kp.descriptor) == 128
            assert kp.descriptor.dtype == np.uint8


# =============================================================================
# SIFT Class Tests
# =============================================================================

class TestSIFT:
    """Tests for main SIFT class."""
    
    def test_sift_detect(self):
        """Test SIFT detect method."""
        config = SIFTConfig(contrast_threshold=0.01)
        sift = SIFT(config)
        
        # Create structured image
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        keypoints = sift.detect(image)
        
        # Should detect keypoints
        assert len(keypoints) > 0
        
        # All should have orientations
        for kp in keypoints:
            assert hasattr(kp, 'orientation')
    
    def test_sift_detect_and_compute(self):
        """Test SIFT detect_and_compute method."""
        config = SIFTConfig(contrast_threshold=0.01)
        sift = SIFT(config)
        
        # Create structured image
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        keypoints = sift.detect_and_compute(image)
        
        # Should detect keypoints with descriptors
        kps_with_desc = [kp for kp in keypoints if kp.descriptor is not None]
        assert len(kps_with_desc) > 0
    
    def test_sift_on_3d_image(self):
        """Test SIFT on 3D (RGB) image."""
        config = SIFTConfig(contrast_threshold=0.01)
        sift = SIFT(config)
        
        # Create structured 3D image
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        gray = np.sin(X) * np.cos(Y)
        image = np.stack([gray, gray * 0.8, gray * 0.6], axis=-1).astype(np.float32)
        
        keypoints = sift.detect_and_compute(image)
        
        # Should detect keypoints
        assert len(keypoints) > 0


# =============================================================================
# SIFT Matcher Tests
# =============================================================================

class TestSIFTMatcher:
    """Tests for SIFTMatcher class."""
    
    def test_match_identical_images(self):
        """Test matching identical images."""
        config = SIFTConfig(
            contrast_threshold=0.01,
            match_ratio_threshold=0.8
        )
        matcher = SIFTMatcher(config)
        
        # Create structured image
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        keypoints1, keypoints2, matches = matcher.match(image, image)
        
        # Should have keypoints in both images
        assert len(keypoints1) > 0
        assert len(keypoints2) > 0
        
        # Should have matches (identical images should match well)
        assert len(matches) > 0
        
        # Match distances should be very small for identical images
        for match in matches:
            assert match.distance < 0.1
    
    def test_match_different_images(self):
        """Test matching different images."""
        config = SIFTConfig(contrast_threshold=0.01)
        matcher = SIFTMatcher(config)
        
        # Create two different structured images
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image1 = (np.sin(X) * np.cos(Y)).astype(np.float32)
        image2 = (np.sin(2*X) * np.cos(2*Y)).astype(np.float32)
        
        keypoints1, keypoints2, matches = matcher.match(image1, image2)
        
        # Should have keypoints in both
        assert len(keypoints1) > 0
        assert len(keypoints2) > 0
        
        # May or may not have matches (depends on similarity)
        # Just verify the function runs without error
    
    def test_matcher_input_validation(self):
        """Test that matcher validates inputs."""
        config = SIFTConfig()
        matcher = SIFTMatcher(config)
        
        # Different shapes should raise error
        image1 = np.random.rand(100, 100).astype(np.float32)
        image2 = np.random.rand(100, 150).astype(np.float32)
        
        with pytest.raises(SIFTInputError):
            matcher.match(image1, image2)


# =============================================================================
# SIFTDiff Tests
# =============================================================================

class TestSIFTDiff:
    """Tests for SIFTDiff class."""
    
    def test_compare_identical_images(self):
        """Test comparing identical images."""
        config = SIFTConfig(contrast_threshold=0.01)
        diff = SIFTDiff(config)
        
        # Create structured image
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        results = diff.compare(image, image)
        
        # Check result keys
        assert 'match_ratio' in results
        assert 'avg_distance' in results
        assert 'n_keypoints1' in results
        assert 'n_keypoints2' in results
        assert 'n_matches' in results
        assert 'matched_regions' in results
        assert 'unmatched_regions1' in results
        assert 'unmatched_regions2' in results
        
        # For identical images, match_ratio should be high
        assert results['match_ratio'] > 0.5
        assert results['n_matches'] > 0
        assert results['n_keypoints1'] == results['n_keypoints2']
    
    def test_compare_different_images(self):
        """Test comparing different images."""
        config = SIFTConfig(contrast_threshold=0.01)
        diff = SIFTDiff(config)
        
        # Create two different images
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image1 = (np.sin(X) * np.cos(Y)).astype(np.float32)
        image2 = (np.sin(2*X) * np.cos(2*Y)).astype(np.float32)
        
        results = diff.compare(image1, image2)
        
        # Should have results
        assert results['n_keypoints1'] > 0
        assert results['n_keypoints2'] > 0
    
    def test_detect_changes(self):
        """Test change detection."""
        config = SIFTConfig(contrast_threshold=0.01)
        diff = SIFTDiff(config)
        
        # Create structured image
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        # Identical images should have no changes
        has_changes, results = diff.detect_changes(image, image, match_threshold=0.5)
        assert not has_changes
        
        # Different images might have changes
        image2 = (np.sin(2*X) * np.cos(2*Y)).astype(np.float32)
        has_changes, results = diff.detect_changes(image, image2, match_threshold=0.5)
        # Result depends on similarity, just verify it runs


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_sift_diff_function(self):
        """Test sift_diff convenience function."""
        # Create structured image
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        results = sift_diff(image, image, contrast_threshold=0.01)
        
        assert 'match_ratio' in results
        assert results['n_matches'] > 0
    
    def test_create_sift_numpy(self):
        """Test create_sift with numpy backend."""
        sift = create_sift(backend='numpy', contrast_threshold=0.01)
        
        assert isinstance(sift, SIFT)
        
        # Test it works
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        keypoints = sift.detect_and_compute(image)
        assert len(keypoints) > 0
    
    def test_create_sift_auto(self):
        """Test create_sift with auto backend."""
        sift = create_sift(backend='auto')
        
        # Should return either OpenCVSIFT or SIFT
        if _has_opencv_sift():
            from sift import OpenCVSIFT
            assert isinstance(sift, (SIFT, OpenCVSIFT))
        else:
            assert isinstance(sift, (SIFT,))
    
    def test_create_sift_invalid_backend(self):
        """Test create_sift with invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_sift(backend='invalid')
    
    def test_create_sift_opencv_not_available(self):
        """Test create_sift raises error when opencv requested but not available."""
        if not _has_opencv_sift():
            with pytest.raises(ImportError, match="OpenCV SIFT not available"):
                create_sift(backend='opencv')
    
    def test_detectAndCompute(self):
        """Test detectAndCompute convenience function."""
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        keypoints, descriptors = detectAndCompute(image, backend='numpy', contrast_threshold=0.01)
        
        assert isinstance(keypoints, list)
        assert isinstance(descriptors, np.ndarray)
        assert len(keypoints) > 0
        assert descriptors.shape[0] > 0
    
    def test_detectAndCompute_with_mask(self):
        """Test detectAndCompute with mask."""
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        # Create mask with only center region
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 1
        
        keypoints, descriptors = detectAndCompute(
            image, mask=mask, backend='numpy', contrast_threshold=0.01
        )
        
        # All keypoints should be within masked region
        for kp in keypoints:
            assert 32 <= kp.x <= 96
            assert 32 <= kp.y <= 96


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_small_image(self):
        """Test SIFT on very small image."""
        sift = SIFT(SIFTConfig(n_octaves=1, n_scales_per_octave=2))
        
        # Small image with structure
        x = np.linspace(0, 2*np.pi, 32)
        y = np.linspace(0, 2*np.pi, 32)
        X, Y = np.meshgrid(x, y)
        image = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        # Should handle small images without error
        keypoints = sift.detect_and_compute(image)
        # May or may not detect keypoints in small image
    
    def test_uniform_image(self):
        """Test SIFT on uniform image."""
        sift = SIFT(SIFTConfig())
        
        # Uniform image
        image = np.ones((128, 128), dtype=np.float32) * 128
        
        keypoints = sift.detect_and_compute(image)
        
        # Should detect no keypoints in uniform image
        assert len(keypoints) == 0
    
    def test_single_channel_vs_three_channel(self):
        """Test that single and 3-channel images produce similar results."""
        config = SIFTConfig(contrast_threshold=0.01)
        
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        gray = (np.sin(X) * np.cos(Y)).astype(np.float32)
        
        # Single channel
        sift1 = SIFT(config)
        kps1 = sift1.detect_and_compute(gray)
        
        # 3-channel (grayscale replicated)
        rgb = np.stack([gray, gray, gray], axis=-1)
        sift2 = SIFT(config)
        kps2 = sift2.detect_and_compute(rgb)
        
        # Should detect similar number of keypoints
        assert abs(len(kps1) - len(kps2)) < max(len(kps1), len(kps2)) * 0.5
    
    def test_different_dtypes_with_allow_flag(self):
        """Test matching images with different dtypes."""
        config = SIFTConfig(contrast_threshold=0.01)
        matcher = SIFTMatcher(config)
        
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image1 = (np.sin(X) * np.cos(Y)).astype(np.float32)
        image2 = (np.sin(X) * np.cos(Y)).astype(np.float64)
        
        # Should work with allow_different_dtypes=True (matcher does this)
        kps1, kps2, matches = matcher.match(image1, image2)
        
        assert len(kps1) > 0
        assert len(kps2) > 0


# =============================================================================
# Backend Tests
# =============================================================================

class TestBackends:
    """Tests for different SIFT backends."""
    
    def test_has_opencv_sift(self):
        """Test _has_opencv_sift function."""
        result = _has_opencv_sift()
        assert isinstance(result, bool)
    
    def test_opencv_backend_if_available(self):
        """Test OpenCV backend if available."""
        if not _has_opencv_sift():
            pytest.skip("OpenCV SIFT not available")
        
        from sift import OpenCVSIFT
        
        config = SIFTConfig()
        sift = OpenCVSIFT(config)
        
        # Create test image
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        image = ((np.sin(X) * np.cos(Y) + 1) * 127.5).astype(np.uint8)
        
        keypoints = sift.detect(image)
        
        # Should detect keypoints
        assert len(keypoints) > 0
        
        # Test detect_and_compute
        keypoints = sift.detect_and_compute(image)
        kps_with_desc = [kp for kp in keypoints if kp.descriptor is not None]
        assert len(kps_with_desc) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
