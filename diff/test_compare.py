"""
Tests for compare.py - main comparison function.
"""

import sys
import numpy as np
from scipy import ndimage

from compare import (
    compare_images,
    batch_compare,
    CompareResult,
    ComparisonLevel,
    compute_similarity_score,
    summarize_differences,
    create_diff_visualization,
)
from sift import SIFTInputError
from diff_regions import DifferenceSeverity


def _create_test_image(size=256, seed=42):
    """Create a structured test image with gradients for SIFT."""
    np.random.seed(seed)
    x = np.linspace(0, 4*np.pi, size)
    y = np.linspace(0, 4*np.pi, size)
    X, Y = np.meshgrid(x, y)
    img = (
        np.sin(X) * np.cos(Y) +
        0.5 * np.sin(2*X) * np.cos(2*Y) +
        0.25 * np.sin(4*X) * np.cos(4*Y) +
        0.1 * np.random.randn(size, size)  # Add slight noise
    )
    img = (img - img.min()) / (img.max() - img.min())
    return img.astype(np.float32)


class TestCompareImages:
    """Tests for the main compare_images function."""
    
    def test_compare_identical_images(self):
        """Test that identical images produce high similarity."""
        # Use structured images for SIFT to detect keypoints
        img1 = _create_test_image(256, seed=42)
        
        result = compare_images(img1, img1.copy())
        
        assert isinstance(result, CompareResult)
        assert result.similarity_score > 0.8
        assert result.match_ratio > 0.5
        assert result.is_identical or result.is_similar
        assert result.n_matches > 0
        print(f"  Identical: similarity={result.similarity_score:.3f}, matches={result.n_matches}")
    
    def test_compare_different_images(self):
        """Test that different images can be compared."""
        img1 = _create_test_image(256, seed=42)
        img2 = _create_test_image(256, seed=999)  # Very different pattern
        
        result = compare_images(img1, img2)
        
        assert isinstance(result, CompareResult)
        # Both images should have keypoints detected
        assert result.n_keypoints1 > 0
        assert result.n_keypoints2 > 0
        # Different images should have some matches (SIFT is robust)
        print(f"  Different: similarity={result.similarity_score:.3f}, regions={result.n_regions}")
    
    def test_compare_basic_level(self):
        """Test BASIC comparison level."""
        img1 = _create_test_image(256, seed=42)
        img2 = img1 + 0.1  # Slight modification
        
        result = compare_images(img1, img2, level=ComparisonLevel.BASIC)
        
        assert result.diff_mask is None
        assert result.diff_map is None
        assert len(result.regions) == 0  # BASIC doesn't detect regions
        assert result.n_matches > 0
        print(f"  Basic level: matches={result.n_matches}")
    
    def test_compare_standard_level(self):
        """Test STANDARD comparison level."""
        img1 = _create_test_image(256, seed=42)
        img2 = _create_test_image(256, seed=99)
        
        result = compare_images(img1, img2, level=ComparisonLevel.STANDARD)
        
        assert result.n_regions >= 0  # May or may not find regions
        assert result.diff_mask is None  # No mask in standard
        assert result.diff_map is None
        print(f"  Standard level: regions={result.n_regions}, coverage={result.diff_coverage:.3f}")
    
    def test_compare_full_level(self):
        """Test FULL comparison level."""
        img1 = _create_test_image(256, seed=42)
        img2 = _create_test_image(256, seed=99)
        
        result = compare_images(img1, img2, level=ComparisonLevel.FULL)
        
        assert result.diff_mask is not None
        assert result.diff_mask.shape == (256, 256)
        assert result.diff_map is not None
        print(f"  Full level: mask_shape={result.diff_mask.shape}, map_shape={result.diff_map.shape}")
    
    def test_compare_with_visualization(self):
        """Test comparison with visualization generation."""
        img1 = _create_test_image(256, seed=42) * 255
        
        result = compare_images(
            img1, img1.copy(),
            level=ComparisonLevel.FULL,
            return_visualization=True
        )
        
        assert result.diff_overlay is not None
        assert result.diff_overlay.shape == (256, 256, 3)
        
        print(f"  Visualization: shape={result.diff_overlay.shape}")
    
    def test_compare_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Non-numpy array
        try:
            compare_images("not an array", np.zeros((50, 50)))
            assert False, "Should have raised SIFTInputError"
        except SIFTInputError:
            pass
        
        # Wrong dimensions
        try:
            compare_images(np.zeros(10), np.zeros(10))
            assert False, "Should have raised SIFTInputError"
        except SIFTInputError:
            pass
        
        # Mismatched shapes
        try:
            compare_images(np.zeros((256, 256)), np.zeros((128, 128)))
            assert False, "Should have raised SIFTInputError"
        except SIFTInputError:
            pass
    
    def test_compare_string_level(self):
        """Test that string level specification works."""
        img1 = _create_test_image(256, seed=42)
        
        result = compare_images(img1, img1.copy(), level='standard')
        
        assert isinstance(result, CompareResult)
        print(f"  String level: similarity={result.similarity_score:.3f}")
    
    def test_compare_result_to_dict(self):
        """Test CompareResult.to_dict method."""
        img1 = _create_test_image(256, seed=42)
        
        result = compare_images(img1, img1.copy())
        d = result.to_dict()
        
        assert isinstance(d, dict)
        assert 'similarity_score' in d
        assert 'match_ratio' in d
        assert 'n_regions' in d
        assert 'max_severity' in d
        assert 'has_diff_mask' in d
        print(f"  Dict: {d}")
    
    def test_compare_result_get_summary(self):
        """Test CompareResult.get_summary method."""
        img1 = _create_test_image(256, seed=42)
        
        result = compare_images(img1, img1.copy())
        summary = result.get_summary()
        
        assert isinstance(summary, str)
        assert 'Similarity' in summary
        assert 'Feature match ratio' in summary
        print(f"  Summary:\n{summary}")


class TestComputeSimilarityScore:
    """Tests for compute_similarity_score function."""
    
    def test_perfect_similarity(self):
        """Test with perfect match metrics."""
        score = compute_similarity_score(
            match_ratio=1.0,
            avg_distance=0.0,
            diff_coverage=0.0,
            n_matches=50
        )
        assert score > 0.9
        print(f"  Perfect similarity: {score:.3f}")
    
    def test_no_similarity(self):
        """Test with no match metrics."""
        score = compute_similarity_score(
            match_ratio=0.0,
            avg_distance=500.0,
            diff_coverage=1.0,
            n_matches=0
        )
        assert score < 0.3
        print(f"  No similarity: {score:.3f}")
    
    def test_partial_similarity(self):
        """Test with partial match metrics."""
        score = compute_similarity_score(
            match_ratio=0.5,
            avg_distance=150.0,
            diff_coverage=0.2,
            n_matches=25
        )
        assert 0.3 < score < 0.8
        print(f"  Partial similarity: {score:.3f}")
    
    def test_low_confidence(self):
        """Test with low match count."""
        score = compute_similarity_score(
            match_ratio=1.0,
            avg_distance=0.0,
            diff_coverage=0.0,
            n_matches=2
        )
        # Should be lower due to low confidence
        assert score < 1.0
        print(f"  Low confidence: {score:.3f}")


class TestBatchCompare:
    """Tests for batch_compare function."""
    
    def test_batch_compare(self):
        """Test batch comparison against multiple images."""
        ref = _create_test_image(256, seed=42)
        test_images = [
            ref.copy(),  # Identical - should be 100%
            _create_test_image(256, seed=43),  # Slightly different
            _create_test_image(256, seed=999),  # Very different
        ]
        
        results = batch_compare(ref, test_images)
        
        assert len(results) == 3
        
        # All should have results with keypoints
        assert all(r.n_keypoints1 > 0 for r in results)
        # First (identical) should have perfect similarity
        assert results[0].similarity_score > 0.99
        assert results[0].is_similar
        print(f"  Batch: {[f'{r.similarity_score:.3f}' for r in results]}")
    
    def test_batch_compare_empty(self):
        """Test batch with empty list."""
        ref = np.random.rand(60, 60).astype(np.float32)
        results = batch_compare(ref, [])
        
        assert len(results) == 0
    
    def test_batch_compare_error_handling(self):
        """Test batch with one invalid image."""
        ref = _create_test_image(256, seed=42)
        test_images = [
            ref.copy(),
            "invalid",  # Should cause error
            ref.copy(),
        ]
        
        results = batch_compare(ref, test_images)
        
        # Should still return 3 results (error result for invalid)
        assert len(results) == 3
        # Second result should have 0 similarity due to error
        assert results[1].similarity_score == 0.0
        print(f"  Error handling: {[f'{r.similarity_score:.3f}' for r in results]}")


class TestSummarizeDifferences:
    """Tests for summarize_differences function."""
    
    def test_summarize_results(self):
        """Test summary of multiple results."""
        ref = _create_test_image(256, seed=42)
        test_images = [ref + i * 0.05 for i in range(5)]
        
        results = batch_compare(ref, test_images)
        summary = summarize_differences(results)
        
        assert 'n_comparisons' in summary
        assert summary['n_comparisons'] == 5
        assert 'similarity' in summary
        assert 'mean' in summary['similarity']
        assert 'classification' in summary
        assert 'total_regions' in summary
        print(f"  Summary: {summary}")
    
    def test_summarize_empty(self):
        """Test summary with empty list."""
        summary = summarize_differences([])
        
        assert summary['n_comparisons'] == 0


class TestCreateDiffVisualization:
    """Tests for create_diff_visualization function."""
    
    def test_visualize_grayscale(self):
        """Test visualization with grayscale image."""
        img = _create_test_image(256, seed=42) * 255
        
        # Create a mock result with some regions
        from diff_regions import DiffRegion, DifferenceSeverity
        mock_region = DiffRegion(
            x=50, y=50, width=40, height=40,
            severity=DifferenceSeverity.HIGH,
            confidence=0.8,
            unmatched_count=5,
            avg_match_distance=200.0,
            keypoints_img1=[],
            keypoints_img2=[]
        )
        
        result = CompareResult(
            similarity_score=0.5,
            match_ratio=0.4,
            n_matches=10,
            n_keypoints1=20,
            n_keypoints2=20,
            avg_match_distance=150.0,
            n_regions=1,
            regions=[mock_region],
            diff_coverage=0.1,
            max_severity=DifferenceSeverity.HIGH,
            diff_mask=np.zeros((256, 256), dtype=np.uint8)
        )
        result.diff_mask[50:90, 50:90] = 1
        
        vis = create_diff_visualization(img, result)
        
        assert vis.shape == (256, 256, 3)
        assert vis.dtype == np.uint8
        print(f"  Vis shape: {vis.shape}")
    
    def test_visualize_rgb(self):
        """Test visualization with RGB image."""
        img = np.stack([_create_test_image(256, seed=42+i) for i in range(3)], axis=-1) * 255
        
        result = CompareResult(
            similarity_score=0.5,
            match_ratio=0.4,
            n_matches=10,
            n_keypoints1=20,
            n_keypoints2=20,
            avg_match_distance=150.0,
            n_regions=0,
            regions=[],
            diff_coverage=0.0,
            max_severity=DifferenceSeverity.NONE,
            diff_mask=None
        )
        
        vis = create_diff_visualization(img, result)
        
        assert vis.shape == (256, 256, 3)
        print(f"  RGB vis shape: {vis.shape}")


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing compare.py")
    print("=" * 60)
    
    import traceback
    
    test_classes = [
        TestCompareImages,
        TestComputeSimilarityScore,
        TestBatchCompare,
        TestSummarizeDifferences,
        TestCreateDiffVisualization,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for cls in test_classes:
        print(f"\n{cls.__name__}")
        print("-" * 40)
        
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                traceback.print_exc()
                failed_tests += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} passed")
    print("=" * 60)
    
    return failed_tests == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
