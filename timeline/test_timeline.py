"""
Comprehensive test suite for timeline module.
Tests KeyFrame, Interpolation, and Timeline classes.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import directly from the module files
from timeline.keyframe import KeyFrame
from timeline.interpolation import Interpolation, InterpolationType
from timeline.timeline import Timeline


class TestKeyFrame(unittest.TestCase):
    """Test cases for KeyFrame class."""
    
    def test_init_basic(self):
        """Test basic KeyFrame initialization."""
        kf = KeyFrame(time=0.0, data=np.array([1.0, 2.0, 3.0]))
        self.assertEqual(kf.time, 0.0)
        np.testing.assert_array_equal(kf.data, np.array([1.0, 2.0, 3.0]))
        self.assertIsNone(kf.event)
    
    def test_init_with_event(self):
        """Test KeyFrame initialization with event callback."""
        event_called = [False]
        def callback():
            event_called[0] = True
        
        kf = KeyFrame(time=1.0, data=np.array([4.0, 5.0]), event=callback)
        self.assertEqual(kf.time, 1.0)
        self.assertIsNotNone(kf.event)
        
        kf.execute_event()
        self.assertTrue(event_called[0])
    
    def test_data_conversion(self):
        """Test that data is converted to numpy array with float64 dtype."""
        kf = KeyFrame(time=0.0, data=[1, 2, 3])  # List input
        self.assertIsInstance(kf.data, np.ndarray)
        self.assertEqual(kf.data.dtype, np.float64)
    
    def test_repr(self):
        """Test string representation."""
        kf = KeyFrame(time=0.5, data=np.array([1.0]))
        repr_str = repr(kf)
        self.assertIn("KeyFrame", repr_str)
        self.assertIn("time=0.5", repr_str)
    
    def test_equality(self):
        """Test KeyFrame equality comparison."""
        kf1 = KeyFrame(time=0.0, data=np.array([1.0, 2.0]))
        kf2 = KeyFrame(time=0.0, data=np.array([1.0, 2.0]))
        kf3 = KeyFrame(time=1.0, data=np.array([1.0, 2.0]))
        
        self.assertEqual(kf1, kf2)
        self.assertNotEqual(kf1, kf3)
        self.assertNotEqual(kf1, "not a keyframe")
    
    def test_get_data_returns_copy(self):
        """Test that get_data returns a copy, not a reference."""
        kf = KeyFrame(time=0.0, data=np.array([1.0, 2.0]))
        data_copy = kf.get_data()
        data_copy[0] = 999.0
        
        self.assertEqual(kf.data[0], 1.0)  # Original should be unchanged
    
    def test_set_data(self):
        """Test set_data method."""
        kf = KeyFrame(time=0.0, data=np.array([1.0, 2.0]))
        kf.set_data(np.array([3.0, 4.0, 5.0]))
        
        np.testing.assert_array_equal(kf.data, np.array([3.0, 4.0, 5.0]))
        self.assertEqual(kf.data.dtype, np.float64)


class TestInterpolation(unittest.TestCase):
    """Test cases for Interpolation class."""
    
    def test_init_linear(self):
        """Test linear interpolation initialization."""
        interp = Interpolation(interp_type=InterpolationType.LINEAR)
        self.assertEqual(interp.interp_type, InterpolationType.LINEAR)
        self.assertIsNone(interp.control_points)
    
    def test_init_bezier(self):
        """Test Bezier interpolation initialization."""
        cp = np.array([[1.0, 2.0], [3.0, 4.0]])
        interp = Interpolation(
            interp_type=InterpolationType.BEZIER,
            control_points=cp
        )
        self.assertEqual(interp.interp_type, InterpolationType.BEZIER)
        np.testing.assert_array_equal(interp.control_points, cp)
    
    def test_linear_interpolation_bounds(self):
        """Test linear interpolation at bounds."""
        interp = Interpolation(interp_type=InterpolationType.LINEAR)
        start = np.array([0.0, 0.0])
        end = np.array([10.0, 10.0])
        
        # At t=0
        result = interp.interpolate(start, end, 0.0)
        np.testing.assert_array_almost_equal(result, start)
        
        # At t=1
        result = interp.interpolate(start, end, 1.0)
        np.testing.assert_array_almost_equal(result, end)
    
    def test_linear_interpolation_midpoint(self):
        """Test linear interpolation at midpoint."""
        interp = Interpolation(interp_type=InterpolationType.LINEAR)
        start = np.array([0.0, 0.0])
        end = np.array([10.0, 20.0])
        
        result = interp.interpolate(start, end, 0.5)
        expected = np.array([5.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_linear_interpolation_clamping(self):
        """Test that t parameter is clamped to [0, 1]."""
        interp = Interpolation(interp_type=InterpolationType.LINEAR)
        start = np.array([0.0])
        end = np.array([10.0])
        
        # t < 0 should be clamped to 0
        result = interp.interpolate(start, end, -0.5)
        np.testing.assert_array_almost_equal(result, start)
        
        # t > 1 should be clamped to 1
        result = interp.interpolate(start, end, 1.5)
        np.testing.assert_array_almost_equal(result, end)
    
    def test_bezier_interpolation_bounds(self):
        """Test Bezier interpolation at bounds."""
        interp = Interpolation(interp_type=InterpolationType.BEZIER)
        start = np.array([0.0, 0.0])
        end = np.array([10.0, 10.0])
        
        # At t=0
        result = interp.interpolate(start, end, 0.0)
        np.testing.assert_array_almost_equal(result, start)
        
        # At t=1
        result = interp.interpolate(start, end, 1.0)
        np.testing.assert_array_almost_equal(result, end)
    
    def test_bezier_with_control_points(self):
        """Test Bezier interpolation with custom control points."""
        cp = np.array([[2.0], [8.0]])  # P1 and P2
        interp = Interpolation(
            interp_type=InterpolationType.BEZIER,
            control_points=cp
        )
        start = np.array([0.0])
        end = np.array([10.0])
        
        # At t=0.5, Bezier with these control points
        result = interp.interpolate(start, end, 0.5)
        # B(0.5) = 0.125*P0 + 0.375*P1 + 0.375*P2 + 0.125*P3
        # = 0.125*0 + 0.375*2 + 0.375*8 + 0.125*10 = 0 + 0.75 + 3 + 1.25 = 5.0
        self.assertAlmostEqual(result[0], 5.0)
    
    def test_sample_method(self):
        """Test the sample method for generating multiple samples."""
        interp = Interpolation(interp_type=InterpolationType.LINEAR)
        start = np.array([0.0, 0.0])
        end = np.array([10.0, 10.0])
        
        samples = interp.sample(start, end, 5)
        self.assertEqual(samples.shape, (5, 2))
        
        # First sample should be start
        np.testing.assert_array_almost_equal(samples[0], start)
        # Last sample should be end
        np.testing.assert_array_almost_equal(samples[-1], end)
        # Middle sample should be midpoint
        np.testing.assert_array_almost_equal(samples[2], np.array([5.0, 5.0]))


class TestTimeline(unittest.TestCase):
    """Test cases for Timeline class."""
    
    def test_empty_timeline(self):
        """Test empty timeline initialization."""
        timeline = Timeline()
        self.assertEqual(len(timeline), 0)
        self.assertEqual(timeline.get_keyframes(), [])
        self.assertEqual(timeline.get_interpolations(), [])
    
    def test_add_single_keyframe(self):
        """Test adding a single keyframe."""
        timeline = Timeline()
        kf = KeyFrame(time=0.0, data=np.array([1.0, 2.0]))
        timeline.add_keyframe(kf)
        
        self.assertEqual(len(timeline), 1)
        self.assertEqual(len(timeline.get_keyframes()), 1)
        self.assertEqual(len(timeline.get_interpolations()), 0)
    
    def test_add_multiple_keyframes(self):
        """Test adding multiple keyframes (auto-interpolation)."""
        timeline = Timeline()
        kf1 = KeyFrame(time=0.0, data=np.array([0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([10.0]))
        
        timeline.add_keyframe(kf1)
        timeline.add_keyframe(kf2)
        
        # Should have: KF + Interp + KF = 3 elements
        self.assertEqual(len(timeline), 3)
        self.assertEqual(len(timeline.get_keyframes()), 2)
        self.assertEqual(len(timeline.get_interpolations()), 1)
    
    def test_add_custom_interpolation(self):
        """Test adding custom interpolation between keyframes."""
        timeline = Timeline()
        kf1 = KeyFrame(time=0.0, data=np.array([0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([10.0]))
        interp = Interpolation(interp_type=InterpolationType.BEZIER)
        
        timeline.add_keyframe(kf1)
        timeline.add_interpolation(interp)
        timeline.add_keyframe(kf2)
        
        self.assertEqual(len(timeline), 3)
        self.assertEqual(timeline.get_interpolations()[0].interp_type, InterpolationType.BEZIER)
    
    def test_add_interpolation_to_empty_fails(self):
        """Test that adding interpolation to empty timeline raises error."""
        timeline = Timeline()
        interp = Interpolation()
        
        with self.assertRaises(ValueError) as context:
            timeline.add_interpolation(interp)
        self.assertIn("empty", str(context.exception))
    
    def test_add_interpolation_after_interpolation_fails(self):
        """Test that adding two interpolations in a row raises error."""
        timeline = Timeline()
        kf1 = KeyFrame(time=0.0, data=np.array([0.0]))
        interp1 = Interpolation()
        interp2 = Interpolation()
        
        timeline.add_keyframe(kf1)
        timeline.add_interpolation(interp1)
        
        with self.assertRaises(ValueError) as context:
            timeline.add_interpolation(interp2)
        self.assertIn("Cannot add Interpolation after Interpolation", str(context.exception))
    
    def test_structure_validation(self):
        """Test that structure validation works correctly."""
        timeline = Timeline()
        
        # Empty timeline is valid
        self.assertTrue(timeline._validate_structure())
        
        # Add keyframes and verify structure
        kf1 = KeyFrame(time=0.0, data=np.array([0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([10.0]))
        timeline.add_keyframe(kf1)
        timeline.add_keyframe(kf2)
        
        self.assertTrue(timeline._validate_structure())
    
    def test_evaluate_empty_timeline(self):
        """Test evaluating empty timeline returns None."""
        timeline = Timeline()
        result = timeline.evaluate(0.5)
        self.assertIsNone(result)
    
    def test_evaluate_single_keyframe(self):
        """Test evaluating timeline with single keyframe."""
        timeline = Timeline()
        kf = KeyFrame(time=0.0, data=np.array([5.0, 10.0]))
        timeline.add_keyframe(kf)
        
        result = timeline.evaluate(0.0)
        np.testing.assert_array_equal(result, np.array([5.0, 10.0]))
        
        # Before and after should return the same keyframe
        result = timeline.evaluate(-1.0)
        np.testing.assert_array_equal(result, np.array([5.0, 10.0]))
        
        result = timeline.evaluate(1.0)
        np.testing.assert_array_equal(result, np.array([5.0, 10.0]))
    
    def test_evaluate_interpolated(self):
        """Test evaluating at interpolated time."""
        timeline = Timeline()
        kf1 = KeyFrame(time=0.0, data=np.array([0.0, 0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([10.0, 20.0]))
        timeline.add_keyframe(kf1)
        timeline.add_keyframe(kf2)
        
        # At midpoint
        result = timeline.evaluate(0.5)
        expected = np.array([5.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)
        
        # At quarter point
        result = timeline.evaluate(0.25)
        expected = np.array([2.5, 5.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_at_keyframe_times(self):
        """Test evaluating exactly at keyframe times."""
        timeline = Timeline()
        kf1 = KeyFrame(time=0.0, data=np.array([0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([10.0]))
        kf3 = KeyFrame(time=2.0, data=np.array([5.0]))
        
        timeline.add_keyframe(kf1)
        timeline.add_keyframe(kf2)
        timeline.add_keyframe(kf3)
        
        np.testing.assert_array_equal(timeline.evaluate(0.0), np.array([0.0]))
        np.testing.assert_array_equal(timeline.evaluate(1.0), np.array([10.0]))
        np.testing.assert_array_equal(timeline.evaluate(2.0), np.array([5.0]))
    
    def test_evaluate_clamping(self):
        """Test that evaluate clamps to first/last keyframe outside range."""
        timeline = Timeline()
        kf1 = KeyFrame(time=0.0, data=np.array([0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([10.0]))
        timeline.add_keyframe(kf1)
        timeline.add_keyframe(kf2)
        
        # Before first keyframe
        result = timeline.evaluate(-5.0)
        np.testing.assert_array_equal(result, np.array([0.0]))
        
        # After last keyframe
        result = timeline.evaluate(5.0)
        np.testing.assert_array_equal(result, np.array([10.0]))
    
    def test_sample_timeline(self):
        """Test sampling timeline at regular intervals."""
        timeline = Timeline()
        kf1 = KeyFrame(time=0.0, data=np.array([0.0, 0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([10.0, 10.0]))
        timeline.add_keyframe(kf1)
        timeline.add_keyframe(kf2)
        
        samples = timeline.sample(5)
        self.assertEqual(samples.shape, (5, 2))
        
        # First sample at start
        np.testing.assert_array_almost_equal(samples[0], np.array([0.0, 0.0]))
        # Last sample at end
        np.testing.assert_array_almost_equal(samples[-1], np.array([10.0, 10.0]))
    
    def test_sample_empty_timeline(self):
        """Test sampling empty timeline returns empty array."""
        timeline = Timeline()
        samples = timeline.sample(5)
        self.assertEqual(samples.size, 0)
    
    def test_three_keyframes_with_interpolation(self):
        """Test complex timeline with three keyframes and custom interpolations."""
        timeline = Timeline()
        
        kf1 = KeyFrame(time=0.0, data=np.array([0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([10.0]))
        kf3 = KeyFrame(time=2.0, data=np.array([0.0]))
        
        interp1 = Interpolation(interp_type=InterpolationType.LINEAR)
        interp2 = Interpolation(interp_type=InterpolationType.BEZIER)
        
        timeline.add_keyframe(kf1)
        timeline.add_interpolation(interp1)
        timeline.add_keyframe(kf2)
        timeline.add_interpolation(interp2)
        timeline.add_keyframe(kf3)
        
        self.assertEqual(len(timeline), 5)
        self.assertEqual(len(timeline.get_keyframes()), 3)
        self.assertEqual(len(timeline.get_interpolations()), 2)
        
        # Test evaluation
        np.testing.assert_array_equal(timeline.evaluate(0.0), np.array([0.0]))
        np.testing.assert_array_equal(timeline.evaluate(1.0), np.array([10.0]))
        np.testing.assert_array_equal(timeline.evaluate(2.0), np.array([0.0]))
    
    def test_repr(self):
        """Test timeline string representation."""
        timeline = Timeline()
        kf1 = KeyFrame(time=0.0, data=np.array([0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([10.0]))
        
        timeline.add_keyframe(kf1)
        timeline.add_keyframe(kf2)
        
        repr_str = repr(timeline)
        self.assertIn("Timeline", repr_str)
        self.assertIn("keyframes=2", repr_str)
        self.assertIn("interpolations=1", repr_str)
    
    def test_len_method(self):
        """Test __len__ method."""
        timeline = Timeline()
        self.assertEqual(len(timeline), 0)
        
        timeline.add_keyframe(KeyFrame(time=0.0, data=np.array([0.0])))
        self.assertEqual(len(timeline), 1)
        
        timeline.add_keyframe(KeyFrame(time=1.0, data=np.array([10.0])))
        self.assertEqual(len(timeline), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full timeline system."""
    
    def test_full_animation_sequence(self):
        """Test a complete animation sequence with events."""
        events_triggered = []
        
        def on_start():
            events_triggered.append("start")
        
        def on_middle():
            events_triggered.append("middle")
        
        def on_end():
            events_triggered.append("end")
        
        timeline = Timeline()
        
        # Create keyframes with events
        kf1 = KeyFrame(time=0.0, data=np.array([0.0, 0.0, 0.0]), event=on_start)
        kf2 = KeyFrame(time=1.0, data=np.array([5.0, 5.0, 5.0]), event=on_middle)
        kf3 = KeyFrame(time=2.0, data=np.array([10.0, 10.0, 10.0]), event=on_end)
        
        # Use Bezier interpolation for smooth curves
        interp = Interpolation(
            interp_type=InterpolationType.BEZIER,
            control_points=np.array([[2.0, 2.0, 2.0], [8.0, 8.0, 8.0]])
        )
        
        timeline.add_keyframe(kf1)
        timeline.add_interpolation(interp)
        timeline.add_keyframe(kf2)
        timeline.add_keyframe(kf3)  # Auto linear interpolation
        
        # Sample the animation
        samples = timeline.sample(10)
        self.assertEqual(samples.shape, (10, 3))
        
        # Verify bounds
        np.testing.assert_array_almost_equal(samples[0], np.array([0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(samples[-1], np.array([10.0, 10.0, 10.0]))
        
        # Trigger events
        kf1.execute_event()
        kf2.execute_event()
        kf3.execute_event()
        
        self.assertEqual(events_triggered, ["start", "middle", "end"])
    
    def test_multi_dimensional_data(self):
        """Test with high-dimensional data (e.g., position + rotation)."""
        timeline = Timeline()
        
        # 6D data: [x, y, z, rx, ry, rz]
        kf1 = KeyFrame(time=0.0, data=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([10.0, 5.0, 3.0, 90.0, 45.0, 0.0]))
        
        timeline.add_keyframe(kf1)
        timeline.add_keyframe(kf2)
        
        result = timeline.evaluate(0.5)
        expected = np.array([5.0, 2.5, 1.5, 45.0, 22.5, 0.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_rapid_sampling(self):
        """Test with many samples for smooth animation."""
        timeline = Timeline()
        kf1 = KeyFrame(time=0.0, data=np.array([0.0]))
        kf2 = KeyFrame(time=1.0, data=np.array([100.0]))
        
        timeline.add_keyframe(kf1)
        timeline.add_keyframe(kf2)
        
        # Sample at 60fps for 1 second
        samples = timeline.sample(61)
        self.assertEqual(samples.shape, (61, 1))
        
        # Verify monotonic increase
        for i in range(1, len(samples)):
            self.assertGreaterEqual(samples[i][0], samples[i-1][0])


def run_tests():
    """Run all tests and return results."""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKeyFrame))
    suite.addTests(loader.loadTestsFromTestCase(TestInterpolation))
    suite.addTests(loader.loadTestsFromTestCase(TestTimeline))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
