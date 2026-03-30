"""
Unit tests for camera controller

Run with: python -m pytest test_camera.py -v
Or: python test_camera.py
"""

import unittest
import math
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import core components from camera_math (no robocute dependencies)
from camera_math import Vector3, lerp, lerp_vector, smooth_damp, FollowMode


class TestVector3(unittest.TestCase):
    """Test Vector3 operations"""
    
    def test_initialization(self):
        v = Vector3(1, 2, 3)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)
    
    def test_default_initialization(self):
        v = Vector3()
        self.assertEqual(v.x, 0.0)
        self.assertEqual(v.y, 0.0)
        self.assertEqual(v.z, 0.0)
    
    def test_addition(self):
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1 + v2
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 7)
        self.assertEqual(result.z, 9)
    
    def test_subtraction(self):
        v1 = Vector3(5, 7, 9)
        v2 = Vector3(1, 2, 3)
        result = v1 - v2
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 5)
        self.assertEqual(result.z, 6)
    
    def test_scalar_multiplication(self):
        v = Vector3(1, 2, 3)
        result = v * 2
        self.assertEqual(result.x, 2)
        self.assertEqual(result.y, 4)
        self.assertEqual(result.z, 6)
    
    def test_scalar_division(self):
        v = Vector3(4, 6, 8)
        result = v / 2
        self.assertEqual(result.x, 2)
        self.assertEqual(result.y, 3)
        self.assertEqual(result.z, 4)
    
    def test_negation(self):
        v = Vector3(1, -2, 3)
        result = -v
        self.assertEqual(result.x, -1)
        self.assertEqual(result.y, 2)
        self.assertEqual(result.z, -3)
    
    def test_length(self):
        v = Vector3(3, 4, 0)
        self.assertAlmostEqual(v.length(), 5.0)
        
        v = Vector3(1, 1, 1)
        self.assertAlmostEqual(v.length(), math.sqrt(3))
    
    def test_length_squared(self):
        v = Vector3(3, 4, 0)
        self.assertEqual(v.length_squared(), 25)
    
    def test_normalized(self):
        v = Vector3(3, 0, 0)
        unit = v.normalized()
        self.assertAlmostEqual(unit.x, 1)
        self.assertAlmostEqual(unit.y, 0)
        self.assertAlmostEqual(unit.z, 0)
        
        # Zero vector should return zero
        v = Vector3(0, 0, 0)
        unit = v.normalized()
        self.assertEqual(unit.x, 0)
        self.assertEqual(unit.y, 0)
        self.assertEqual(unit.z, 0)
    
    def test_copy(self):
        v1 = Vector3(1, 2, 3)
        v2 = v1.copy()
        self.assertEqual(v1.x, v2.x)
        self.assertEqual(v1.y, v2.y)
        self.assertEqual(v1.z, v2.z)
        
        # Modifying copy shouldn't affect original
        v2.x = 10
        self.assertEqual(v1.x, 1)
    
    def test_repr(self):
        v = Vector3(1.5, 2.5, 3.5)
        repr_str = repr(v)
        self.assertIn("1.500", repr_str)
        self.assertIn("2.500", repr_str)
        self.assertIn("3.500", repr_str)


class TestInterpolation(unittest.TestCase):
    """Test interpolation functions"""
    
    def test_lerp_0(self):
        """t=0 should return a"""
        self.assertAlmostEqual(lerp(0, 10, 0), 0)
    
    def test_lerp_1(self):
        """t=1 should return b"""
        self.assertAlmostEqual(lerp(0, 10, 1), 10)
    
    def test_lerp_half(self):
        """t=0.5 should return midpoint"""
        self.assertAlmostEqual(lerp(0, 10, 0.5), 5)
    
    def test_lerp_vector(self):
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 20, 30)
        
        result = lerp_vector(v1, v2, 0.5)
        self.assertAlmostEqual(result.x, 5)
        self.assertAlmostEqual(result.y, 10)
        self.assertAlmostEqual(result.z, 15)


class TestSmoothDamp(unittest.TestCase):
    """Test smooth damp function"""
    
    def test_converges_to_target(self):
        """Smooth damp should eventually reach target"""
        current = Vector3(0, 0, 0)
        target = Vector3(10, 0, 0)
        velocity = Vector3(0, 0, 0)
        smooth_time = 0.5
        
        # Simulate over time
        for _ in range(100):  # ~1.6 seconds at 60fps
            current, velocity = smooth_damp(current, target, velocity, smooth_time, 0.016)
        
        # Should be very close to target (within 1 unit after ~1.6 seconds)
        self.assertAlmostEqual(current.x, 10, places=0)
        # Velocity should be small (exact zero takes longer with spring system)
        self.assertLess(abs(velocity.x), 1.0, "Velocity should be small")
    
    def test_zero_delta_time(self):
        """Zero delta time shouldn't change position"""
        current = Vector3(5, 5, 5)
        target = Vector3(10, 10, 10)
        velocity = Vector3(1, 1, 1)
        
        new_pos, new_vel = smooth_damp(current, target, velocity, 0.3, 0)
        
        # Position should remain unchanged with zero dt
        self.assertAlmostEqual(new_pos.x, current.x, places=5)
        self.assertAlmostEqual(new_pos.y, current.y, places=5)
        self.assertAlmostEqual(new_pos.z, current.z, places=5)
    
    def test_very_small_smooth_time(self):
        """Very small smooth time should be clamped"""
        current = Vector3(0, 0, 0)
        target = Vector3(10, 0, 0)
        velocity = Vector3(0, 0, 0)
        
        # Should not crash with very small smooth_time
        new_pos, new_vel = smooth_damp(current, target, velocity, 0.00001, 0.016)
        # Should move closer to target
        self.assertGreater(new_pos.x, current.x)


class TestFollowMode(unittest.TestCase):
    """Test FollowMode enum"""
    
    def test_enum_values(self):
        """Test that all modes exist"""
        self.assertIsNotNone(FollowMode.SMOOTH)
        self.assertIsNotNone(FollowMode.SPRING)
        self.assertIsNotNone(FollowMode.PREDICTIVE)
        self.assertIsNotNone(FollowMode.ORBITAL)
    
    def test_enum_uniqueness(self):
        """All modes should be unique"""
        modes = [FollowMode.SMOOTH, FollowMode.SPRING, 
                FollowMode.PREDICTIVE, FollowMode.ORBITAL]
        self.assertEqual(len(modes), len(set(modes)))


class TestVector3Operations(unittest.TestCase):
    """Additional Vector3 operation tests"""
    
    def test_chained_operations(self):
        """Test that operations can be chained"""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        
        # (v1 + v2) * 2 - v1
        result = (v1 + v2) * 2 - v1
        # (5,7,9)*2 - (1,2,3) = (10,14,18) - (1,2,3) = (9,12,15)
        self.assertAlmostEqual(result.x, 9)
        self.assertAlmostEqual(result.y, 12)
        self.assertAlmostEqual(result.z, 15)
    
    def test_normalization_preserves_direction(self):
        """Normalization should preserve direction"""
        v = Vector3(3, 4, 0)
        unit = v.normalized()
        
        # Check length is 1
        self.assertAlmostEqual(unit.length(), 1.0, places=5)
        
        # Check direction is preserved (same ratio of components)
        if unit.x != 0:
            self.assertAlmostEqual(unit.y / unit.x, v.y / v.x, places=5)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_smooth_follow_simulation(self):
        """Simulate a complete smooth follow scenario"""
        # Target moving in a circle
        target_pos = Vector3(10, 0, 0)
        
        # Camera starting at origin
        camera_pos = Vector3(0, 5, -10)
        camera_vel = Vector3(0, 0, 0)
        
        smooth_time = 0.3
        
        # Simulate 2 seconds at 60fps
        for frame in range(120):
            # Target moves in circle
            angle = frame * 0.05
            target_pos = Vector3(
                math.cos(angle) * 10,
                0,
                math.sin(angle) * 10
            )
            
            # Desired position with offset
            offset = Vector3(0, 5, -10)
            desired = target_pos + offset
            
            # Apply smooth damp
            camera_pos, camera_vel = smooth_damp(
                camera_pos, desired, camera_vel, smooth_time, 0.016
            )
        
        # Camera should be reasonably close to desired position
        final_desired = target_pos + offset
        distance = (camera_pos - final_desired).length()
        self.assertLess(distance, 10)  # Within 10 units (following a moving target is hard!)
    
    def test_different_offset_directions(self):
        """Test camera follow with different offset directions"""
        target_pos = Vector3(0, 0, 0)
        
        offsets = [
            Vector3(0, 5, -10),   # Behind and above
            Vector3(0, 0, -10),   # Just behind
            Vector3(10, 0, 0),    # To the side
            Vector3(0, 10, 0),    # Above
            Vector3(-5, 3, -5),   # Diagonal
        ]
        
        for offset in offsets:
            camera_pos = Vector3(0, 0, 0)
            camera_vel = Vector3(0, 0, 0)
            
            # Simulate 1 second
            for _ in range(60):
                desired = target_pos + offset
                camera_pos, camera_vel = smooth_damp(
                    camera_pos, desired, camera_vel, 0.3, 0.016
                )
            
            # Camera should be close to desired offset position
            final_offset = camera_pos - target_pos
            offset_error = (final_offset - offset).length()
            self.assertLess(offset_error, 2, f"Failed for offset {offset}")


def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
