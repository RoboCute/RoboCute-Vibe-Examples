"""
Comprehensive Unit Tests for 3D Camera Follow System

This test suite covers:
- camera_math.py: Core math utilities (Vector3, interpolation, smooth damping)
- camera_controller.py: Camera controller with mocked RoboCute dependencies
- Edge cases and integration scenarios

Run with: python test_camera_comprehensive.py
Or: python -m pytest test_camera_comprehensive.py -v
"""

import unittest
import math
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import core components from camera_math (no robocute dependencies)
from camera_math import Vector3, lerp, lerp_vector, smooth_damp, FollowMode

# Mock robocute modules BEFORE importing camera_controller
# This must be done at module level to prevent numpy reimport issues
_rbc_mock = Mock()
_re_mock = Mock()
_lc_mock = Mock()

def _mock_double3(x, y, z):
    m = Mock()
    m.x = x
    m.y = y
    m.z = z
    return m

_lc_mock.double3 = _mock_double3

def _mock_float4(x, y, z, w):
    m = Mock()
    m.x = x
    m.y = y
    m.z = z
    m.w = w
    return m

_lc_mock.float4 = _mock_float4

# Inject mocks into sys.modules before importing camera_controller
sys.modules['robocute'] = _rbc_mock
sys.modules['robocute.rbc_ext'] = _re_mock
sys.modules['robocute.rbc_ext.luisa'] = _lc_mock

# Now import camera_controller - this will use the mocked modules
from camera_controller import CameraController, CameraManager


class TestVector3Basics(unittest.TestCase):
    """Basic Vector3 operations"""
    
    def test_initialization_with_values(self):
        v = Vector3(1, 2, 3)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 2)
        self.assertEqual(v.z, 3)
    
    def test_default_initialization(self):
        v = Vector3()
        self.assertEqual(v.x, 0.0)
        self.assertEqual(v.y, 0.0)
        self.assertEqual(v.z, 0.0)
    
    def test_mixed_initialization(self):
        """Test partial initialization"""
        v = Vector3(5.0)
        self.assertEqual(v.x, 5.0)
        self.assertEqual(v.y, 0.0)
        self.assertEqual(v.z, 0.0)


class TestVector3Arithmetic(unittest.TestCase):
    """Vector3 arithmetic operations"""
    
    def test_addition_vectors(self):
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = v1 + v2
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 7)
        self.assertEqual(result.z, 9)
    
    def test_addition_scalar(self):
        v = Vector3(1, 2, 3)
        result = v + 10
        self.assertEqual(result.x, 11)
        self.assertEqual(result.y, 12)
        self.assertEqual(result.z, 13)
    
    def test_subtraction_vectors(self):
        v1 = Vector3(5, 7, 9)
        v2 = Vector3(1, 2, 3)
        result = v1 - v2
        self.assertEqual(result.x, 4)
        self.assertEqual(result.y, 5)
        self.assertEqual(result.z, 6)
    
    def test_subtraction_scalar(self):
        v = Vector3(10, 20, 30)
        result = v - 5
        self.assertEqual(result.x, 5)
        self.assertEqual(result.y, 15)
        self.assertEqual(result.z, 25)
    
    def test_scalar_multiplication(self):
        v = Vector3(1, 2, 3)
        result = v * 2
        self.assertEqual(result.x, 2)
        self.assertEqual(result.y, 4)
        self.assertEqual(result.z, 6)
    
    def test_scalar_multiplication_zero(self):
        v = Vector3(1, 2, 3)
        result = v * 0
        self.assertEqual(result.x, 0)
        self.assertEqual(result.y, 0)
        self.assertEqual(result.z, 0)
    
    def test_scalar_multiplication_negative(self):
        v = Vector3(1, 2, 3)
        result = v * -1
        self.assertEqual(result.x, -1)
        self.assertEqual(result.y, -2)
        self.assertEqual(result.z, -3)
    
    def test_scalar_division(self):
        v = Vector3(4, 6, 8)
        result = v / 2
        self.assertEqual(result.x, 2)
        self.assertEqual(result.y, 3)
        self.assertEqual(result.z, 4)
    
    def test_scalar_division_float(self):
        v = Vector3(1, 1, 1)
        result = v / 3
        self.assertAlmostEqual(result.x, 1/3)
        self.assertAlmostEqual(result.y, 1/3)
        self.assertAlmostEqual(result.z, 1/3)
    
    def test_negation(self):
        v = Vector3(1, -2, 3)
        result = -v
        self.assertEqual(result.x, -1)
        self.assertEqual(result.y, 2)
        self.assertEqual(result.z, -3)
    
    def test_double_negation(self):
        v = Vector3(1, 2, 3)
        result = -(-v)
        self.assertEqual(result.x, v.x)
        self.assertEqual(result.y, v.y)
        self.assertEqual(result.z, v.z)


class TestVector3Methods(unittest.TestCase):
    """Vector3 methods"""
    
    def test_length_zero(self):
        v = Vector3(0, 0, 0)
        self.assertEqual(v.length(), 0)
    
    def test_length_unit(self):
        v = Vector3(1, 0, 0)
        self.assertEqual(v.length(), 1)
        v = Vector3(0, 1, 0)
        self.assertEqual(v.length(), 1)
        v = Vector3(0, 0, 1)
        self.assertEqual(v.length(), 1)
    
    def test_length_3d(self):
        v = Vector3(3, 4, 0)
        self.assertEqual(v.length(), 5.0)
        v = Vector3(1, 2, 2)
        self.assertEqual(v.length(), 3.0)
    
    def test_length_squared(self):
        v = Vector3(3, 4, 0)
        self.assertEqual(v.length_squared(), 25)
        v = Vector3(1, 2, 2)
        self.assertEqual(v.length_squared(), 9)
    
    def test_normalized_unit_vector(self):
        v = Vector3(5, 0, 0)
        unit = v.normalized()
        self.assertAlmostEqual(unit.x, 1)
        self.assertAlmostEqual(unit.y, 0)
        self.assertAlmostEqual(unit.z, 0)
    
    def test_normalized_zero_vector(self):
        """Zero vector should return zero, not crash"""
        v = Vector3(0, 0, 0)
        unit = v.normalized()
        self.assertEqual(unit.x, 0)
        self.assertEqual(unit.y, 0)
        self.assertEqual(unit.z, 0)
    
    def test_normalized_diagonal(self):
        v = Vector3(1, 1, 1)
        unit = v.normalized()
        expected = 1 / math.sqrt(3)
        self.assertAlmostEqual(unit.x, expected, places=5)
        self.assertAlmostEqual(unit.y, expected, places=5)
        self.assertAlmostEqual(unit.z, expected, places=5)
        self.assertAlmostEqual(unit.length(), 1.0, places=5)
    
    def test_copy_independence(self):
        v1 = Vector3(1, 2, 3)
        v2 = v1.copy()
        v2.x = 100
        v2.y = 200
        v2.z = 300
        self.assertEqual(v1.x, 1)
        self.assertEqual(v1.y, 2)
        self.assertEqual(v1.z, 3)
    
    def test_repr_format(self):
        v = Vector3(1.5, 2.5, 3.5)
        repr_str = repr(v)
        self.assertIn("Vector3", repr_str)
        self.assertIn("1.500", repr_str)
        self.assertIn("2.500", repr_str)
        self.assertIn("3.500", repr_str)


class TestLerp(unittest.TestCase):
    """Linear interpolation tests"""
    
    def test_lerp_0(self):
        """t=0 should return a"""
        self.assertAlmostEqual(lerp(0, 10, 0), 0)
        self.assertAlmostEqual(lerp(5, 15, 0), 5)
    
    def test_lerp_1(self):
        """t=1 should return b"""
        self.assertAlmostEqual(lerp(0, 10, 1), 10)
        self.assertAlmostEqual(lerp(5, 15, 1), 15)
    
    def test_lerp_half(self):
        """t=0.5 should return midpoint"""
        self.assertAlmostEqual(lerp(0, 10, 0.5), 5)
        self.assertAlmostEqual(lerp(10, 20, 0.5), 15)
    
    def test_lerp_quarter(self):
        """t=0.25 should return quarter point"""
        self.assertAlmostEqual(lerp(0, 100, 0.25), 25)
    
    def test_lerp_out_of_range_high(self):
        """t>1 should extrapolate"""
        self.assertAlmostEqual(lerp(0, 10, 2), 20)
    
    def test_lerp_out_of_range_low(self):
        """t<0 should extrapolate"""
        self.assertAlmostEqual(lerp(0, 10, -1), -10)
    
    def test_lerp_negative_range(self):
        """Lerp with negative values"""
        self.assertAlmostEqual(lerp(-10, 10, 0.5), 0)
        self.assertAlmostEqual(lerp(-10, -5, 0.5), -7.5)


class TestLerpVector(unittest.TestCase):
    """Vector linear interpolation tests"""
    
    def test_lerp_vector_0(self):
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 20, 30)
        result = lerp_vector(v1, v2, 0)
        self.assertAlmostEqual(result.x, 0)
        self.assertAlmostEqual(result.y, 0)
        self.assertAlmostEqual(result.z, 0)
    
    def test_lerp_vector_1(self):
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 20, 30)
        result = lerp_vector(v1, v2, 1)
        self.assertAlmostEqual(result.x, 10)
        self.assertAlmostEqual(result.y, 20)
        self.assertAlmostEqual(result.z, 30)
    
    def test_lerp_vector_half(self):
        v1 = Vector3(0, 0, 0)
        v2 = Vector3(10, 20, 30)
        result = lerp_vector(v1, v2, 0.5)
        self.assertAlmostEqual(result.x, 5)
        self.assertAlmostEqual(result.y, 10)
        self.assertAlmostEqual(result.z, 15)
    
    def test_lerp_vector_nonzero_start(self):
        v1 = Vector3(5, 10, 15)
        v2 = Vector3(15, 30, 45)
        result = lerp_vector(v1, v2, 0.5)
        self.assertAlmostEqual(result.x, 10)
        self.assertAlmostEqual(result.y, 20)
        self.assertAlmostEqual(result.z, 30)


class TestSmoothDamp(unittest.TestCase):
    """Smooth damp (critically damped spring) tests"""
    
    def test_converges_to_target(self):
        """Smooth damp should eventually reach target"""
        current = Vector3(0, 0, 0)
        target = Vector3(10, 0, 0)
        velocity = Vector3(0, 0, 0)
        smooth_time = 0.5
        
        # Simulate over time
        for _ in range(100):  # ~1.6 seconds at 60fps
            current, velocity = smooth_damp(current, target, velocity, smooth_time, 0.016)
        
        # Should be very close to target
        self.assertAlmostEqual(current.x, 10, places=0)
        self.assertLess(abs(velocity.x), 1.0)
    
    def test_zero_delta_time(self):
        """Zero delta time shouldn't change position"""
        current = Vector3(5, 5, 5)
        target = Vector3(10, 10, 10)
        velocity = Vector3(1, 1, 1)
        
        new_pos, new_vel = smooth_damp(current, target, velocity, 0.3, 0)
        
        self.assertAlmostEqual(new_pos.x, current.x, places=5)
        self.assertAlmostEqual(new_pos.y, current.y, places=5)
        self.assertAlmostEqual(new_pos.z, current.z, places=5)
    
    def test_very_small_smooth_time(self):
        """Very small smooth time should be clamped"""
        current = Vector3(0, 0, 0)
        target = Vector3(10, 0, 0)
        velocity = Vector3(0, 0, 0)
        
        new_pos, new_vel = smooth_damp(current, target, velocity, 0.00001, 0.016)
        self.assertGreater(new_pos.x, current.x)
    
    def test_very_small_delta_time(self):
        """Very small delta time should work"""
        current = Vector3(0, 0, 0)
        target = Vector3(10, 0, 0)
        velocity = Vector3(0, 0, 0)
        
        new_pos, new_vel = smooth_damp(current, target, velocity, 0.3, 0.000001)
        # Should return current position
        self.assertAlmostEqual(new_pos.x, current.x, places=5)
    
    def test_no_overshoot(self):
        """Critical damping should not overshoot"""
        current = Vector3(0, 0, 0)
        target = Vector3(10, 0, 0)
        velocity = Vector3(0, 0, 0)
        smooth_time = 0.5
        
        max_value = current.x
        for _ in range(60):
            current, velocity = smooth_damp(current, target, velocity, smooth_time, 0.016)
            max_value = max(max_value, current.x)
        
        # Should not overshoot target
        self.assertLessEqual(max_value, 10.0001)  # Small tolerance for float error
    
    def test_velocity_decreases_near_target(self):
        """Velocity should decrease as we approach target"""
        current = Vector3(0, 0, 0)
        target = Vector3(10, 0, 0)
        velocity = Vector3(5, 0, 0)  # Start with some velocity
        smooth_time = 0.5
        
        velocities = []
        for _ in range(20):
            current, velocity = smooth_damp(current, target, velocity, smooth_time, 0.016)
            velocities.append(abs(velocity.x))
        
        # Velocity should generally decrease after initial acceleration
        # Check last few velocities are smaller than middle ones
        self.assertLess(velocities[-1], velocities[10])


class TestFollowMode(unittest.TestCase):
    """FollowMode enum tests"""
    
    def test_all_modes_exist(self):
        self.assertIsNotNone(FollowMode.SMOOTH)
        self.assertIsNotNone(FollowMode.SPRING)
        self.assertIsNotNone(FollowMode.PREDICTIVE)
        self.assertIsNotNone(FollowMode.ORBITAL)
    
    def test_modes_are_unique(self):
        modes = [FollowMode.SMOOTH, FollowMode.SPRING, 
                FollowMode.PREDICTIVE, FollowMode.ORBITAL]
        self.assertEqual(len(modes), len(set(modes)))
    
    def test_mode_names(self):
        self.assertEqual(FollowMode.SMOOTH.name, "SMOOTH")
        self.assertEqual(FollowMode.SPRING.name, "SPRING")
        self.assertEqual(FollowMode.PREDICTIVE.name, "PREDICTIVE")
        self.assertEqual(FollowMode.ORBITAL.name, "ORBITAL")


class TestChainedOperations(unittest.TestCase):
    """Complex chained operations"""
    
    def test_complex_chain_1(self):
        """Test (v1 + v2) * 2 - v1"""
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        result = (v1 + v2) * 2 - v1
        # (5,7,9)*2 - (1,2,3) = (10,14,18) - (1,2,3) = (9,12,15)
        self.assertAlmostEqual(result.x, 9)
        self.assertAlmostEqual(result.y, 12)
        self.assertAlmostEqual(result.z, 15)
    
    def test_complex_chain_2(self):
        """Test v1 * 3 + v2 * 2 - (v1 + v2)"""
        v1 = Vector3(1, 1, 1)
        v2 = Vector3(2, 2, 2)
        result = v1 * 3 + v2 * 2 - (v1 + v2)
        # (3,3,3) + (4,4,4) - (3,3,3) = (4,4,4)
        self.assertAlmostEqual(result.x, 4)
        self.assertAlmostEqual(result.y, 4)
        self.assertAlmostEqual(result.z, 4)
    
    def test_normalization_chain(self):
        """Test that normalization preserves direction in complex chains"""
        v = Vector3(3, 4, 0)
        normalized = v.normalized()
        scaled = normalized * 10
        self.assertAlmostEqual(scaled.length(), 10, places=5)
        self.assertAlmostEqual(scaled.x / scaled.y, v.x / v.y, places=5)


class TestCameraControllerMocked(unittest.TestCase):
    """Camera controller tests with mocked dependencies"""
    
    def setUp(self):
        """Set up mocks for RoboCute dependencies"""
        # Create mock app
        self.mock_app = Mock()
        self.mock_transform = Mock()
        self.mock_transform.position.return_value = Mock(x=0, y=0, z=0)
        self.mock_app.get_display_transform.return_value = self.mock_transform
        
        # Use module-level imported classes (already imported with mocked dependencies)
        self.CameraController = CameraController
        self.CameraManager = CameraManager
        self.FollowMode = FollowMode
        self.Vector3 = Vector3
    
    def test_camera_controller_initialization(self):
        """Test camera controller initializes correctly"""
        controller = self.CameraController(self.mock_app)
        self.assertEqual(controller.mode, self.FollowMode.SMOOTH)
        self.assertIsNone(controller.target_entity)
        self.assertEqual(controller.offset.x, 0)
        self.assertEqual(controller.offset.y, 5)
        self.assertEqual(controller.offset.z, -10)
    
    def test_camera_controller_set_target(self):
        """Test setting target entity - patches the re module in camera_controller"""
        # Need to patch the 're' reference inside camera_controller module
        with patch('camera_controller.re') as mock_re:
            # Setup the mock re module with proper position values
            class MockPos:
                def __init__(self):
                    self.x = 5.0
                    self.y = 1.0
                    self.z = 5.0
            
            mock_pos = MockPos()
            mock_trans_component = Mock()
            mock_trans_component.position.return_value = mock_pos
            
            def mock_transform_constructor(comp):
                return mock_trans_component
            
            mock_re.world.TransformComponent.side_effect = mock_transform_constructor
            
            # Create target entity mock
            mock_target = Mock()
            mock_target.get_component.return_value = mock_trans_component
            
            # Create controller with patched re
            controller = self.CameraController(self.mock_app)
            controller.set_target(mock_target)
            
            self.assertEqual(controller.target_entity, mock_target)
            self.assertEqual(controller.target_position.x, 5.0)
    
    def test_camera_controller_set_mode(self):
        """Test setting follow mode"""
        controller = self.CameraController(self.mock_app)
        
        controller.set_mode(self.FollowMode.SPRING)
        self.assertEqual(controller.mode, self.FollowMode.SPRING)
        
        controller.set_mode(self.FollowMode.PREDICTIVE)
        self.assertEqual(controller.mode, self.FollowMode.PREDICTIVE)
        
        controller.set_mode(self.FollowMode.ORBITAL)
        self.assertEqual(controller.mode, self.FollowMode.ORBITAL)
    
    def test_calculate_smooth_position(self):
        """Test smooth position calculation"""
        controller = self.CameraController(self.mock_app)
        controller.target_position = Vector3(10, 0, 10)
        
        pos = controller._calculate_smooth_position()
        
        # Should be target + offset
        self.assertEqual(pos.x, 10)  # 10 + 0
        self.assertEqual(pos.y, 5)   # 0 + 5
        self.assertEqual(pos.z, 0)   # 10 + (-10)
    
    def test_calculate_predictive_position(self):
        """Test predictive position calculation"""
        controller = self.CameraController(self.mock_app)
        controller.target_position = Vector3(10, 0, 10)
        controller.target_velocity = Vector3(5, 0, 0)
        controller.look_ahead_factor = 0.3
        
        pos = controller._calculate_predictive_position(0.016)
        
        # Should be target + offset + velocity * look_ahead
        self.assertEqual(pos.x, 11.5)  # 10 + 0 + 5*0.3
        self.assertEqual(pos.y, 5)     # 0 + 5
        self.assertEqual(pos.z, 0)     # 10 + (-10)
    
    def test_calculate_orbital_position(self):
        """Test orbital position calculation"""
        controller = self.CameraController(self.mock_app)
        controller.target_position = Vector3(0, 0, 0)
        controller.orbit_distance = 10
        controller.orbit_height = 5
        controller.orbit_angle = 0
        controller.orbit_speed = 0  # No rotation for this test
        
        pos = controller._calculate_orbital_position(0.016)
        
        # At angle 0: x = cos(0)*10 = 10, z = sin(0)*10 = 0
        self.assertAlmostEqual(pos.x, 10, places=5)
        self.assertEqual(pos.y, 5)
        self.assertAlmostEqual(pos.z, 0, places=5)
    
    def test_orbital_angle_increments(self):
        """Test that orbital angle increments each update"""
        controller = self.CameraController(self.mock_app)
        controller.target_position = Vector3(0, 0, 0)
        controller.orbit_distance = 10
        controller.orbit_height = 5
        controller.orbit_angle = 0
        controller.orbit_speed = 1.0
        
        initial_angle = controller.orbit_angle
        controller._calculate_orbital_position(0.016)
        
        self.assertGreater(controller.orbit_angle, initial_angle)
    
    def test_clamp_distance_min(self):
        """Test distance clamping at minimum"""
        controller = self.CameraController(self.mock_app)
        controller.target_position = Vector3(0, 0, 0)
        controller.min_distance = 5.0
        controller.current_position = Vector3(2, 0, 0)  # Only 2 units away
        
        controller._clamp_distance()
        
        # Should be pushed to min_distance
        self.assertAlmostEqual((controller.current_position - controller.target_position).length(), 5.0, places=5)
    
    def test_clamp_distance_max(self):
        """Test distance clamping at maximum"""
        controller = self.CameraController(self.mock_app)
        controller.target_position = Vector3(0, 0, 0)
        controller.max_distance = 20.0
        controller.current_position = Vector3(30, 0, 0)  # 30 units away
        
        controller._clamp_distance()
        
        # Should be pulled to max_distance
        self.assertAlmostEqual((controller.current_position - controller.target_position).length(), 20.0, places=5)
    
    def test_collision_avoidance(self):
        """Test collision avoidance pushes camera back"""
        controller = self.CameraController(self.mock_app)
        controller.target_position = Vector3(0, 0, 0)
        controller.min_distance = 2.0
        controller.collision_buffer = 0.5
        controller.current_position = Vector3(2.2, 0, 0)  # Too close
        
        new_pos = controller._apply_collision_avoidance()
        
        # Should be pushed to min_distance + buffer
        expected_distance = 2.5
        self.assertAlmostEqual((new_pos - controller.target_position).length(), expected_distance, places=5)
    
    def test_collision_avoidance_no_change_when_far(self):
        """Test collision avoidance doesn't change position when far enough"""
        controller = self.CameraController(self.mock_app)
        controller.target_position = Vector3(0, 0, 0)
        controller.min_distance = 2.0
        controller.collision_buffer = 0.5
        controller.current_position = Vector3(10, 0, 0)  # Far enough
        
        new_pos = controller._apply_collision_avoidance()
        
        # Should not change
        self.assertEqual(new_pos.x, 10)
        self.assertEqual(new_pos.y, 0)
        self.assertEqual(new_pos.z, 0)
    
    def test_get_camera_info(self):
        """Test camera info retrieval"""
        controller = self.CameraController(self.mock_app)
        controller.target_position = Vector3(10, 0, 0)
        controller.current_position = Vector3(10, 5, -10)
        controller.mode = self.FollowMode.SMOOTH
        
        info = controller.get_camera_info()
        
        self.assertEqual(info["mode"], "SMOOTH")
        self.assertIn("position", info)
        self.assertIn("target_position", info)
        self.assertIn("distance_to_target", info)
        self.assertIn("target_velocity", info)


class TestCameraManager(unittest.TestCase):
    """Camera manager tests with mocked dependencies"""
    
    def setUp(self):
        """Set up mocks"""
        self.mock_app = Mock()
        self.mock_transform = Mock()
        self.mock_transform.position.return_value = Mock(x=0, y=0, z=0)
        self.mock_app.get_display_transform.return_value = self.mock_transform
        
        # Use module-level imported classes (already imported with mocked dependencies)
        self.CameraManager = CameraManager
        self.CameraController = CameraController
        self.FollowMode = FollowMode
    
    def test_manager_create_camera(self):
        """Test creating cameras"""
        manager = self.CameraManager(self.mock_app)
        camera = manager.create_camera("main")
        
        self.assertIsInstance(camera, self.CameraController)
        self.assertEqual(manager.active_camera, "main")
        self.assertIn("main", manager.cameras)
    
    def test_manager_get_camera(self):
        """Test getting camera by name"""
        manager = self.CameraManager(self.mock_app)
        manager.create_camera("main")
        
        camera = manager.get_camera("main")
        self.assertIsNotNone(camera)
        
        missing = manager.get_camera("nonexistent")
        self.assertIsNone(missing)
    
    def test_manager_set_active_camera(self):
        """Test setting active camera"""
        manager = self.CameraManager(self.mock_app)
        manager.create_camera("main")
        manager.create_camera("secondary")
        
        manager.set_active_camera("secondary")
        self.assertEqual(manager.active_camera, "secondary")
    
    def test_manager_cycle_mode(self):
        """Test cycling through follow modes"""
        manager = self.CameraManager(self.mock_app)
        manager.create_camera("main")
        
        # Start with SMOOTH
        self.assertEqual(manager.cameras["main"].mode, self.FollowMode.SMOOTH)
        
        # Cycle to next mode
        mode_name = manager.cycle_mode()
        self.assertEqual(mode_name, "SPRING")
        
        # Cycle again
        mode_name = manager.cycle_mode()
        self.assertEqual(mode_name, "PREDICTIVE")


class TestIntegration(unittest.TestCase):
    """Integration tests simulating real scenarios"""
    
    def test_smooth_follow_simulation(self):
        """Simulate complete smooth follow scenario"""
        target_pos = Vector3(10, 0, 0)
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
        self.assertLess(distance, 10)
    
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
    
    def test_high_speed_target_following(self):
        """Test following a very fast target"""
        target_pos = Vector3(0, 0, 0)
        camera_pos = Vector3(0, 5, -10)
        camera_vel = Vector3(0, 0, 0)
        smooth_time = 0.1  # Fast response
        
        # Simulate target moving very fast
        for frame in range(100):
            target_pos = Vector3(frame * 2, 0, 0)  # Moving 2 units per frame
            offset = Vector3(0, 5, -10)
            desired = target_pos + offset
            camera_pos, camera_vel = smooth_damp(
                camera_pos, desired, camera_vel, smooth_time, 0.016
            )
        
        # Camera should be somewhat close (high speed is hard to follow perfectly)
        final_desired = target_pos + offset
        distance = (camera_pos - final_desired).length()
        self.assertLess(distance, 50)  # More tolerance for high speed
    
    def test_oscillating_target(self):
        """Test following an oscillating target"""
        target_pos = Vector3(0, 0, 0)
        camera_pos = Vector3(0, 5, -10)
        camera_vel = Vector3(0, 0, 0)
        smooth_time = 0.2
        
        positions = []
        for frame in range(180):  # 3 seconds
            # Oscillate in sine wave
            target_pos = Vector3(
                math.sin(frame * 0.1) * 20,
                0,
                0
            )
            offset = Vector3(0, 5, -10)
            desired = target_pos + offset
            camera_pos, camera_vel = smooth_damp(
                camera_pos, desired, camera_vel, smooth_time, 0.016
            )
            positions.append(camera_pos.x)
        
        # Camera should show oscillating behavior
        # Check that camera moved significantly
        self.assertGreater(max(positions) - min(positions), 10)


class TestEdgeCases(unittest.TestCase):
    """Edge case tests"""
    
    def test_very_large_values(self):
        """Test with very large vector values"""
        v1 = Vector3(1e10, 1e10, 1e10)
        v2 = Vector3(1e10 + 1, 1e10 + 2, 1e10 + 3)
        result = v2 - v1
        self.assertEqual(result.x, 1)
        self.assertEqual(result.y, 2)
        self.assertEqual(result.z, 3)
    
    def test_very_small_values(self):
        """Test with very small vector values"""
        v1 = Vector3(1e-10, 1e-10, 1e-10)
        v2 = Vector3(2e-10, 3e-10, 4e-10)
        result = v2 - v1
        self.assertAlmostEqual(result.x, 1e-10)
        self.assertAlmostEqual(result.y, 2e-10)
        self.assertAlmostEqual(result.z, 3e-10)
    
    def test_infinity_handling(self):
        """Test with infinity values"""
        v = Vector3(float('inf'), 0, 0)
        self.assertEqual(v.length(), float('inf'))
    
    def test_nan_handling(self):
        """Test with NaN values"""
        v = Vector3(float('nan'), 0, 0)
        self.assertTrue(math.isnan(v.length()))


def run_tests():
    """Run all tests with verbosity"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
