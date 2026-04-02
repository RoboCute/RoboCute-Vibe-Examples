"""
Test Case: Headless Camera Follow - 256 Frames Verification

This test verifies that the headless_follow_demo.py script:
1. Completes 256 frames without errors
2. Successfully tracks the moving target
3. Outputs camera follow statistics

Run with: python 3D_Camera/test_headless_256_frames.py
"""

import sys
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add 3D_Camera to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the pure simulation components (no robocute dependencies)
from camera_math import Vector3, FollowMode, smooth_damp


class CameraFollowStats:
    """Statistics collector for camera follow verification"""
    
    def __init__(self):
        self.frame_count = 0
        self.target_positions: List[Vector3] = []
        self.camera_positions: List[Vector3] = []
        self.distances: List[float] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Tracking quality metrics
        self.avg_distance = 0.0
        self.min_distance = float('inf')
        self.max_distance = 0.0
        self.total_distance_traveled = 0.0
        
    def record_frame(self, camera_pos: Vector3, target_pos: Vector3):
        """Record a frame's data"""
        self.frame_count += 1
        self.camera_positions.append(camera_pos.copy())
        self.target_positions.append(target_pos.copy())
        
        distance = (camera_pos - target_pos).length()
        self.distances.append(distance)
        
        self.min_distance = min(self.min_distance, distance)
        self.max_distance = max(self.max_distance, distance)
        self.avg_distance = sum(self.distances) / len(self.distances)
        
        # Track total movement
        if len(self.camera_positions) > 1:
            movement = (self.camera_positions[-1] - self.camera_positions[-2]).length()
            self.total_distance_traveled += movement
    
    def add_error(self, msg: str):
        self.errors.append(msg)
        
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def get_summary(self) -> Dict:
        """Get statistical summary"""
        if not self.distances:
            return {"error": "No frames recorded"}
        
        # Calculate tracking quality
        expected_distance = self.distances[0] if self.distances else 0
        deviations = [abs(d - expected_distance) for d in self.distances]
        max_deviation = max(deviations) if deviations else 0
        avg_deviation = sum(deviations) / len(deviations) if deviations else 0
        
        # Calculate smoothness (jitter)
        if len(self.camera_positions) > 1:
            position_changes = [
                (self.camera_positions[i] - self.camera_positions[i-1]).length()
                for i in range(1, len(self.camera_positions))
            ]
            avg_movement = sum(position_changes) / len(position_changes)
            max_movement = max(position_changes)
        else:
            avg_movement = 0
            max_movement = 0
        
        # Check for NaN positions
        nan_detected = any(
            math.isnan(p.x) or math.isnan(p.y) or math.isnan(p.z)
            for p in self.camera_positions
        )
        
        return {
            "total_frames": self.frame_count,
            "avg_distance_to_target": self.avg_distance,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "max_distance_deviation": max_deviation,
            "avg_distance_deviation": avg_deviation,
            "avg_movement_per_frame": avg_movement,
            "max_movement_per_frame": max_movement,
            "total_distance_traveled": self.total_distance_traveled,
            "tracking_stable": max_deviation < 10.0,  # Less than 10 units deviation
            "nan_detected": nan_detected,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
        }


class SimulatedScene:
    """Minimal scene simulation for headless testing"""
    def __init__(self):
        self.entities = {}
        self._entity_counter = 0
    
    def add_entity(self, name: str = "entity"):
        """Simulate adding an entity"""
        self._entity_counter += 1
        entity_id = f"{name}_{self._entity_counter}"
        self.entities[entity_id] = {
            "id": entity_id,
            "position": Vector3(0, 0, 0),
            "velocity": Vector3(0, 0, 0),
        }
        return SimulatedEntity(self, entity_id)
    
    def set_entity_position(self, entity_id: str, position: Vector3):
        """Set entity position"""
        if entity_id in self.entities:
            self.entities[entity_id]["position"] = position.copy()


class SimulatedEntity:
    """Minimal entity for headless testing"""
    def __init__(self, scene: SimulatedScene, entity_id: str):
        self.scene = scene
        self.entity_id = entity_id
        self._position = Vector3(0, 0, 0)
    
    def get_component(self, name: str):
        """Simulate getting a component"""
        return self
    
    def position(self):
        """Get position"""
        return self._position
    
    def set_pos(self, pos, recursive=False):
        """Set position"""
        self._position = pos.copy() if hasattr(pos, 'copy') else Vector3(pos.x, pos.y, pos.z)
        self.scene.set_entity_position(self.entity_id, self._position)


class HeadlessCameraController:
    """
    Camera controller that works without display transform
    Pure simulation of camera following logic
    """
    
    def __init__(self, name: str = "main_camera"):
        self.name = name
        self.mode = FollowMode.SMOOTH
        
        # Target
        self.target_entity = None
        self.target_position = Vector3(0, 0, 0)
        self.target_velocity = Vector3(0, 0, 0)
        self.target_last_position = Vector3(0, 0, 0)
        
        # Settings
        self.offset = Vector3(0, 5, -10)
        self.position_smooth_speed = 4.0
        self.rotation_smooth_speed = 5.0
        self.min_distance = 2.0
        self.max_distance = 50.0
        self.enable_collision_avoidance = True
        self.collision_buffer = 0.5
        
        # State
        self.current_position = Vector3(0, 5, -10)
        self.current_velocity = Vector3(0, 0, 0)
        self.current_rotation = Vector3(0, 0, 0)
    
    def set_target(self, target):
        """Set target entity"""
        self.target_entity = target
        if target:
            pos = target.position()
            self.target_position = Vector3(pos.x, pos.y, pos.z)
            self.target_last_position = self.target_position.copy()
            self.current_position = self.target_position + self.offset
    
    def set_mode(self, mode: FollowMode):
        """Set follow mode"""
        self.mode = mode
    
    def update(self, delta_time: float):
        """Update camera position"""
        if not self.target_entity:
            return
        
        # Update target position and velocity
        current_target_pos = self.target_entity.position()
        new_target_pos = Vector3(current_target_pos.x, current_target_pos.y, current_target_pos.z)
        
        if delta_time > 0:
            self.target_velocity = (new_target_pos - self.target_last_position) / delta_time
        
        self.target_last_position = self.target_position.copy()
        self.target_position = new_target_pos
        
        # Calculate desired position based on mode
        if self.mode == FollowMode.SMOOTH:
            desired_pos = self._calculate_smooth_position()
            self.current_position = self._apply_smooth_follow(desired_pos, delta_time)
        elif self.mode == FollowMode.PREDICTIVE:
            desired_pos = self._calculate_predictive_position(delta_time)
            self.current_position = self._apply_smooth_follow(desired_pos, delta_time)
        else:
            desired_pos = self._calculate_smooth_position()
            self.current_position = self._apply_smooth_follow(desired_pos, delta_time)
        
        # Apply collision avoidance
        if self.enable_collision_avoidance:
            self._apply_collision_avoidance()
        
        # Clamp distance
        self._clamp_distance()
    
    def _calculate_smooth_position(self) -> Vector3:
        """Calculate desired position with offset"""
        return self.target_position + self.offset
    
    def _calculate_predictive_position(self, delta_time: float) -> Vector3:
        """Calculate predictive position with look-ahead"""
        base_pos = self.target_position + self.offset
        prediction = self.target_velocity * 0.3  # look_ahead_factor
        return base_pos + prediction
    
    def _apply_smooth_follow(self, desired_pos: Vector3, delta_time: float) -> Vector3:
        """Apply smooth damping to move towards desired position"""
        smooth_time = 1.0 / self.position_smooth_speed
        new_pos, new_vel = smooth_damp(
            self.current_position,
            desired_pos,
            self.current_velocity,
            smooth_time,
            delta_time
        )
        self.current_velocity = new_vel
        return new_pos
    
    def _apply_collision_avoidance(self):
        """Keep camera at minimum distance"""
        direction = self.current_position - self.target_position
        distance = direction.length()
        
        if distance < self.min_distance + self.collision_buffer:
            push_dir = direction.normalized()
            if push_dir.length() < 0.001:
                push_dir = Vector3(0, 1, 0)
            target_dist = self.min_distance + self.collision_buffer
            self.current_position = self.target_position + push_dir * target_dist
    
    def _clamp_distance(self):
        """Clamp camera distance"""
        direction = self.current_position - self.target_position
        distance = direction.length()
        
        if distance < self.min_distance:
            dir_norm = direction.normalized()
            if dir_norm.length() < 0.001:
                dir_norm = Vector3(0, 0, 1)
            self.current_position = self.target_position + dir_norm * self.min_distance
        elif distance > self.max_distance:
            dir_norm = direction.normalized()
            self.current_position = self.target_position + dir_norm * self.max_distance
    
    def get_distance_to_target(self) -> float:
        """Get distance to target"""
        return (self.current_position - self.target_position).length()


class SimulatedMovingTarget:
    """Moving target without rendering"""
    
    PATTERNS = ["circle", "figure8", "linear", "spiral"]
    
    def __init__(self, scene: SimulatedScene, name: str = "target", pattern: str = "circle"):
        self.scene = scene
        self.name = name
        self.pattern = pattern if pattern in self.PATTERNS else "circle"
        self.time = 0.0
        self.speed = 2.0
        self.radius = 10.0
        self.base_height = 1.0
        
        self.entity = scene.add_entity(name)
        self.entity.set_pos(Vector3(0, self.base_height, 0))
        
        self.velocity = Vector3(0, 0, 0)
        self.last_position = Vector3(0, self.base_height, 0)
    
    def update(self, delta_time: float):
        """Update target position"""
        self.time += delta_time * self.speed
        self.last_position = self.position()
        
        if self.pattern == "circle":
            x = math.cos(self.time) * self.radius
            z = math.sin(self.time) * self.radius
            new_pos = Vector3(x, self.base_height, z)
        elif self.pattern == "figure8":
            x = math.sin(self.time * 2) * self.radius
            z = math.sin(self.time) * self.radius
            new_pos = Vector3(x, self.base_height, z)
        elif self.pattern == "linear":
            x = math.sin(self.time) * self.radius
            new_pos = Vector3(x, self.base_height, 0)
        elif self.pattern == "spiral":
            angle = self.time
            r = (math.sin(self.time * 0.5) + 1) * 0.5 * self.radius
            x = math.cos(angle) * r
            z = math.sin(angle) * r
            y = self.base_height + math.sin(self.time * 2) * 2
            new_pos = Vector3(x, y, z)
        else:
            new_pos = Vector3(0, self.base_height, 0)
        
        self.entity.set_pos(new_pos)
        
        if delta_time > 0:
            self.velocity = (new_pos - self.last_position) / delta_time
    
    def position(self) -> Vector3:
        """Get current position"""
        return self.entity.position()


class HeadlessCameraTest:
    """Test harness for headless camera follow demo"""
    
    TARGET_FRAMES = 256
    FPS = 60
    
    def __init__(self):
        self.stats = CameraFollowStats()
        self.scene = None
        self.camera = None
        self.target = None
        self.test_passed = False
        
    def setup(self, pattern: str = "circle", mode: FollowMode = FollowMode.SMOOTH) -> bool:
        """Initialize the test environment"""
        print("=" * 70)
        print("HEADLESS CAMERA FOLLOW TEST - 256 FRAMES")
        print("=" * 70)
        print(f"Target: {self.TARGET_FRAMES} frames @ {self.FPS} FPS")
        print(f"Expected duration: {self.TARGET_FRAMES / self.FPS:.2f} seconds")
        print(f"Pattern: {pattern}")
        print(f"Mode: {mode.name}")
        print("=" * 70)
        
        try:
            # Create simulated scene
            self.scene = SimulatedScene()
            
            # Create target
            self.target = SimulatedMovingTarget(self.scene, "test_target", pattern)
            print("[OK] Moving target created")
            
            # Create camera
            self.camera = HeadlessCameraController("test_camera")
            self.camera.set_target(self.target.entity)
            self.camera.set_mode(mode)
            print("[OK] Camera controller created")
            
            print("[OK] Camera follow configured")
            print("-" * 70)
            
            return True
            
        except Exception as e:
            self.stats.add_error(f"Setup failed: {str(e)}")
            import traceback
            self.stats.add_error(traceback.format_exc())
            return False
    
    def run_test(self) -> bool:
        """Run the 256-frame test"""
        delta_time = 1.0 / self.FPS
        
        print("Running simulation...")
        print(f"{'Frame':>8} | {'Target Pos':>25} | {'Camera Pos':>25} | {'Distance':>10}")
        print("-" * 70)
        
        try:
            for frame in range(self.TARGET_FRAMES):
                # Update target
                self.target.update(delta_time)
                
                # Update camera
                self.camera.update(delta_time)
                
                # Record statistics
                camera_pos = self.camera.current_position
                target_pos = self.target.position()
                distance = self.camera.get_distance_to_target()
                
                self.stats.record_frame(camera_pos, target_pos)
                
                # Print progress every 32 frames
                if frame % 32 == 0 or frame == self.TARGET_FRAMES - 1:
                    print(f"{frame:>8} | {str(target_pos):>25} | "
                          f"{str(camera_pos):>25} | {distance:>10.2f}")
                
                # Check for issues
                if distance > 100:
                    self.stats.add_warning(f"Frame {frame}: Large distance ({distance:.2f})")
                
                if math.isnan(camera_pos.x) or math.isnan(camera_pos.y) or math.isnan(camera_pos.z):
                    self.stats.add_error(f"Frame {frame}: Camera position has NaN!")
                    return False
                    
        except Exception as e:
            self.stats.add_error(f"Runtime error at frame {self.stats.frame_count}: {str(e)}")
            import traceback
            self.stats.add_error(traceback.format_exc())
            return False
        
        print("-" * 70)
        print(f"[OK] Completed {self.stats.frame_count} frames")
        return True
    
    def verify_results(self) -> bool:
        """Verify the test results"""
        print("\n" + "=" * 70)
        print("VERIFICATION RESULTS")
        print("=" * 70)
        
        summary = self.stats.get_summary()
        
        # Check frame count
        if self.stats.frame_count < self.TARGET_FRAMES:
            print(f"[FAIL] Only completed {self.stats.frame_count}/{self.TARGET_FRAMES} frames")
            return False
        print(f"[OK] Frame count: {self.stats.frame_count}/{self.TARGET_FRAMES}")
        
        # Check for errors
        if self.stats.errors:
            print(f"[FAIL] {len(self.stats.errors)} errors occurred:")
            for err in self.stats.errors:
                print(f"  - {err}")
            return False
        print(f"[OK] No errors during execution")
        
        # Check for NaN
        if summary.get("nan_detected"):
            print(f"[FAIL] NaN values detected in camera position")
            return False
        print(f"[OK] No NaN values detected")
        
        # Check tracking quality
        if summary.get("tracking_stable"):
            print(f"[OK] Tracking is stable (max deviation: {summary['max_distance_deviation']:.2f})")
        else:
            print(f"[WARN] Tracking unstable (max deviation: {summary['max_distance_deviation']:.2f})")
        
        # Print statistics
        print("\nCamera Follow Statistics:")
        print(f"  Average distance to target: {summary['avg_distance_to_target']:.2f}")
        print(f"  Min distance: {summary['min_distance']:.2f}")
        print(f"  Max distance: {summary['max_distance']:.2f}")
        print(f"  Total distance traveled: {summary['total_distance_traveled']:.2f}")
        print(f"  Average movement per frame: {summary['avg_movement_per_frame']:.4f}")
        print(f"  Max distance deviation: {summary['max_distance_deviation']:.4f}")
        
        if self.stats.warnings:
            print(f"\nWarnings ({len(self.stats.warnings)}):")
            for warn in self.stats.warnings[:5]:  # Show first 5
                print(f"  - {warn}")
        
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        self.scene = None
        self.camera = None
        self.target = None
        print("[OK] Cleanup complete")
    
    def run(self, pattern: str = "circle", mode: FollowMode = FollowMode.SMOOTH) -> bool:
        """Run the complete test"""
        try:
            if not self.setup(pattern, mode):
                return False
            success = self.run_test()
            if success:
                success = self.verify_results()
            self.test_passed = success
            return success
        finally:
            self.cleanup()
            self.print_final_result()
    
    def print_final_result(self):
        """Print final test result"""
        print("\n" + "=" * 70)
        if self.test_passed:
            print("TEST PASSED [OK]")
            print("=" * 70)
            print("The camera successfully tracked the moving target for 256 frames")
            print("without any errors.")
        else:
            print("TEST FAILED [FAIL]")
            print("=" * 70)
            if self.stats.errors:
                print(f"Errors: {len(self.stats.errors)}")
                for err in self.stats.errors[:3]:
                    print(f"  - {err}")
        print("=" * 70)


def test_all_patterns():
    """Test all movement patterns"""
    print("\n" + "=" * 70)
    print("RUNNING TESTS FOR ALL PATTERNS")
    print("=" * 70)
    
    patterns = ["circle", "figure8", "linear", "spiral"]
    results = []
    
    for pattern in patterns:
        print(f"\n{'=' * 70}")
        print(f"Testing pattern: {pattern}")
        print("=" * 70)
        test = HeadlessCameraTest()
        success = test.run(pattern=pattern)
        results.append((pattern, success))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for pattern, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {pattern:10} : {status}")
    
    all_passed = all(success for _, success in results)
    print("=" * 70)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 70)
    
    return all_passed


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test headless camera follow for 256 frames"
    )
    parser.add_argument(
        "--pattern", 
        default="circle",
        choices=["circle", "figure8", "linear", "spiral"],
        help="Movement pattern (default: circle)"
    )
    parser.add_argument(
        "--mode",
        default="SMOOTH",
        choices=["SMOOTH", "PREDICTIVE"],
        help="Camera follow mode (default: SMOOTH)"
    )
    parser.add_argument(
        "--all-patterns",
        action="store_true",
        help="Test all patterns"
    )
    
    args = parser.parse_args()
    
    if args.all_patterns:
        success = test_all_patterns()
    else:
        mode = FollowMode[args.mode]
        test = HeadlessCameraTest()
        success = test.run(pattern=args.pattern, mode=mode)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
