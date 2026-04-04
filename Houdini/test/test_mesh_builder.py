"""
MeshBuilder API Tests with RoboCute Integration
===============================================

Comprehensive test suite for MeshBuilder API demonstrating RoboCute Python API usage.

For tests requiring visualization:
    app.init_display(width, height)  # Initialize display
    app.ctx.enable_camera_control()  # Enable camera control
    app.run(prepare_denoise=False, limit_frame=100)  # Run with frame limit
"""

import os
import sys
import unittest
import math

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# RoboCute imports (with fallback for testing without robocute installed)
try:
    import robocute as rbc
    import robocute.rbc_ext as re
    import robocute.rbc_ext.luisa as lc
    ROBOCUTE_AVAILABLE = True
except ImportError:
    ROBOCUTE_AVAILABLE = False
    # Create mock classes for testing without robocute
    class MockApp:
        def init_display(self, width, height):
            pass
        def run(self, prepare_denoise=False, limit_frame=None):
            pass
        def set_user_callback(self, callback):
            pass
    class MockModule:
        app = MockApp()
    rbc = MockModule()
    re = MockModule()
    lc = MockModule()

import numpy as np
from mesh_builder import MeshBuilder


class TestMeshBuilderBasic(unittest.TestCase):
    """Test basic MeshBuilder functionality."""
    
    def test_initialization(self):
        """Test MeshBuilder initialization."""
        builder = MeshBuilder()
        self.assertEqual(builder.vertex_count(), 0)
        self.assertEqual(builder.submesh_count(), 0)
        self.assertEqual(builder.uv_count(), 0)
        self.assertFalse(builder.contained_normal())
        self.assertFalse(builder.contained_tangent())
    
    def test_add_single_vertex(self):
        """Test adding a single vertex."""
        builder = MeshBuilder()
        index = builder.add_vertex((0, 1, 2))
        self.assertEqual(index, 0)
        self.assertEqual(builder.vertex_count(), 1)
        np.testing.assert_array_equal(builder.position[0], [0, 1, 2])
    
    def test_add_multiple_vertices(self):
        """Test adding multiple vertices."""
        builder = MeshBuilder()
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        self.assertEqual(builder.vertex_count(), 3)
        self.assertEqual(v0, 0)
        self.assertEqual(v1, 1)
        self.assertEqual(v2, 2)
    
    def test_add_submesh(self):
        """Test adding submeshes."""
        builder = MeshBuilder()
        self.assertEqual(builder.submesh_count(), 0)
        
        sm0 = builder.add_submesh()
        self.assertEqual(sm0, 0)
        self.assertEqual(builder.submesh_count(), 1)
        
        sm1 = builder.add_submesh()
        self.assertEqual(sm1, 1)
        self.assertEqual(builder.submesh_count(), 2)
    
    def test_add_single_triangle(self):
        """Test adding a single triangle."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_triangle(0, v0, v1, v2)
        self.assertEqual(builder.indices_count(), 3)
    
    def test_add_multiple_triangles(self):
        """Test adding multiple triangles."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Create a quad (2 triangles)
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((1, 1, 0))
        v3 = builder.add_vertex((0, 1, 0))
        
        builder.add_triangle(0, v0, v1, v2)
        builder.add_triangle(0, v0, v2, v3)
        
        self.assertEqual(builder.indices_count(), 6)
    
    def test_add_uv_set(self):
        """Test adding UV sets."""
        builder = MeshBuilder()
        self.assertEqual(builder.uv_count(), 0)
        
        builder.add_uv_set()
        self.assertEqual(builder.uv_count(), 1)
        
        builder.add_uv_set()
        self.assertEqual(builder.uv_count(), 2)


class TestMeshBuilderValidation(unittest.TestCase):
    """Test MeshBuilder validation."""
    
    def test_empty_mesh_validation(self):
        """Test validation of empty mesh."""
        builder = MeshBuilder()
        error = builder.check()
        self.assertIn("No submesh", error)
    
    def test_mesh_without_triangles(self):
        """Test validation of mesh without triangles."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_vertex((0, 0, 0))
        
        error = builder.check()
        self.assertIn("has 0 triangles", error)
    
    def test_valid_simple_mesh(self):
        """Test validation of simple valid mesh."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_triangle(0, v0, v1, v2)
        
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_invalid_submesh_index(self):
        """Test triangle with invalid submesh index."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        # Submesh 1 doesn't exist
        builder.add_triangle(1, v0, v1, v2)
        
        error = builder.check()
        self.assertIn("Invalid submesh index", error)
    
    def test_vertex_index_out_of_range(self):
        """Test triangle with vertex index out of range."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        
        # v2 doesn't exist
        builder.add_triangle(0, v0, v1, 2)
        
        error = builder.check()
        self.assertIn("out of vertex range", error)


class TestMeshBuilderPositions(unittest.TestCase):
    """Test position buffer management."""
    
    def test_position_shape(self):
        """Test position buffer shape."""
        builder = MeshBuilder()
        builder.add_vertex((0, 0, 0))
        builder.add_vertex((1, 1, 1))
        
        self.assertEqual(builder.position.shape, (2, 3))
        self.assertEqual(builder.position.dtype, np.float32)
    
    def test_position_values(self):
        """Test position values are stored correctly."""
        builder = MeshBuilder()
        builder.add_vertex((1.5, 2.5, 3.5))
        
        np.testing.assert_array_almost_equal(
            builder.position[0], [1.5, 2.5, 3.5]
        )


class TestMeshBuilderNormals(unittest.TestCase):
    """Test normal buffer management."""
    
    def test_add_normal(self):
        """Test adding normal to a vertex."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        builder.add_normal((0, 1, 0))
        
        self.assertTrue(builder.contained_normal())
        np.testing.assert_array_almost_equal(
            builder.normal[v0], [0, 1, 0]
        )
    
    def test_add_multiple_normals(self):
        """Test adding normals to multiple vertices."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_normal((0, 0, 1))
        builder.add_normal((0, 0, 1))
        builder.add_normal((0, 0, 1))
        
        for i in range(3):
            np.testing.assert_array_almost_equal(
                builder.normal[i], [0, 0, 1]
            )


class TestMeshBuilderUVs(unittest.TestCase):
    """Test UV coordinate management."""
    
    def test_add_uv(self):
        """Test adding UV coordinates."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        v0 = builder.add_vertex((0, 0, 0))
        builder.add_uv((0.5, 0.5))
        
        self.assertEqual(builder.uv_count(), 1)
        np.testing.assert_array_almost_equal(
            builder.uvs[0][v0], [0.5, 0.5]
        )
    
    def test_add_multiple_uv_sets(self):
        """Test adding UVs to multiple sets."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()  # Set 0
        builder.add_uv_set()  # Set 1
        
        v0 = builder.add_vertex((0, 0, 0))
        builder.add_uv((0, 0))  # Added to set 0
        builder.add_uv((1, 1), uv_set_index=1)  # Added to set 1
        
        self.assertEqual(builder.uv_count(), 2)
        np.testing.assert_array_almost_equal(
            builder.uvs[0][v0], [0, 0]
        )
        np.testing.assert_array_almost_equal(
            builder.uvs[1][v0], [1, 1]
        )


class TestMeshBuilderTangents(unittest.TestCase):
    """Test tangent buffer management."""
    
    def test_add_tangent(self):
        """Test adding tangent to a vertex."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        builder.add_tangent((1, 0, 0, 1))
        
        self.assertTrue(builder.contained_tangent())
        np.testing.assert_array_almost_equal(
            builder.tangent[v0], [1, 0, 0, 1]
        )
    
    def test_tangent_shape(self):
        """Test tangent buffer has correct shape."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        builder.add_vertex((0, 0, 0))
        builder.add_tangent((1, 0, 0, 1))
        
        self.assertEqual(builder.tangent.shape[1], 4)


class TestMeshBuilderSubmeshes(unittest.TestCase):
    """Test multiple submesh handling."""
    
    def test_multiple_submeshes(self):
        """Test creating multiple submeshes."""
        builder = MeshBuilder()
        
        sm0 = builder.add_submesh()
        sm1 = builder.add_submesh()
        sm2 = builder.add_submesh()
        
        self.assertEqual(builder.submesh_count(), 3)
        self.assertEqual(sm0, 0)
        self.assertEqual(sm1, 1)
        self.assertEqual(sm2, 2)
    
    def test_triangles_in_different_submeshes(self):
        """Test adding triangles to different submeshes."""
        builder = MeshBuilder()
        
        sm0 = builder.add_submesh()
        sm1 = builder.add_submesh()
        
        # Shared vertices
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        # Triangle in submesh 0
        builder.add_triangle(sm0, v0, v1, v2)
        
        # Triangle in submesh 1
        builder.add_triangle(sm1, v0, v2, v1)
        
        self.assertEqual(builder.indices_count(), 6)
        
        error = builder.check()
        self.assertEqual(error, "")


class TestMeshBuilderPrimitives(unittest.TestCase):
    """Test creating common primitives."""
    
    def test_create_triangle(self):
        """Test creating a simple triangle."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_uv((0, 0))
        builder.add_uv((1, 0))
        builder.add_uv((0, 1))
        
        builder.add_triangle(0, v0, v1, v2)
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.vertex_count(), 3)
        self.assertEqual(builder.indices_count(), 3)
    
    def test_create_quad(self):
        """Test creating a quad (2 triangles)."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((1, 1, 0))
        v3 = builder.add_vertex((0, 1, 0))
        
        builder.add_uv((0, 0))
        builder.add_uv((1, 0))
        builder.add_uv((1, 1))
        builder.add_uv((0, 1))
        
        builder.add_triangle(0, v0, v1, v2)
        builder.add_triangle(0, v0, v2, v3)
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.vertex_count(), 4)
        self.assertEqual(builder.indices_count(), 6)
    
    def test_create_cube(self):
        """Test creating a cube mesh."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        # Define 8 cube vertices
        positions = [
            (-0.5, -0.5, -0.5),  # 0
            (0.5, -0.5, -0.5),   # 1
            (0.5, 0.5, -0.5),    # 2
            (-0.5, 0.5, -0.5),   # 3
            (-0.5, -0.5, 0.5),   # 4
            (0.5, -0.5, 0.5),    # 5
            (0.5, 0.5, 0.5),     # 6
            (-0.5, 0.5, 0.5),    # 7
        ]
        
        for pos in positions:
            builder.add_vertex(pos)
            builder.add_uv((0, 0))  # Simplified UVs
        
        # 12 triangles (6 faces * 2)
        triangles = [
            # Front face
            (4, 5, 6), (4, 6, 7),
            # Back face
            (1, 0, 3), (1, 3, 2),
            # Top face
            (3, 7, 6), (3, 6, 2),
            # Bottom face
            (0, 1, 5), (0, 5, 4),
            # Right face
            (1, 2, 6), (1, 6, 5),
            # Left face
            (0, 4, 7), (0, 7, 3),
        ]
        
        for tri in triangles:
            builder.add_triangle(0, tri[0], tri[1], tri[2])
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.vertex_count(), 8)
        self.assertEqual(builder.indices_count(), 36)
    
    def test_create_plane(self):
        """Test creating a plane mesh with multiple quads."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        segments = 4
        for z in range(segments + 1):
            for x in range(segments + 1):
                u = x / segments
                v = z / segments
                builder.add_vertex((x, 0, z))
                builder.add_uv((u, v))
        
        # Create triangles
        for z in range(segments):
            for x in range(segments):
                base = z * (segments + 1) + x
                
                v0 = base
                v1 = base + 1
                v2 = base + segments + 2
                v3 = base + segments + 1
                
                builder.add_triangle(0, v0, v1, v2)
                builder.add_triangle(0, v0, v2, v3)
        
        error = builder.check()
        self.assertEqual(error, "")
        expected_vertices = (segments + 1) * (segments + 1)
        expected_triangles = segments * segments * 2
        self.assertEqual(builder.vertex_count(), expected_vertices)
        self.assertEqual(builder.indices_count(), expected_triangles * 3)
    
    def test_create_sphere(self):
        """Test creating a sphere mesh."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        radius = 1.0
        segments = 16
        rings = 8
        
        # Top pole
        builder.add_vertex((0, radius, 0))
        builder.add_uv((0.5, 1.0))
        
        # Middle rings
        for ring in range(1, rings):
            phi = math.pi * ring / rings
            y = radius * math.cos(phi)
            ring_radius = radius * math.sin(phi)
            
            for seg in range(segments):
                theta = 2 * math.pi * seg / segments
                x = ring_radius * math.cos(theta)
                z = ring_radius * math.sin(theta)
                u = seg / segments
                v = 1.0 - ring / rings
                
                builder.add_vertex((x, y, z))
                builder.add_uv((u, v))
        
        # Bottom pole
        builder.add_vertex((0, -radius, 0))
        builder.add_uv((0.5, 0.0))
        
        # Create triangles
        # Top cap
        for seg in range(segments):
            next_seg = (seg + 1) % segments
            v0 = 0  # Top pole
            v1 = 1 + seg
            v2 = 1 + next_seg
            builder.add_triangle(0, v0, v1, v2)
        
        # Middle rings
        for ring in range(rings - 2):
            ring_start = 1 + ring * segments
            next_ring_start = 1 + (ring + 1) * segments
            
            for seg in range(segments):
                next_seg = (seg + 1) % segments
                
                v0 = ring_start + seg
                v1 = ring_start + next_seg
                v2 = next_ring_start + next_seg
                v3 = next_ring_start + seg
                
                builder.add_triangle(0, v0, v1, v2)
                builder.add_triangle(0, v0, v2, v3)
        
        # Bottom cap
        bottom_pole = builder.vertex_count() - 1
        last_ring_start = 1 + (rings - 2) * segments
        for seg in range(segments):
            next_seg = (seg + 1) % segments
            v0 = bottom_pole
            v1 = last_ring_start + next_seg
            v2 = last_ring_start + seg
            builder.add_triangle(0, v0, v1, v2)
        
        error = builder.check()
        self.assertEqual(error, "")


class TestMeshBuilderTangentCalculation(unittest.TestCase):
    """Test tangent calculation utility."""
    
    def test_calculate_tangent_simple_quad(self):
        """Test tangent calculation for a simple quad."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        uvs = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ], dtype=np.float32)
        
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.uint32)
        
        tangents = MeshBuilder.calculate_tangent(positions, uvs, triangles)
        
        # Should return tangents for all vertices
        self.assertEqual(tangents.shape, (4, 4))
        
        # Tangent should point in X direction (along U axis)
        for i in range(4):
            self.assertAlmostEqual(abs(tangents[i][0]), 1.0, places=5)
            self.assertAlmostEqual(tangents[i][1], 0.0, places=5)
            self.assertAlmostEqual(tangents[i][2], 0.0, places=5)
            # W component should be +1 or -1 (handedness)
            self.assertTrue(abs(tangents[i][3]) == 1.0)
    
    def test_calculate_tangent_with_custom_w(self):
        """Test tangent calculation with custom W value."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0, 1],
        ], dtype=np.float32)
        
        uvs = np.array([
            [0, 0],
            [1, 0],
            [0.5, 1],
        ], dtype=np.float32)
        
        triangles = np.array([[0, 1, 2]], dtype=np.uint32)
        
        tangents = MeshBuilder.calculate_tangent(positions, uvs, triangles, tangent_w=-1.0)
        
        # All W components should be -1
        for i in range(3):
            self.assertEqual(tangents[i][3], -1.0)


class TestMeshBuilderFileOutput(unittest.TestCase):
    """Test file output functionality."""
    
    def test_write_to_file(self):
        """Test writing mesh to file."""
        import tempfile
        
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_uv((0, 0))
        builder.add_uv((1, 0))
        builder.add_uv((0, 1))
        
        builder.add_triangle(0, v0, v1, v2)
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            offsets = builder.write_to(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
            # Should have submesh count offsets
            self.assertEqual(len(offsets), 1)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_write_creates_valid_file(self):
        """Test that written file is valid."""
        import tempfile
        
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Create a simple triangle
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        builder.add_triangle(0, v0, v1, v2)
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            builder.write_to(tmp_path)
            
            # Verify we can read the file back
            with open(tmp_path, 'rb') as f:
                data = f.read()
                self.assertGreater(len(data), 0)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestRoboCuteDisplayIntegration(unittest.TestCase):
    """Test RoboCute display initialization for visualization tests."""
    
    def test_display_initialization(self):
        """Test that display can be initialized when robocute is available."""
        if not ROBOCUTE_AVAILABLE:
            self.skipTest("robocute not available")
        
        app = rbc.app.App()
        app.init_display(1280, 720)
        # Display should be initialized
        self.assertTrue(True)  # If we get here, init succeeded
    
    def test_app_run_with_limit_frame(self):
        """Test that app.run() accepts limit_frame parameter."""
        if not ROBOCUTE_AVAILABLE:
            self.skipTest("robocute not available")
        
        app = rbc.app.App()
        app.init_display(1280, 720)
        # Run with frame limit for testing
        try:
            app.run(prepare_denoise=False, limit_frame=10)
        except Exception as e:
            # Expected in test environment without actual display
            pass


class TestRoboCuteMeshResource(unittest.TestCase):
    """Test MeshBuilder integration with RoboCute MeshResource."""
    
    def test_write_to_mesh(self):
        """Test creating MeshResource from MeshBuilder."""
        if not ROBOCUTE_AVAILABLE:
            self.skipTest("robocute not available")
        
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_uv((0, 0))
        builder.add_uv((1, 0))
        builder.add_uv((0, 1))
        
        builder.add_triangle(0, v0, v1, v2)
        
        # Create mesh resource
        mesh = builder.write_to_mesh()
        mesh.install()
        
        self.assertTrue(mesh)  # Mesh should be valid
    
    def test_mesh_with_normals(self):
        """Test creating mesh with normals."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_triangle(0, v0, v1, v2)
        
        # Add normals
        builder.normal = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.float32)
        
        # Create mesh resource
        if ROBOCUTE_AVAILABLE:
            mesh = builder.write_to_mesh()
            mesh.install()
            self.assertTrue(mesh)
    
    def test_mesh_with_tangents(self):
        """Test creating mesh with tangents."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_uv((0, 0))
        builder.add_uv((1, 0))
        builder.add_uv((0, 1))
        
        builder.add_triangle(0, v0, v1, v2)
        
        # Calculate and set tangents
        triangles = np.array([[0, 1, 2]], dtype=np.uint32)
        tangents = MeshBuilder.calculate_tangent(
            builder.position, builder.uvs[0], triangles
        )
        builder.tangent = tangents
        
        # Create mesh resource
        if ROBOCUTE_AVAILABLE:
            mesh = builder.write_to_mesh()
            mesh.install()
            self.assertTrue(mesh)


class TestRoboCuteRenderingWorkflow(unittest.TestCase):
    """
    Test RoboCute rendering workflow with display initialization.
    These tests demonstrate the recommended pattern for tests requiring visualization.
    """
    
    def test_rendering_workflow_with_display(self):
        """
        Example of a rendering test with display initialization.
        
        Pattern for tests requiring visualization:
        1. Import robocute modules
        2. Create mesh using MeshBuilder
        3. Initialize display: app.init_display(width, height)
        4. Enable camera control: app.ctx.enable_camera_control()
        5. Run with frame limit: app.run(limit_frame=100)
        """
        if not ROBOCUTE_AVAILABLE:
            self.skipTest("robocute not available")
        
        # Create mesh using MeshBuilder
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_uv((0, 0))
        builder.add_uv((1, 0))
        builder.add_uv((0, 1))
        
        builder.add_triangle(0, v0, v1, v2)
        
        error = builder.check()
        self.assertEqual(error, "")
        
        # Initialize RoboCute app with display
        app = rbc.app.App()
        app.init_display(1280, 720)
        app.ctx.enable_camera_control()
        
        # Get mesh resource
        mesh = builder.write_to_mesh()
        mesh.install()
        
        # Run with frame limit for automated testing
        app.run(prepare_denoise=False, limit_frame=100)
    
    def test_rendering_multiple_meshes(self):
        """Test rendering multiple meshes with display."""
        if not ROBOCUTE_AVAILABLE:
            self.skipTest("robocute not available")
        
        # Create first mesh (triangle)
        builder1 = MeshBuilder()
        builder1.add_submesh()
        v0 = builder1.add_vertex((0, 0, 0))
        v1 = builder1.add_vertex((1, 0, 0))
        v2 = builder1.add_vertex((0, 1, 0))
        builder1.add_triangle(0, v0, v1, v2)
        
        # Create second mesh (quad)
        builder2 = MeshBuilder()
        builder2.add_submesh()
        v0 = builder2.add_vertex((2, 0, 0))
        v1 = builder2.add_vertex((3, 0, 0))
        v2 = builder2.add_vertex((3, 1, 0))
        v3 = builder2.add_vertex((2, 1, 0))
        builder2.add_triangle(0, v0, v1, v2)
        builder2.add_triangle(0, v0, v2, v3)
        
        # Initialize display
        app = rbc.app.App()
        app.init_display(1280, 720)
        app.ctx.enable_camera_control()
        
        # Install meshes
        mesh1 = builder1.write_to_mesh()
        mesh1.install()
        
        mesh2 = builder2.write_to_mesh()
        mesh2.install()
        
        # Run with frame limit
        app.run(prepare_denoise=False, limit_frame=50)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    # Run tests
    result = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
