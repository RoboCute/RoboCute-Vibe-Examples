"""
Comprehensive test suite for MeshBuilder API.

This test file covers:
- MeshBuilder initialization
- Vertex/submesh/UV management
- Validation (check method)
- File output (write_to)
- MeshResource output (write_to_mesh)
- Tangent calculation
- Edge cases

Usage:
    cd samples/Houdini
    python -m pytest test/test_mesh_builder.py -v
    python test/test_mesh_builder.py  # Run directly
"""

import sys
import os
import unittest
import tempfile
import numpy as np
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mesh_builder import MeshBuilder


class TestMeshBuilderInitialization(unittest.TestCase):
    """Test MeshBuilder initialization."""
    
    def test_default_initialization(self):
        """Test creating MeshBuilder with default values."""
        builder = MeshBuilder()
        
        self.assertEqual(builder.vertex_count(), 0)
        self.assertEqual(builder.submesh_count(), 0)
        self.assertEqual(builder.uv_count(), 0)
        self.assertEqual(builder.indices_count(), 0)
        self.assertFalse(builder.contained_normal())
        self.assertFalse(builder.contained_tangent())
        
        # Check numpy arrays are initialized correctly
        self.assertEqual(builder.position.shape, (0, 3))
        self.assertEqual(builder.normal.shape, (0, 3))
        self.assertEqual(builder.tangent.shape, (0, 4))
        self.assertEqual(len(builder.uvs), 0)
        self.assertEqual(len(builder.triangle_indices), 0)
    
    def test_repr(self):
        """Test string representation."""
        builder = MeshBuilder()
        repr_str = repr(builder)
        
        self.assertIn("MeshBuilder", repr_str)
        self.assertIn("vertices=0", repr_str)
        self.assertIn("submeshes=0", repr_str)
        self.assertIn("indices=0", repr_str)
        self.assertIn("normals=False", repr_str)
        self.assertIn("tangents=False", repr_str)
        self.assertIn("uv_sets=0", repr_str)


class TestVertexManagement(unittest.TestCase):
    """Test vertex management methods."""
    
    def test_add_single_vertex(self):
        """Test adding a single vertex."""
        builder = MeshBuilder()
        idx = builder.add_vertex((1.0, 2.0, 3.0))
        
        self.assertEqual(idx, 0)
        self.assertEqual(builder.vertex_count(), 1)
        np.testing.assert_array_almost_equal(
            builder.position[0], [1.0, 2.0, 3.0]
        )
    
    def test_add_multiple_vertices(self):
        """Test adding multiple vertices."""
        builder = MeshBuilder()
        
        idx0 = builder.add_vertex((0.0, 0.0, 0.0))
        idx1 = builder.add_vertex((1.0, 0.0, 0.0))
        idx2 = builder.add_vertex((0.0, 1.0, 0.0))
        
        self.assertEqual(idx0, 0)
        self.assertEqual(idx1, 1)
        self.assertEqual(idx2, 2)
        self.assertEqual(builder.vertex_count(), 3)
    
    def test_add_vertex_with_list(self):
        """Test adding vertex with list input."""
        builder = MeshBuilder()
        idx = builder.add_vertex([5.0, 10.0, 15.0])
        
        self.assertEqual(idx, 0)
        np.testing.assert_array_almost_equal(
            builder.position[0], [5.0, 10.0, 15.0]
        )
    
    def test_add_vertex_with_numpy_array(self):
        """Test adding vertex with numpy array input."""
        builder = MeshBuilder()
        pos = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        idx = builder.add_vertex(pos)
        
        self.assertEqual(idx, 0)
        np.testing.assert_array_almost_equal(builder.position[0], pos)


class TestSubmeshManagement(unittest.TestCase):
    """Test submesh management methods."""
    
    def test_add_single_submesh(self):
        """Test adding a single submesh."""
        builder = MeshBuilder()
        idx = builder.add_submesh()
        
        self.assertEqual(idx, 0)
        self.assertEqual(builder.submesh_count(), 1)
        self.assertEqual(len(builder.triangle_indices), 1)
        self.assertEqual(builder.triangle_indices[0].shape[0], 0)
    
    def test_add_multiple_submeshes(self):
        """Test adding multiple submeshes."""
        builder = MeshBuilder()
        
        idx0 = builder.add_submesh()
        idx1 = builder.add_submesh()
        idx2 = builder.add_submesh()
        
        self.assertEqual(idx0, 0)
        self.assertEqual(idx1, 1)
        self.assertEqual(idx2, 2)
        self.assertEqual(builder.submesh_count(), 3)
    
    def test_get_submesh_indices(self):
        """Test getting submesh indices."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add vertices
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        # Add triangle
        builder.add_triangle(0, v0, v1, v2)
        
        # Get indices
        indices = builder.get_submesh_indices(0)
        self.assertEqual(len(indices), 3)
        self.assertEqual(indices[0], v0)
        self.assertEqual(indices[1], v1)
        self.assertEqual(indices[2], v2)
    
    def test_get_submesh_indices_out_of_range(self):
        """Test getting submesh indices with invalid index."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        with self.assertRaises(IndexError):
            builder.get_submesh_indices(5)
        
        with self.assertRaises(IndexError):
            builder.get_submesh_indices(-1)


class TestTriangleManagement(unittest.TestCase):
    """Test triangle management methods."""
    
    def test_add_single_triangle(self):
        """Test adding a single triangle."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_triangle(0, v0, v1, v2)
        
        self.assertEqual(builder.indices_count(), 3)
        np.testing.assert_array_equal(
            builder.triangle_indices[0],
            np.array([v0, v1, v2], dtype=np.uint32)
        )
    
    def test_add_multiple_triangles(self):
        """Test adding multiple triangles to same submesh."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Create a quad from two triangles
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((1, 1, 0))
        v3 = builder.add_vertex((0, 1, 0))
        
        builder.add_triangle(0, v0, v1, v2)
        builder.add_triangle(0, v0, v2, v3)
        
        self.assertEqual(builder.indices_count(), 6)
        self.assertEqual(len(builder.triangle_indices[0]), 6)
    
    def test_add_triangle_to_multiple_submeshes(self):
        """Test adding triangles to different submeshes."""
        builder = MeshBuilder()
        
        submesh0 = builder.add_submesh()
        submesh1 = builder.add_submesh()
        
        # Add vertices (shared)
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        # Add triangle to each submesh
        builder.add_triangle(submesh0, v0, v1, v2)
        builder.add_triangle(submesh1, v0, v2, v1)  # Reversed winding
        
        self.assertEqual(builder.indices_count(), 6)
        self.assertEqual(len(builder.triangle_indices[0]), 3)
        self.assertEqual(len(builder.triangle_indices[1]), 3)
    
    def test_add_triangle_invalid_submesh(self):
        """Test adding triangle to invalid submesh."""
        builder = MeshBuilder()
        
        with self.assertRaises(IndexError):
            builder.add_triangle(0, 0, 1, 2)
        
        builder.add_submesh()
        with self.assertRaises(IndexError):
            builder.add_triangle(5, 0, 1, 2)


class TestUVManagement(unittest.TestCase):
    """Test UV set management methods."""
    
    def test_add_uv_set(self):
        """Test adding UV sets."""
        builder = MeshBuilder()
        
        idx0 = builder.add_uv_set()
        self.assertEqual(idx0, 0)
        self.assertEqual(builder.uv_count(), 1)
        
        idx1 = builder.add_uv_set()
        self.assertEqual(idx1, 1)
        self.assertEqual(builder.uv_count(), 2)
    
    def test_uv_array_shape(self):
        """Test UV array is initialized with correct shape."""
        builder = MeshBuilder()
        builder.add_uv_set()
        
        self.assertEqual(builder.uvs[0].shape, (0, 2))


class TestValidation(unittest.TestCase):
    """Test mesh validation (check method)."""
    
    def test_empty_mesh_error(self):
        """Test validation fails for empty mesh."""
        builder = MeshBuilder()
        
        error = builder.check()
        self.assertIn("No vertices", error)
    
    def test_valid_mesh(self):
        """Test validation passes for valid mesh."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        builder.add_triangle(0, v0, v1, v2)
        
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_invalid_index_count(self):
        """Test validation fails for incomplete triangles."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        
        # Manually add invalid index count
        builder.triangle_indices[0] = np.array([v0, v1], dtype=np.uint32)
        
        error = builder.check()
        self.assertIn("not divisible by 3", error)
    
    def test_out_of_range_index(self):
        """Test validation fails for out-of-range indices."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        # Add triangle with out-of-range index
        builder.add_triangle(0, v0, v1, 999)
        
        error = builder.check()
        self.assertIn("out of vertex range", error)
    
    def test_normal_size_mismatch(self):
        """Test validation fails for normal size mismatch."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add 3 vertices
        builder.add_vertex((0, 0, 0))
        builder.add_vertex((1, 0, 0))
        builder.add_vertex((0, 1, 0))
        
        # Add normals for only 2 vertices
        builder.normal = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        
        error = builder.check()
        self.assertIn("Normal size", error)
        self.assertIn("does not match position size", error)
    
    def test_tangent_size_mismatch(self):
        """Test validation fails for tangent size mismatch."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add 3 vertices
        builder.add_vertex((0, 0, 0))
        builder.add_vertex((1, 0, 0))
        builder.add_vertex((0, 1, 0))
        
        # Add tangents for only 2 vertices
        builder.tangent = np.array([[1, 0, 0, 1], [0, 1, 0, 1]], dtype=np.float32)
        
        error = builder.check()
        self.assertIn("Tangent size", error)
    
    def test_uv_size_mismatch(self):
        """Test validation fails for UV size mismatch."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        # Add 3 vertices
        builder.add_vertex((0, 0, 0))
        builder.add_vertex((1, 0, 0))
        builder.add_vertex((0, 1, 0))
        
        # Add UVs for only 2 vertices
        builder.uvs[0] = np.array([[0, 0], [1, 0]], dtype=np.float32)
        
        error = builder.check()
        self.assertIn("UV0 size", error)


class TestWriteToFile(unittest.TestCase):
    """Test file output (write_to method)."""
    
    def test_write_to_file_path(self):
        """Test writing mesh to file path."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        builder.add_triangle(0, v0, v1, v2)
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            offsets = builder.write_to(tmp_path)
            
            # Check file was created
            self.assertTrue(Path(tmp_path).exists())
            
            # Check offsets (single submesh = empty array)
            self.assertEqual(len(offsets), 0)
            
            # Check file has content
            file_size = Path(tmp_path).stat().st_size
            self.assertGreater(file_size, 0)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_write_to_bytearray(self):
        """Test writing mesh to bytearray."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        builder.add_triangle(0, v0, v1, v2)
        
        buffer = bytearray()
        initial_len = len(buffer)
        
        offsets = builder.write_to(buffer)
        
        self.assertGreater(len(buffer), initial_len)
        self.assertEqual(len(offsets), 0)
    
    def test_write_to_multiple_submeshes(self):
        """Test writing mesh with multiple submeshes."""
        builder = MeshBuilder()
        
        submesh0 = builder.add_submesh()
        submesh1 = builder.add_submesh()
        
        # Add vertices
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        # Add 2 triangles to first submesh
        builder.add_triangle(submesh0, v0, v1, v2)
        builder.add_triangle(submesh0, v0, v2, v1)
        
        # Add 1 triangle to second submesh
        builder.add_triangle(submesh1, v0, v1, v2)
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            offsets = builder.write_to(tmp_path)
            
            # Should have offsets for multiple submeshes
            self.assertEqual(len(offsets), 2)
            self.assertEqual(offsets[0], 0)  # First submesh starts at triangle 0
            self.assertEqual(offsets[1], 2)  # Second submesh starts at triangle 2
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_write_invalid_mesh_raises(self):
        """Test that writing invalid mesh raises ValueError."""
        builder = MeshBuilder()
        # Empty mesh should fail validation
        
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with self.assertRaises(ValueError) as context:
                builder.write_to(tmp_path)
            
            self.assertIn("Mesh validation failed", str(context.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    def test_write_to_invalid_type(self):
        """Test that writing to invalid type raises TypeError."""
        builder = MeshBuilder()
        builder.add_submesh()
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        builder.add_triangle(0, v0, v1, v2)
        
        with self.assertRaises(TypeError):
            builder.write_to(12345)


class TestTangentCalculation(unittest.TestCase):
    """Test tangent calculation static method."""
    
    def test_calculate_tangent_simple_triangle(self):
        """Test tangent calculation for simple triangle."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        uvs = np.array([
            [0, 0],
            [1, 0],
            [0, 1]
        ], dtype=np.float32)
        
        triangles = np.array([
            [0, 1, 2]
        ], dtype=np.uint32)
        
        tangents = MeshBuilder.calculate_tangent(positions, uvs, triangles)
        
        # Should return array of shape (N, 4)
        self.assertEqual(tangents.shape, (3, 4))
        
        # Tangents should be normalized (first 3 components)
        for i in range(3):
            tangent_len = np.linalg.norm(tangents[i, :3])
            self.assertAlmostEqual(tangent_len, 1.0, places=5)
        
        # W component should be set (handedness)
        self.assertEqual(tangents[0, 3], 1.0)
    
    def test_calculate_tangent_quad(self):
        """Test tangent calculation for quad (2 triangles)."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        uvs = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)
        
        triangles = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.uint32)
        
        tangents = MeshBuilder.calculate_tangent(positions, uvs, triangles)
        
        self.assertEqual(tangents.shape, (4, 4))
        
        # All tangents should be normalized
        for i in range(4):
            tangent_len = np.linalg.norm(tangents[i, :3])
            self.assertAlmostEqual(tangent_len, 1.0, places=5)
    
    def test_calculate_tangent_custom_w(self):
        """Test tangent calculation with custom W value."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        uvs = np.array([
            [0, 0],
            [1, 0],
            [0, 1]
        ], dtype=np.float32)
        
        triangles = np.array([[0, 1, 2]], dtype=np.uint32)
        
        tangents = MeshBuilder.calculate_tangent(positions, uvs, triangles, tangent_w=-1.0)
        
        self.assertEqual(tangents[0, 3], -1.0)
    
    def test_calculate_tangent_degenerate_uv(self):
        """Test tangent calculation with degenerate UVs."""
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=np.float32)
        
        # Degenerate UVs (all the same)
        uvs = np.array([
            [0, 0],
            [0, 0],
            [0, 0]
        ], dtype=np.float32)
        
        triangles = np.array([[0, 1, 2]], dtype=np.uint32)
        
        # Should not crash, though results may be zero
        tangents = MeshBuilder.calculate_tangent(positions, uvs, triangles)
        
        self.assertEqual(tangents.shape, (3, 4))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and unusual scenarios."""
    
    def test_empty_submesh(self):
        """Test mesh with empty submesh."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add vertices but no triangles
        builder.add_vertex((0, 0, 0))
        builder.add_vertex((1, 0, 0))
        builder.add_vertex((0, 1, 0))
        
        # Empty submesh is valid (no indices to check)
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_many_submeshes(self):
        """Test mesh with many submeshes."""
        builder = MeshBuilder()
        
        # Create 10 submeshes
        for _ in range(10):
            builder.add_submesh()
        
        self.assertEqual(builder.submesh_count(), 10)
        
        # Add vertices
        for i in range(4):
            builder.add_vertex((i, 0, 0))
        
        # Add one triangle to each submesh
        for i in range(10):
            builder.add_triangle(i, 0, 1, 2)
        
        error = builder.check()
        self.assertEqual(error, "")
        
        self.assertEqual(builder.indices_count(), 30)  # 10 * 3
    
    def test_many_uv_sets(self):
        """Test mesh with multiple UV sets."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add 4 UV sets
        for _ in range(4):
            builder.add_uv_set()
        
        self.assertEqual(builder.uv_count(), 4)
    
    def test_large_vertex_count(self):
        """Test mesh with large number of vertices."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add 1000 vertices
        for i in range(1000):
            builder.add_vertex((i, i * 0.5, i * 0.25))
        
        self.assertEqual(builder.vertex_count(), 1000)
        
        # Add some triangles
        for i in range(0, 997, 3):
            builder.add_triangle(0, i, i + 1, i + 2)
        
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_negative_coordinates(self):
        """Test mesh with negative coordinates."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((-1, -2, -3))
        v1 = builder.add_vertex((-4, -5, -6))
        v2 = builder.add_vertex((-7, -8, -9))
        
        builder.add_triangle(0, v0, v1, v2)
        
        error = builder.check()
        self.assertEqual(error, "")
        
        np.testing.assert_array_almost_equal(
            builder.position[v0], [-1, -2, -3]
        )
    
    def test_floating_point_precision(self):
        """Test mesh with floating point precision values."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0.1, 0.2, 0.3))
        v1 = builder.add_vertex((1.234567, 2.345678, 3.456789))
        v2 = builder.add_vertex((1e-6, 1e-7, 1e-8))
        
        builder.add_triangle(0, v0, v1, v2)
        
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_reused_vertices(self):
        """Test mesh with vertices reused across triangles."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Create a fan from one vertex
        center = builder.add_vertex((0, 0, 0))
        
        # Create 8 triangles around center
        vertices = [center]
        for i in range(8):
            angle = i * (2 * np.pi / 8)
            x = np.cos(angle)
            y = np.sin(angle)
            vertices.append(builder.add_vertex((x, y, 0)))
        
        # Create triangles: center, i, i+1
        for i in range(1, 8):
            builder.add_triangle(0, center, vertices[i], vertices[i + 1] if i < 8 else vertices[1])
        
        error = builder.check()
        self.assertEqual(error, "")
        
        # Center vertex used in 7 triangles
        self.assertEqual(builder.vertex_count(), 9)
        self.assertEqual(builder.indices_count(), 21)  # 7 * 3


class TestMeshWithNormalsAndTangents(unittest.TestCase):
    """Test mesh with normals and tangents."""
    
    def test_add_normals_manually(self):
        """Test setting normals manually."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add 3 vertices
        for _ in range(3):
            builder.add_vertex((0, 0, 0))
        
        # Add normals
        builder.normal = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.assertTrue(builder.contained_normal())
        self.assertEqual(builder.normal.shape, (3, 3))
    
    def test_add_tangents_manually(self):
        """Test setting tangents manually."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add 3 vertices
        for _ in range(3):
            builder.add_vertex((0, 0, 0))
        
        # Add tangents
        builder.tangent = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, -1]
        ], dtype=np.float32)
        
        self.assertTrue(builder.contained_tangent())
        self.assertEqual(builder.tangent.shape, (3, 4))
    
    def test_complete_mesh_with_all_attributes(self):
        """Test complete mesh with positions, normals, tangents, and UVs."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        # Create a simple quad (2 triangles)
        positions = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ]
        
        normals = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ]
        
        uvs = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ]
        
        for pos in positions:
            builder.add_vertex(pos)
        
        builder.normal = np.array(normals, dtype=np.float32)
        builder.uvs[0] = np.array(uvs, dtype=np.float32)
        
        # Calculate tangents
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        builder.tangent = MeshBuilder.calculate_tangent(
            builder.position, builder.uvs[0], triangles
        )
        
        # Add triangles
        builder.add_triangle(0, 0, 1, 2)
        builder.add_triangle(0, 0, 2, 3)
        
        # Validate
        error = builder.check()
        self.assertEqual(error, "")
        
        # Check all attributes present
        self.assertTrue(builder.contained_normal())
        self.assertTrue(builder.contained_tangent())
        self.assertEqual(builder.uv_count(), 1)


class TestSubmeshOffsets(unittest.TestCase):
    """Test submesh offset calculation."""
    
    def test_single_submesh_empty_offsets(self):
        """Test single submesh returns empty offsets array."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        builder.add_triangle(0, v0, v1, v2)
        
        offsets = builder._gen_submesh_offsets()
        self.assertEqual(len(offsets), 0)
    
    def test_multiple_submesh_offsets(self):
        """Test multiple submesh offset calculation."""
        builder = MeshBuilder()
        
        # Add 3 submeshes with different triangle counts
        for _ in range(3):
            builder.add_submesh()
        
        # Add vertices
        for i in range(6):
            builder.add_vertex((i, 0, 0))
        
        # Submesh 0: 2 triangles
        builder.add_triangle(0, 0, 1, 2)
        builder.add_triangle(0, 0, 2, 3)
        
        # Submesh 1: 1 triangle
        builder.add_triangle(1, 0, 1, 4)
        
        # Submesh 2: 3 triangles
        builder.add_triangle(2, 0, 4, 5)
        builder.add_triangle(2, 0, 5, 1)
        builder.add_triangle(2, 1, 2, 3)
        
        offsets = builder._gen_submesh_offsets()
        
        self.assertEqual(len(offsets), 3)
        self.assertEqual(offsets[0], 0)  # First submesh starts at 0
        self.assertEqual(offsets[1], 2)  # Second submesh starts at triangle 2
        self.assertEqual(offsets[2], 3)  # Third submesh starts at triangle 3


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
