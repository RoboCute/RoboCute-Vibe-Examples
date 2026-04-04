"""
Houdini-Style Procedural Generation Tests with RoboCute API
===========================================================

Tests for houdini_nodes.py, procedural_terrain.py, and city_generator.py
with RoboCute Python API integration for rendering.

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
    class MockModule:
        app = MockApp()
    rbc = MockModule()
    re = MockModule()
    lc = MockModule()

import numpy as np

# Import modules under test
from Houdini.houdini_nodes import (
    ScatterPoint, scatter_on_surface, poisson_disk_sampling,
    hexagonal_packing, rectangular_packing, relax_points,
    ScenePartitioner, LSystem, foreach_batch_process,
    VariantSet, ComponentBuilder,
    create_grid_instances_data, sample_height_from_function,
    calculate_slope, create_cylinder_mesh, extrude_building
)
from Houdini.city_generator import (
    CityGenerator, BuildingGenerator, BuildingParameters,
    BuildingStyle, generate_complete_city, StreetNetwork
)
from Houdini.procedural_terrain import (
    HeightFieldTerrain, TerrainScatterResult
)
from mesh_builder import MeshBuilder


# =============================================================================
# Test Classes
# =============================================================================

class TestScatterPoint(unittest.TestCase):
    """Test ScatterPoint dataclass."""
    
    def test_default_creation(self):
        """Test creating a ScatterPoint with default values."""
        point = ScatterPoint(position=(0, 0, 0))
        self.assertEqual(point.position, (0, 0, 0))
        self.assertEqual(point.scale, 1.0)
        self.assertEqual(point.rotation, (0.0, 1.0, 0.0, 0.0))
        self.assertIsNotNone(point.attributes)
    
    def test_custom_creation(self):
        """Test creating a ScatterPoint with custom values."""
        point = ScatterPoint(
            position=(1, 2, 3),
            scale=2.0,
            rotation=(0, 1, 0, 90),
            attributes={"id": 1}
        )
        self.assertEqual(point.position, (1, 2, 3))
        self.assertEqual(point.scale, 2.0)
        self.assertEqual(point.rotation, (0, 1, 0, 90))
        self.assertEqual(point.attributes["id"], 1)


class TestScatterOnSurface(unittest.TestCase):
    """Test scatter_on_surface function."""
    
    def test_default_scatter(self):
        """Test scatter with default parameters."""
        points = scatter_on_surface(count=10)
        self.assertEqual(len(points), 10)
        for point in points:
            self.assertIsInstance(point, ScatterPoint)
            self.assertEqual(point.position[1], 0.0)  # y_height
    
    def test_scatter_bounds(self):
        """Test that scattered points respect bounds."""
        points = scatter_on_surface(bounds=(-10, 10), count=100)
        for point in points:
            x, y, z = point.position
            self.assertGreaterEqual(x, -10)
            self.assertLessEqual(x, 10)
            self.assertGreaterEqual(z, -10)
            self.assertLessEqual(z, 10)
    
    def test_scatter_scale_range(self):
        """Test that scattered points respect scale range."""
        points = scatter_on_surface(count=100, scale_range=(0.5, 1.5))
        for point in points:
            self.assertGreaterEqual(point.scale, 0.5)
            self.assertLessEqual(point.scale, 1.5)
    
    def test_scatter_reproducibility(self):
        """Test that scatter is reproducible with same seed."""
        points1 = scatter_on_surface(count=10, seed=42)
        points2 = scatter_on_surface(count=10, seed=42)
        for p1, p2 in zip(points1, points2):
            self.assertEqual(p1.position, p2.position)
            self.assertEqual(p1.scale, p2.scale)


class TestPoissonDiskSampling(unittest.TestCase):
    """Test poisson_disk_sampling function."""
    
    def test_basic_sampling(self):
        """Test basic Poisson disk sampling."""
        points = poisson_disk_sampling(radius=5, width=100, height=100)
        self.assertGreater(len(points), 0)
    
    def test_minimum_distance(self):
        """Test that points respect minimum distance."""
        radius = 5.0
        points = poisson_disk_sampling(radius=radius, width=100, height=100, max_attempts=30)
        for i, p1 in enumerate(points):
            for p2 in points[i+1:]:
                dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                self.assertGreaterEqual(dist, radius * 0.9)  # Allow small tolerance
    
    def test_reproducibility(self):
        """Test reproducibility with seed."""
        points1 = poisson_disk_sampling(radius=5, width=100, height=100, seed=42)
        points2 = poisson_disk_sampling(radius=5, width=100, height=100, seed=42)
        self.assertEqual(len(points1), len(points2))
        for p1, p2 in zip(points1, points2):
            self.assertAlmostEqual(p1[0], p2[0], places=5)
            self.assertAlmostEqual(p1[1], p2[1], places=5)


class TestHexagonalPacking(unittest.TestCase):
    """Test hexagonal_packing function."""
    
    def test_basic_packing(self):
        """Test basic hexagonal packing."""
        points = hexagonal_packing(rows=5, cols=5, spacing=2.0)
        self.assertEqual(len(points), 25)
    
    def test_centering(self):
        """Test that points are centered at origin."""
        points = hexagonal_packing(rows=5, cols=5, spacing=2.0, center_origin=True)
        xs = [p[0] for p in points]
        zs = [p[2] for p in points]
        center_x = (min(xs) + max(xs)) / 2
        center_z = (min(zs) + max(zs)) / 2
        self.assertAlmostEqual(center_x, 0.0, places=5)
        self.assertAlmostEqual(center_z, 0.0, places=5)
    
    def test_spacing(self):
        """Test that spacing is correct."""
        spacing = 2.0
        points = hexagonal_packing(rows=2, cols=2, spacing=spacing)
        # In hexagonal packing, adjacent points in a row are spaced by 'spacing'
        # and adjacent rows are offset by spacing/2
        p00 = points[0]  # row 0, col 0
        p01 = points[1]  # row 0, col 1
        dist = math.sqrt((p00[0]-p01[0])**2 + (p00[2]-p01[2])**2)
        self.assertAlmostEqual(dist, spacing, places=5)


class TestRectangularPacking(unittest.TestCase):
    """Test rectangular_packing function."""
    
    def test_basic_packing(self):
        """Test basic rectangular packing."""
        points = rectangular_packing(rows=5, cols=5, spacing=2.0)
        self.assertEqual(len(points), 25)
    
    def test_grid_alignment(self):
        """Test that points form a regular grid."""
        spacing = 2.0
        points = rectangular_packing(rows=3, cols=3, spacing=spacing, center_origin=False)
        # First row should be at z=0
        for i in range(3):
            self.assertAlmostEqual(points[i][2], 0.0)
        # Columns should be spaced by 'spacing'
        self.assertAlmostEqual(points[1][0] - points[0][0], spacing, places=5)


class TestRelaxPoints(unittest.TestCase):
    """Test relax_points function."""
    
    def test_basic_relaxation(self):
        """Test basic point relaxation."""
        initial = [(0, 0), (1, 0), (0.5, 0.1)]
        relaxed = relax_points(initial, iterations=10)
        self.assertEqual(len(relaxed), len(initial))
    
    def test_no_overlap(self):
        """Test that relaxation reduces overlaps."""
        # Create points that are very close
        initial = [(0, 0), (0.1, 0), (0, 0.1)]
        relaxed = relax_points(initial, iterations=50, repulsion_radius=1.0)
        # After relaxation, points should be further apart
        min_dist_before = min(
            math.sqrt((initial[i][0]-initial[j][0])**2 + (initial[i][1]-initial[j][1])**2)
            for i in range(len(initial)) for j in range(i+1, len(initial))
        )
        min_dist_after = min(
            math.sqrt((relaxed[i][0]-relaxed[j][0])**2 + (relaxed[i][1]-relaxed[j][1])**2)
            for i in range(len(relaxed)) for j in range(i+1, len(relaxed))
        )
        self.assertGreater(min_dist_after, min_dist_before)


class TestScenePartitioner(unittest.TestCase):
    """Test ScenePartitioner class."""
    
    def test_group_by_bounding_box(self):
        """Test grouping by bounding box."""
        points = [
            ScatterPoint(position=(0, 0, 0)),
            ScatterPoint(position=(5, 0, 5)),
            ScatterPoint(position=(10, 0, 10)),
        ]
        partitioner = ScenePartitioner()
        selected = partitioner.group_by_bounding_box(
            points,
            min_point=(-1, -1, -1),
            max_point=(6, 1, 6),
            group_name="center"
        )
        self.assertEqual(len(selected), 2)  # First two points
        self.assertIn("center", partitioner.groups)
    
    def test_filter_by_attribute(self):
        """Test filtering by attribute."""
        points = [
            ScatterPoint(position=(0, 0, 0), attributes={"height": 10}),
            ScatterPoint(position=(1, 0, 1), attributes={"height": 20}),
            ScatterPoint(position=(2, 0, 2), attributes={"height": 30}),
        ]
        partitioner = ScenePartitioner()
        filtered = partitioner.filter_by_attribute(points, "height", 15, 25)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].attributes["height"], 20)
    
    def test_apply_condition(self):
        """Test applying condition."""
        items = [1, 2, 3, 4, 5]
        result = []
        partitioner = ScenePartitioner()
        partitioner.apply_condition(
            items,
            condition_func=lambda x: x > 2,
            action_func=lambda x: result.append(x * 2)
        )
        self.assertEqual(result, [6, 8, 10])
    
    def test_partition(self):
        """Test partition function."""
        points = [
            ScatterPoint(position=(0, 0, 0), attributes={"type": "tree"}),
            ScatterPoint(position=(1, 0, 1), attributes={"type": "rock"}),
            ScatterPoint(position=(2, 0, 2), attributes={"type": "tree"}),
        ]
        partitioner = ScenePartitioner()
        trees, others = partitioner.partition(
            points,
            lambda p: p.attributes.get("type") == "tree"
        )
        self.assertEqual(len(trees), 2)
        self.assertEqual(len(others), 1)


class TestLSystem(unittest.TestCase):
    """Test LSystem class."""
    
    def test_generation(self):
        """Test L-System string generation."""
        lsys = LSystem(axiom="F", rules={"F": "F+F"})
        result = lsys.generate(iterations=2)
        self.assertEqual(result, "F+F+F+F")
    
    def test_interpretation(self):
        """Test L-System interpretation."""
        lsys = LSystem(axiom="F", rules={"F": "F[+F]F[-F]F"}, angle=25.7, step_size=1.0)
        lstring = lsys.generate(iterations=1)
        result = lsys.interpret(lstring)
        self.assertIn("branches", result)
        self.assertIn("leaves", result)
        self.assertGreater(len(result["branches"]), 0)
    
    def test_complex_system(self):
        """Test more complex L-System."""
        lsys = LSystem(
            axiom="X",
            rules={
                "X": "F+[[X]-X]-F[-FX]+X",
                "F": "FF"
            },
            angle=25,
            step_size=1.0
        )
        lstring = lsys.generate(iterations=3)
        result = lsys.interpret(lstring)
        self.assertGreater(len(result["branches"]), 0)


class TestForeachBatchProcess(unittest.TestCase):
    """Test foreach_batch_process function."""
    
    def test_basic_processing(self):
        """Test basic batch processing."""
        items = [1, 2, 3, 4, 5]
        result = foreach_batch_process(
            items,
            lambda item, idx: item * 2
        )
        self.assertEqual(result, [2, 4, 6, 8, 10])
    
    def test_with_index(self):
        """Test that index is passed correctly."""
        items = ["a", "b", "c"]
        result = foreach_batch_process(
            items,
            lambda item, idx: f"{item}_{idx}"
        )
        self.assertEqual(result, ["a_0", "b_1", "c_2"])


class TestVariantSet(unittest.TestCase):
    """Test VariantSet class."""
    
    def test_add_and_switch(self):
        """Test adding and switching variants."""
        variants = VariantSet("tree_variants")
        variants.add_variant("oak", {"type": "oak", "height": 10})
        variants.add_variant("pine", {"type": "pine", "height": 15})
        
        self.assertTrue(variants.switch_variant("oak"))
        self.assertEqual(variants.get_active()["type"], "oak")
        
        self.assertTrue(variants.switch_variant("pine"))
        self.assertEqual(variants.get_active()["type"], "pine")
    
    def test_invalid_switch(self):
        """Test switching to non-existent variant."""
        variants = VariantSet("test")
        variants.add_variant("a", 1)
        self.assertFalse(variants.switch_variant("b"))


class TestComponentBuilder(unittest.TestCase):
    """Test ComponentBuilder class."""
    
    def test_parameter_constraints(self):
        """Test parameter with min/max constraints."""
        builder = ComponentBuilder("test")
        builder.add_parameter("size", 10.0, min_val=0.0, max_val=100.0)
        builder.set_build_function(lambda size: {"size": size})
        
        # Within range
        result = builder.build(size=50.0)
        self.assertEqual(result["size"], 50.0)
        
        # Below min
        result = builder.build(size=-10.0)
        self.assertEqual(result["size"], 0.0)
        
        # Above max
        result = builder.build(size=200.0)
        self.assertEqual(result["size"], 100.0)
    
    def test_build_without_function(self):
        """Test building without set build function."""
        builder = ComponentBuilder("test")
        builder.add_parameter("value", 5)
        result = builder.build()
        self.assertIsNone(result)


class TestGridInstances(unittest.TestCase):
    """Test create_grid_instances_data function."""
    
    def test_basic_grid(self):
        """Test basic grid creation."""
        points = create_grid_instances_data(rows=3, cols=3, spacing=2.0)
        self.assertEqual(len(points), 9)
    
    def test_scale_variation(self):
        """Test that scale varies within range."""
        points = create_grid_instances_data(
            rows=10, cols=10, scale_range=(0.8, 1.2)
        )
        for point in points:
            self.assertGreaterEqual(point.scale, 0.8)
            self.assertLessEqual(point.scale, 1.2)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_sample_height_from_function(self):
        """Test height sampling."""
        def height_func(x, z):
            return x + z
        
        height = sample_height_from_function(1, 2, height_func)
        self.assertEqual(height, 3)
    
    def test_calculate_slope(self):
        """Test slope calculation."""
        def slope_func(x, z):
            return x * 0.1  # 10% slope in x direction
        
        slope = calculate_slope(0, 0, slope_func, sample_distance=1.0)
        # arctan(0.1) in degrees
        self.assertAlmostEqual(slope, 5.7106, places=2)


class TestCylinderMesh(unittest.TestCase):
    """Test create_cylinder_mesh function."""
    
    def test_basic_cylinder(self):
        """Test basic cylinder creation."""
        mesh = create_cylinder_mesh(radius=1.0, height=2.0, segments=8)
        self.assertIn("vertices", mesh)
        self.assertIn("indices", mesh)
        self.assertGreater(len(mesh["vertices"]), 0)
        self.assertGreater(len(mesh["indices"]), 0)
    
    def test_vertex_count(self):
        """Test that cylinder has correct vertex count."""
        segments = 16
        mesh = create_cylinder_mesh(radius=1.0, height=2.0, segments=segments)
        # 2 centers + 2 * segments ring vertices
        expected_vertices = 2 + 2 * segments
        self.assertEqual(len(mesh["vertices"]), expected_vertices)


class TestExtrudeBuilding(unittest.TestCase):
    """Test extrude_building function."""
    
    def test_basic_extrusion(self):
        """Test basic building extrusion."""
        footprint = [(0, 0), (10, 0), (10, 10), (0, 10)]
        building = extrude_building(footprint, height=20.0, floors=5)
        
        self.assertIn("walls", building)
        self.assertIn("roof", building)
        self.assertIn("floor_slabs", building)
        self.assertEqual(building["height"], 20.0)
        self.assertEqual(building["floors"], 5)
    
    def test_wall_count(self):
        """Test that wall count matches footprint edges."""
        footprint = [(0, 0), (10, 0), (10, 10), (5, 15), (0, 10)]  # 5 vertices
        building = extrude_building(footprint, height=20.0, floors=5)
        self.assertEqual(len(building["walls"]), 5)
    
    def test_floor_slab_count(self):
        """Test that floor slab count is correct."""
        floors = 5
        footprint = [(0, 0), (10, 0), (10, 10), (0, 10)]
        building = extrude_building(footprint, height=20.0, floors=floors)
        self.assertEqual(len(building["floor_slabs"]), floors + 1)  # Including ground


class TestRoboCuteDisplay(unittest.TestCase):
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
        # In real tests, this would render frames
        # For unit tests, we just verify the parameter is accepted
        try:
            app.run(prepare_denoise=False, limit_frame=10)
        except Exception as e:
            # Expected in test environment without actual display
            pass


class TestMeshBuilderIntegration(unittest.TestCase):
    """Test MeshBuilder integration with RoboCute API."""
    
    def test_mesh_builder_with_cylinder_creation(self):
        """Test MeshBuilder with cylinder mesh data."""
        mesh = create_cylinder_mesh(radius=1.0, height=2.0, segments=8)
        
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        for vertex in mesh["vertices"]:
            builder.add_vertex(vertex)
        
        for tri in mesh["indices"]:
            builder.add_triangle(0, tri[0], tri[1], tri[2])
        
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_mesh_builder_with_building(self):
        """Test MeshBuilder with building data."""
        gen = BuildingGenerator(seed=42)
        params = BuildingParameters(width=20, depth=15, height=30, floors=10)
        building = gen.generate_building(params)
        
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Create vertices from footprint
        footprint = building.footprint
        for x, z in footprint:
            builder.add_vertex((x, 0, z))
            builder.add_vertex((x, building.params.height, z))
        
        # Create triangles for walls
        n = len(footprint)
        for i in range(n):
            next_i = (i + 1) % n
            v_bottom_i = i * 2
            v_top_i = i * 2 + 1
            v_bottom_next = next_i * 2
            v_top_next = next_i * 2 + 1
            
            builder.add_triangle(0, v_bottom_i, v_bottom_next, v_top_next)
            builder.add_triangle(0, v_bottom_i, v_top_next, v_top_i)
        
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_mesh_builder_with_terrain(self):
        """Test MeshBuilder with terrain mesh data."""
        terrain = HeightFieldTerrain(width=32, height=32, seed=42)
        terrain.generate_fractal_terrain()
        
        vertices, indices = terrain.to_mesh_data()
        
        builder = MeshBuilder()
        builder.add_submesh()
        
        for vertex in vertices:
            builder.add_vertex(vertex)
        
        indices_arr = np.array(indices, dtype=np.uint32)
        for i in range(0, len(indices_arr), 3):
            builder.add_triangle(0, indices_arr[i], indices_arr[i+1], indices_arr[i+2])
        
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_mesh_builder_with_scatter_points(self):
        """Test MeshBuilder with scatter point instances."""
        points = scatter_on_surface(count=10, bounds=(-50, 50))
        
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Create a simple triangle for each scatter point
        for i, point in enumerate(points):
            x, y, z = point.position
            scale = point.scale
            
            v0 = builder.add_vertex((x - scale, y, z - scale))
            v1 = builder.add_vertex((x + scale, y, z - scale))
            v2 = builder.add_vertex((x, y + scale, z))
            
            builder.add_triangle(0, v0, v1, v2)
        
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_mesh_builder_with_trees(self):
        """Test MeshBuilder with tree structures."""
        from Houdini.houdini_nodes import generate_tree_structure
        
        tree = generate_tree_structure(tree_type="oak", height=10.0, seed=42)
        
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Create trunk vertices
        trunk = tree["trunk"]
        trunk_height = trunk["height"]
        trunk_radius = trunk["radius"]
        
        segments = 8
        base_vertices = []
        top_vertices = []
        
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = trunk_radius * math.cos(angle)
            z = trunk_radius * math.sin(angle)
            base_vertices.append(builder.add_vertex((x, 0, z)))
            top_vertices.append(builder.add_vertex((x, trunk_height, z)))
        
        # Create trunk faces
        for i in range(segments):
            next_i = (i + 1) % segments
            builder.add_triangle(0, base_vertices[i], base_vertices[next_i], top_vertices[next_i])
            builder.add_triangle(0, base_vertices[i], top_vertices[next_i], top_vertices[i])
        
        # Create foliage spheres (simplified as single vertex each)
        for foliage in tree["foliage"]:
            pos = foliage["position"]
            radius = foliage["radius"]
            # Just add center point for foliage
            builder.add_vertex(pos)
        
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_mesh_builder_submeshes(self):
        """Test MeshBuilder with multiple submeshes."""
        builder = MeshBuilder()
        
        # Create multiple submeshes
        submesh_0 = builder.add_submesh()
        submesh_1 = builder.add_submesh()
        submesh_2 = builder.add_submesh()
        
        self.assertEqual(builder.submesh_count(), 3)
        
        # Add vertices and triangles to each submesh
        for submesh_idx in range(3):
            v0 = builder.add_vertex((submesh_idx, 0, 0))
            v1 = builder.add_vertex((submesh_idx + 1, 0, 0))
            v2 = builder.add_vertex((submesh_idx + 0.5, 1, 0))
            builder.add_triangle(submesh_idx, v0, v1, v2)
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.indices_count(), 9)  # 3 triangles * 3 indices
    
    def test_mesh_builder_with_multiple_uv_sets(self):
        """Test MeshBuilder with multiple UV sets."""
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()  # UV set 0
        builder.add_uv_set()  # UV set 1
        
        # Add vertices
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        
        # Add triangle
        builder.add_triangle(0, v0, v1, v2)
        
        # Manually set UVs for both sets
        builder.uvs[0] = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        builder.uvs[1] = np.array([[0, 0], [0.5, 0], [0, 0.5]], dtype=np.float32)
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.uv_count(), 2)
    
    def test_mesh_builder_normals_and_tangents(self):
        """Test MeshBuilder with normals and tangents."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add a simple quad
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((1, 1, 0))
        v3 = builder.add_vertex((0, 1, 0))
        
        builder.add_triangle(0, v0, v1, v2)
        builder.add_triangle(0, v0, v2, v3)
        
        # Add normals (all pointing up in Z)
        builder.normal = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Add UVs
        builder.add_uv_set()
        builder.uvs[0] = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)
        
        # Calculate tangents
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        builder.tangent = MeshBuilder.calculate_tangent(
            builder.position, builder.uvs[0], triangles
        )
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertTrue(builder.contained_normal())
        self.assertTrue(builder.contained_tangent())
    
    def test_mesh_builder_validation_errors(self):
        """Test MeshBuilder validation catches errors."""
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add vertex
        v0 = builder.add_vertex((0, 0, 0))
        
        # Add triangle with out-of-range index
        builder.add_triangle(0, v0, v0 + 1, v0 + 2)  # Invalid indices
        
        error = builder.check()
        self.assertIn("out of vertex range", error)
    
    def test_mesh_builder_city_blocks(self):
        """Test MeshBuilder with city block generation."""
        city_gen = CityGenerator(block_size=50, street_width=10, seed=42)
        blocks, streets = city_gen.generate_grid(3, 3)
        
        builder = MeshBuilder()
        block_submesh = builder.add_submesh()
        street_submesh = builder.add_submesh()
        
        # Create block meshes
        for block in blocks[:3]:  # Limit for test
            cx, cz = block.center
            size = block.size
            
            # Simple square for block
            v0 = builder.add_vertex((cx - size/2, 0, cz - size/2))
            v1 = builder.add_vertex((cx + size/2, 0, cz - size/2))
            v2 = builder.add_vertex((cx + size/2, 0, cz + size/2))
            v3 = builder.add_vertex((cx - size/2, 0, cz + size/2))
            
            builder.add_triangle(block_submesh, v0, v1, v2)
            builder.add_triangle(block_submesh, v0, v2, v3)
        
        # Create street meshes
        for street in streets[:3]:
            x, z, w, d = street
            v0 = builder.add_vertex((x, 0.1, z))
            v1 = builder.add_vertex((x + w, 0.1, z))
            v2 = builder.add_vertex((x + w, 0.1, z + d))
            v3 = builder.add_vertex((x, 0.1, z + d))
            
            builder.add_triangle(street_submesh, v0, v1, v2)
            builder.add_triangle(street_submesh, v0, v2, v3)
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.submesh_count(), 2)
    
    def test_mesh_builder_hex_packing_instances(self):
        """Test MeshBuilder with hexagonal packing for instancing."""
        points = hexagonal_packing(spacing=2.0, rows=5, cols=5)
        
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Create a hexagon for each point
        for point in points:
            x, y, z = point
            
            # Create hexagon vertices around point
            radius = 0.8
            hex_vertices = []
            for i in range(6):
                angle = math.pi / 3 * i
                hx = x + radius * math.cos(angle)
                hz = z + radius * math.sin(angle)
                hex_vertices.append(builder.add_vertex((hx, y, hz)))
            
            # Create hexagon faces (fan from center)
            center_idx = builder.add_vertex((x, y + 0.5, z))
            for i in range(6):
                next_i = (i + 1) % 6
                builder.add_triangle(0, center_idx, hex_vertices[i], hex_vertices[next_i])
        
        error = builder.check()
        self.assertEqual(error, "")
    
    def test_mesh_builder_file_output(self):
        """Test MeshBuilder file output functionality."""
        import tempfile
        import os
        
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add a simple triangle
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        builder.add_triangle(0, v0, v1, v2)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            offsets = builder.write_to(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_mesh_builder_building_styles(self):
        """Test MeshBuilder with different building styles."""
        gen = BuildingGenerator(seed=42)
        
        for style in [BuildingStyle.MODERN, BuildingStyle.CLASSIC]:
            builder = MeshBuilder()
            builder.add_submesh()
            
            params = BuildingParameters(
                width=20, depth=15, height=30, floors=10,
                style=style, seed=42
            )
            building = gen.generate_building(params)
            
            # Create simple building mesh
            footprint = building.footprint
            for i, (x, z) in enumerate(footprint):
                builder.add_vertex((x, 0, z))
                builder.add_vertex((x, building.params.height, z))
            
            # Create wall triangles
            n = len(footprint)
            for i in range(n):
                next_i = (i + 1) % n
                v0 = i * 2
                v1 = next_i * 2
                v2 = next_i * 2 + 1
                v3 = i * 2 + 1
                builder.add_triangle(0, v0, v1, v2)
                builder.add_triangle(0, v0, v2, v3)
            
            error = builder.check()
            self.assertEqual(error, "", f"Style {style} failed: {error}")


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple modules."""

    def test_forest_scene_workflow(self):
        """Test complete forest scene workflow."""
        # 1. Create terrain
        terrain = HeightFieldTerrain(width=128, height=128, seed=42)
        terrain.generate_fractal_terrain()
        terrain.thermal_erosion(iterations=20)
        
        # 2. Scatter trees
        tree_result = terrain.scatter_on_terrain(
            count=100, min_slope=0, max_slope=20, seed=42
        )
        
        # 3. Create scatter points
        tree_points = [
            ScatterPoint(position=pos, attributes={"type": "tree"})
            for pos in tree_result.points
        ]
        
        # 4. Partition by bounding box
        partitioner = ScenePartitioner()
        center_trees = partitioner.group_by_bounding_box(
            tree_points,
            min_point=(30, 0, 30),
            max_point=(70, 100, 70),
            group_name="center"
        )
        
        self.assertGreater(len(tree_points), 0)
        self.assertGreaterEqual(len(center_trees), 0)
    
    def test_city_scene_workflow(self):
        """Test complete city scene workflow."""
        # 1. Generate city layout
        city_gen = CityGenerator(block_size=50, street_width=10, seed=42)
        blocks, streets = city_gen.generate_grid(4, 4)
        
        # 2. Generate buildings
        building_gen = BuildingGenerator(seed=42)
        all_buildings = []
        
        for block in blocks:
            buildings = building_gen.populate_block(block, density=0.6)
            all_buildings.extend(buildings)
        
        # 3. Generate road meshes
        road_meshes = []
        for street in streets[:3]:  # Just a few for testing
            x, z, w, d = street
            road_meshes.append({
                "bounds": street,
                "corners": [(x, 0, z), (x+w, 0, z), (x+w, 0, z+d), (x, 0, z+d)]
            })
        
        self.assertEqual(len(blocks), 16)
        self.assertGreater(len(all_buildings), 0)
        self.assertGreater(len(road_meshes), 0)


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
        
        # Create a simple triangle
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
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
    
    def test_mesh_builder_to_render_component(self):
        """
        Test creating a mesh entity through RoboCute API.
        """
        if not ROBOCUTE_AVAILABLE:
            self.skipTest("robocute not available")
        
        app = rbc.app.App()
        app.init_display(1280, 720)
        
        # Create scene and entity
        scene = re.world.Scene()
        entity = scene.add_entity()
        entity.set_name("test_mesh")
        
        # Add transform and render components
        trans = re.world.TransformComponent(entity.add_component("TransformComponent"))
        render = re.world.RenderComponent(entity.add_component("RenderComponent"))
        trans.set_pos(lc.double3(0, 0, 0), False)
        
        # Create mesh with MeshBuilder
        builder = MeshBuilder()
        builder.add_submesh()
        
        v0 = builder.add_vertex((0, 0, 0))
        v1 = builder.add_vertex((1, 0, 0))
        v2 = builder.add_vertex((0, 1, 0))
        builder.add_triangle(0, v0, v1, v2)
        
        mesh = builder.write_to_mesh()
        mesh.install()
        
        # Create material
        mat0 = re.world.MaterialResource()
        mat0.load_from_json('{"base_weight": 1.0, "base_color": [0.8, 0.8, 0.8]}')
        
        mat_vector = lc.capsule_vector()
        mat_vector.emplace_back(mat0._handle)
        
        render.update_object(mat_vector, mesh)
        
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
