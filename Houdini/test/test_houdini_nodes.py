"""
Comprehensive test suite for Houdini-style procedural generation system.

This test file covers:
- houdini_nodes.py: Scattering, partitioning, L-systems, components
- procedural_terrain.py: HeightField terrain, erosion, terrain scatter
- city_generator.py: City layouts, building generation

Usage:
    cd samples/Houdini
    python -m pytest test/test_houdini_nodes.py -v
    python test/test_houdini_nodes.py  # Run directly
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from houdini_nodes import (
    ScatterPoint,
    scatter_on_surface,
    poisson_disk_sampling,
    hexagonal_packing,
    rectangular_packing,
    relax_points,
    ScenePartitioner,
    LSystem,
    foreach_batch_process,
    VariantSet,
    ComponentBuilder,
    generate_city_block_points,
    generate_tree_structure,
    sample_height_from_function,
    calculate_slope,
    extrude_building,
    create_cylinder_mesh,
    create_grid_instances_data,
)
from procedural_terrain import HeightFieldTerrain, TerrainScatterResult
from city_generator import (
    CityGenerator,
    BuildingGenerator,
    BuildingParameters,
    BuildingStyle,
    BuildingData,
    CityBlock,
    StreetNetwork,
    generate_complete_city,
)
# Add project root for mesh_builder import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from mesh_builder import MeshBuilder


class TestScatterPoint(unittest.TestCase):
    """Test ScatterPoint dataclass."""
    
    def test_default_creation(self):
        """Test creating ScatterPoint with defaults."""
        point = ScatterPoint(position=(0, 0, 0))
        self.assertEqual(point.position, (0, 0, 0))
        self.assertEqual(point.scale, 1.0)
        self.assertEqual(point.rotation, (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(point.attributes, {})
    
    def test_custom_creation(self):
        """Test creating ScatterPoint with custom values."""
        point = ScatterPoint(
            position=(1, 2, 3),
            scale=2.0,
            rotation=(0, 1, 0, 45),
            attributes={"type": "tree", "id": 1}
        )
        self.assertEqual(point.position, (1, 2, 3))
        self.assertEqual(point.scale, 2.0)
        self.assertEqual(point.rotation, (0, 1, 0, 45))
        self.assertEqual(point.attributes["type"], "tree")


class TestScatterOnSurface(unittest.TestCase):
    """Test scatter_on_surface function."""
    
    def test_basic_scatter(self):
        """Test basic scatter functionality."""
        points = scatter_on_surface(bounds=(-10, 10), count=50, seed=42)
        self.assertEqual(len(points), 50)
        
        # Check all points are within bounds
        for p in points:
            self.assertGreaterEqual(p.position[0], -10)
            self.assertLessEqual(p.position[0], 10)
            self.assertGreaterEqual(p.position[2], -10)
            self.assertLessEqual(p.position[2], 10)
    
    def test_scatter_reproducibility(self):
        """Test scatter with same seed produces same results."""
        points1 = scatter_on_surface(bounds=(-10, 10), count=10, seed=123)
        points2 = scatter_on_surface(bounds=(-10, 10), count=10, seed=123)
        
        for p1, p2 in zip(points1, points2):
            self.assertEqual(p1.position, p2.position)
            self.assertEqual(p1.scale, p2.scale)
    
    def test_scatter_attributes(self):
        """Test scatter point attributes."""
        points = scatter_on_surface(bounds=(-10, 10), count=5, seed=42)
        
        for i, p in enumerate(points):
            self.assertIn("id", p.attributes)
            self.assertIn("pscale", p.attributes)
            self.assertEqual(p.attributes["id"], i)


class TestPoissonDiskSampling(unittest.TestCase):
    """Test poisson_disk_sampling function."""
    
    def test_basic_sampling(self):
        """Test basic Poisson disk sampling."""
        points = poisson_disk_sampling(radius=5.0, width=100, height=100, seed=42)
        self.assertGreater(len(points), 0)
    
    def test_minimum_distance(self):
        """Test that points maintain minimum distance."""
        radius = 5.0
        points = poisson_disk_sampling(radius=radius, width=50, height=50, seed=42)
        
        # Check minimum distance between all pairs
        for i, p1 in enumerate(points):
            for p2 in points[i+1:]:
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                self.assertGreaterEqual(dist, radius - 0.001)  # Small tolerance
    
    def test_points_within_bounds(self):
        """Test that all points are within specified bounds."""
        points = poisson_disk_sampling(radius=3.0, width=50, height=80, seed=42)
        
        for p in points:
            self.assertGreaterEqual(p[0], 0)
            self.assertLess(p[0], 50)
            self.assertGreaterEqual(p[1], 0)
            self.assertLess(p[1], 80)


class TestHexagonalPacking(unittest.TestCase):
    """Test hexagonal_packing function."""
    
    def test_correct_count(self):
        """Test that correct number of points is generated."""
        points = hexagonal_packing(spacing=2.0, rows=5, cols=5, center_origin=False)
        self.assertEqual(len(points), 25)
    
    def test_centered_origin(self):
        """Test that points are centered at origin when requested."""
        points = hexagonal_packing(spacing=2.0, rows=10, cols=10, center_origin=True)
        
        xs = [p[0] for p in points]
        zs = [p[2] for p in points]
        
        # Center should be approximately at origin
        center_x = (min(xs) + max(xs)) / 2
        center_z = (min(zs) + max(zs)) / 2
        
        self.assertAlmostEqual(center_x, 0, places=5)
        self.assertAlmostEqual(center_z, 0, places=5)
    
    def test_hexagonal_spacing(self):
        """Test hexagonal spacing pattern."""
        points = hexagonal_packing(spacing=2.0, rows=3, cols=3, center_origin=False)
        
        # Check that odd rows are offset
        # Row 0: x = 0, 2, 4
        # Row 1: x = 1, 3, 5 (offset by spacing/2)
        row0_x = [p[0] for p in points[:3]]
        row1_x = [p[0] for p in points[3:6]]
        
        self.assertAlmostEqual(row1_x[0] - row0_x[0], 1.0, places=5)


class TestRectangularPacking(unittest.TestCase):
    """Test rectangular_packing function."""
    
    def test_correct_count(self):
        """Test that correct number of points is generated."""
        points = rectangular_packing(spacing=2.0, rows=5, cols=8, center_origin=False)
        self.assertEqual(len(points), 40)
    
    def test_centered_origin(self):
        """Test that points are centered at origin when requested."""
        points = rectangular_packing(spacing=2.0, rows=10, cols=10, center_origin=True)
        
        xs = [p[0] for p in points]
        zs = [p[2] for p in points]
        
        center_x = (min(xs) + max(xs)) / 2
        center_z = (min(zs) + max(zs)) / 2
        
        self.assertAlmostEqual(center_x, 0, places=5)
        self.assertAlmostEqual(center_z, 0, places=5)
    
    def test_rectangular_grid(self):
        """Test rectangular grid pattern."""
        points = rectangular_packing(spacing=3.0, rows=3, cols=3, center_origin=False)
        
        # Check uniform spacing
        # Points should be at: (0,0), (3,0), (6,0), (0,3), (3,3), etc.
        self.assertEqual(points[0], (0, 0, 0))
        self.assertEqual(points[1], (3, 0, 0))
        self.assertEqual(points[3], (0, 0, 3))


class TestRelaxPoints(unittest.TestCase):
    """Test relax_points function."""
    
    def test_points_preserved(self):
        """Test that number of points is preserved."""
        initial = [(0, 0), (1, 0), (0, 1), (1, 1)]
        relaxed = relax_points(initial, iterations=10)
        self.assertEqual(len(relaxed), len(initial))
    
    def test_points_move(self):
        """Test that points actually move during relaxation."""
        # Create points that are very close together
        initial = [(0, 0), (0.1, 0), (0, 0.1), (0.1, 0.1)]
        relaxed = relax_points(initial, iterations=50, repulsion_radius=1.0)
        
        # Points should have moved apart
        min_dist_initial = min(
            np.sqrt((initial[i][0] - initial[j][0])**2 + (initial[i][1] - initial[j][1])**2)
            for i in range(len(initial)) for j in range(i+1, len(initial))
        )
        min_dist_relaxed = min(
            np.sqrt((relaxed[i][0] - relaxed[j][0])**2 + (relaxed[i][1] - relaxed[j][1])**2)
            for i in range(len(relaxed)) for j in range(i+1, len(relaxed))
        )
        
        self.assertGreater(min_dist_relaxed, min_dist_initial)


class TestScenePartitioner(unittest.TestCase):
    """Test ScenePartitioner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.points = [
            ScatterPoint(position=(-30, 0, -30), scale=0.5),
            ScatterPoint(position=(-10, 0, -10), scale=1.0),
            ScatterPoint(position=(0, 0, 0), scale=1.5),
            ScatterPoint(position=(10, 0, 10), scale=2.0),
            ScatterPoint(position=(30, 0, 30), scale=2.5),
        ]
        self.partitioner = ScenePartitioner()
    
    def test_group_by_bounding_box(self):
        """Test grouping by bounding box."""
        group = self.partitioner.group_by_bounding_box(
            self.points,
            min_point=(-15, -1, -15),
            max_point=(15, 1, 15),
            group_name="center"
        )
        
        self.assertEqual(len(group), 3)  # -10, 0, 10
        self.assertIn("center", self.partitioner.groups)
    
    def test_filter_by_attribute(self):
        """Test filtering by attribute."""
        # Add scale to attributes
        for p in self.points:
            p.attributes["scale"] = p.scale
        
        filtered = self.partitioner.filter_by_attribute(
            self.points, "scale", 1.0, 2.0
        )
        
        self.assertEqual(len(filtered), 3)  # scale 1.0, 1.5, 2.0
    
    def test_partition(self):
        """Test partitioning by condition."""
        group_a, group_b = self.partitioner.partition(
            self.points,
            condition_func=lambda p: p.scale > 1.0
        )
        
        self.assertEqual(len(group_a), 3)  # scale 1.5, 2.0, 2.5
        self.assertEqual(len(group_b), 2)  # scale 0.5, 1.0
    
    def test_apply_condition(self):
        """Test applying condition with action."""
        def double_scale(p):
            p.scale *= 2
        
        self.partitioner.apply_condition(
            self.points,
            condition_func=lambda p: p.scale > 1.0,
            action_func=double_scale
        )
        
        # Check that scales were doubled
        self.assertEqual(self.points[3].scale, 4.0)  # was 2.0
        self.assertEqual(self.points[4].scale, 5.0)  # was 2.5


class TestLSystem(unittest.TestCase):
    """Test LSystem class."""
    
    def test_generate_string(self):
        """Test L-system string generation."""
        lsys = LSystem(axiom="F", rules={"F": "F+F"}, angle=90)
        result = lsys.generate(iterations=2)
        self.assertEqual(result, "F+F+F+F")
    
    def test_interpret_branches(self):
        """Test L-system interpretation."""
        lsys = LSystem(axiom="F", rules={"F": "F[+F]F[-F]F"}, angle=25.7)
        lstring = lsys.generate(iterations=1)
        result = lsys.interpret(lstring)
        
        self.assertIn("branches", result)
        self.assertIn("leaves", result)
        self.assertGreater(len(result["branches"]), 0)
    
    def test_branch_structure(self):
        """Test that branches have correct structure."""
        lsys = LSystem(axiom="F+F", rules={"F": "F"}, angle=90)
        result = lsys.interpret("FF")
        
        for branch in result["branches"]:
            self.assertEqual(len(branch), 2)  # start and end
            self.assertEqual(len(branch[0]), 3)  # x, y, z


class TestForEachBatchProcess(unittest.TestCase):
    """Test foreach_batch_process function."""
    
    def test_basic_processing(self):
        """Test basic batch processing."""
        items = [1, 2, 3, 4, 5]
        results = foreach_batch_process(items, lambda x, i: x * 2)
        self.assertEqual(results, [2, 4, 6, 8, 10])
    
    def test_with_index(self):
        """Test that index is passed correctly."""
        items = ["a", "b", "c"]
        results = foreach_batch_process(items, lambda x, i: f"{x}{i}")
        self.assertEqual(results, ["a0", "b1", "c2"])


class TestVariantSet(unittest.TestCase):
    """Test VariantSet class."""
    
    def test_add_and_switch(self):
        """Test adding variants and switching."""
        vs = VariantSet("test_variants")
        vs.add_variant("red", {"color": "#ff0000"})
        vs.add_variant("blue", {"color": "#0000ff"})
        
        self.assertTrue(vs.switch_variant("red"))
        self.assertEqual(vs.get_active(), {"color": "#ff0000"})
        
        self.assertTrue(vs.switch_variant("blue"))
        self.assertEqual(vs.get_active(), {"color": "#0000ff"})
    
    def test_invalid_variant(self):
        """Test switching to non-existent variant."""
        vs = VariantSet("test_variants")
        vs.add_variant("valid", {"data": 1})
        
        self.assertFalse(vs.switch_variant("invalid"))


class TestComponentBuilder(unittest.TestCase):
    """Test ComponentBuilder class."""
    
    def test_build_with_defaults(self):
        """Test building with default parameters."""
        builder = ComponentBuilder("test")
        builder.add_parameter("width", 10.0)
        builder.add_parameter("height", 20.0)
        builder.set_build_function(lambda **kwargs: {"w": kwargs["width"], "h": kwargs["height"]})
        
        result = builder.build()
        self.assertEqual(result, {"w": 10.0, "h": 20.0})
    
    def test_build_with_override(self):
        """Test building with parameter override."""
        builder = ComponentBuilder("test")
        builder.add_parameter("width", 10.0)
        builder.add_parameter("height", 20.0)
        builder.set_build_function(lambda **kwargs: {"w": kwargs["width"], "h": kwargs["height"]})
        
        result = builder.build(width=15.0)
        self.assertEqual(result, {"w": 15.0, "h": 20.0})
    
    def test_parameter_constraints(self):
        """Test parameter min/max constraints."""
        builder = ComponentBuilder("test")
        builder.add_parameter("value", 50.0, min_val=0.0, max_val=100.0)
        builder.set_build_function(lambda value: {"v": value})
        
        # Below min
        result = builder.build(value=-10.0)
        self.assertEqual(result["v"], 0.0)
        
        # Above max
        result = builder.build(value=150.0)
        self.assertEqual(result["v"], 100.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_generate_city_block_points(self):
        """Test city block point generation."""
        centers, streets = generate_city_block_points(
            block_size=50, street_width=10, num_blocks_x=3, num_blocks_z=3
        )
        self.assertEqual(len(centers), 9)
        self.assertGreater(len(streets), 0)
    
    def test_generate_tree_structure(self):
        """Test tree structure generation."""
        tree = generate_tree_structure(tree_type="oak", height=10.0, seed=42)
        
        self.assertEqual(tree["type"], "oak")
        self.assertIn("trunk", tree)
        self.assertIn("branches", tree)
        self.assertIn("foliage", tree)
        self.assertGreater(len(tree["branches"]), 0)
    
    def test_sample_height_from_function(self):
        """Test height sampling from function."""
        height_func = lambda x, z: x + z
        height = sample_height_from_function(2, 3, height_func)
        self.assertEqual(height, 5)
    
    def test_calculate_slope(self):
        """Test slope calculation."""
        # Simple slope: z = x
        height_func = lambda x, z: x
        slope = calculate_slope(0, 0, height_func, sample_distance=0.1)
        self.assertGreater(slope, 0)
    
    def test_extrude_building(self):
        """Test building extrusion."""
        footprint = [(0, 0), (10, 0), (10, 10), (0, 10)]
        building = extrude_building(footprint, height=20.0, floors=5)
        
        self.assertEqual(building["height"], 20.0)
        self.assertEqual(building["floors"], 5)
        self.assertIn("walls", building)
        self.assertIn("roof", building)
        self.assertEqual(len(building["walls"]), 4)
    
    def test_create_cylinder_mesh(self):
        """Test cylinder mesh creation."""
        mesh = create_cylinder_mesh(radius=1.0, height=2.0, segments=16)
        
        self.assertIn("vertices", mesh)
        self.assertIn("indices", mesh)
        # 2 centers + 16*2 ring vertices = 34
        self.assertEqual(len(mesh["vertices"]), 34)
    
    def test_create_grid_instances_data(self):
        """Test grid instances data creation."""
        points = create_grid_instances_data(rows=3, cols=4, spacing=2.0)
        
        self.assertEqual(len(points), 12)
        for p in points:
            self.assertIn("row", p.attributes)
            self.assertIn("col", p.attributes)


class TestHeightFieldTerrain(unittest.TestCase):
    """Test HeightFieldTerrain class."""
    
    def test_initialization(self):
        """Test terrain initialization."""
        terrain = HeightFieldTerrain(width=64, height=64, cell_size=1.0, seed=42)
        
        self.assertEqual(terrain.width, 64)
        self.assertEqual(terrain.height, 64)
        self.assertEqual(terrain.height_map.shape, (64, 64))
    
    def test_generate_fractal_terrain(self):
        """Test fractal terrain generation."""
        terrain = HeightFieldTerrain(width=64, height=64, seed=42)
        terrain.generate_fractal_terrain(roughness=0.5, initial_height=100.0)
        
        # Heightmap should have non-zero values
        self.assertGreater(terrain.height_map.max(), 0)
        self.assertNotEqual(terrain.height_map.min(), terrain.height_map.max())
    
    def test_thermal_erosion(self):
        """Test thermal erosion."""
        terrain = HeightFieldTerrain(width=64, height=64, seed=42)
        terrain.generate_fractal_terrain()
        
        initial_range = terrain.height_map.max() - terrain.height_map.min()
        terrain.thermal_erosion(iterations=10)
        eroded_range = terrain.height_map.max() - terrain.height_map.min()
        
        # Erosion should reduce height variation
        self.assertLessEqual(eroded_range, initial_range)
    
    def test_get_height_at(self):
        """Test height query."""
        terrain = HeightFieldTerrain(width=32, height=32, cell_size=1.0)
        terrain.height_map.fill(10.0)
        
        height = terrain.get_height_at(5, 5)
        self.assertEqual(height, 10.0)
    
    def test_get_slope_at(self):
        """Test slope query."""
        terrain = HeightFieldTerrain(width=32, height=32, cell_size=1.0)
        # Create a slope: height increases with x
        for y in range(32):
            for x in range(32):
                terrain.height_map[y, x] = x * 0.1
        
        slope = terrain.get_slope_at(15, 15)
        self.assertIsInstance(slope, float)
        self.assertGreater(slope, 0)
    
    def test_get_normal_at(self):
        """Test normal query."""
        terrain = HeightFieldTerrain(width=32, height=32, cell_size=1.0)
        terrain.height_map.fill(5.0)
        
        normal = terrain.get_normal_at(10, 10)
        self.assertEqual(len(normal), 3)
        # On flat terrain, normal should point up
        self.assertAlmostEqual(normal[1], 1.0, places=5)
    
    def test_scatter_on_terrain(self):
        """Test terrain scatter."""
        terrain = HeightFieldTerrain(width=64, height=64, cell_size=1.0, seed=42)
        terrain.generate_fractal_terrain()
        
        result = terrain.scatter_on_terrain(count=50, seed=42)
        
        self.assertIsInstance(result, TerrainScatterResult)
        self.assertGreater(result.placed_count, 0)
        self.assertLessEqual(result.placed_count, 50)
    
    def test_create_mask_by_feature(self):
        """Test feature mask creation."""
        terrain = HeightFieldTerrain(width=32, height=32, seed=42)
        terrain.generate_fractal_terrain()
        
        mask = terrain.create_mask_by_feature(feature="height", min_value=0, max_value=100)
        
        self.assertEqual(mask.shape, (32, 32))
        self.assertTrue(np.all((mask >= 0) & (mask <= 1)))
    
    def test_to_mesh_data(self):
        """Test mesh data conversion."""
        terrain = HeightFieldTerrain(width=8, height=8, cell_size=1.0)
        terrain.generate_fractal_terrain()
        
        vertices, indices = terrain.to_mesh_data()
        
        self.assertEqual(vertices.shape[0], 64)  # 8x8 vertices
        self.assertGreater(len(indices), 0)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        terrain = HeightFieldTerrain(width=32, height=32, seed=42)
        terrain.generate_fractal_terrain()
        
        stats = terrain.get_statistics()
        
        self.assertIn("min_height", stats)
        self.assertIn("max_height", stats)
        self.assertIn("mean_height", stats)
        self.assertIn("dimensions", stats)


class TestCityGenerator(unittest.TestCase):
    """Test CityGenerator class."""
    
    def test_generate_grid(self):
        """Test grid city generation."""
        gen = CityGenerator(block_size=50, street_width=10, seed=42)
        blocks, streets = gen.generate_grid(num_blocks_x=3, num_blocks_z=4)
        
        self.assertEqual(len(blocks), 12)
        self.assertGreater(len(streets), 0)
        
        for block in blocks:
            self.assertIsInstance(block, CityBlock)
    
    def test_generate_organic(self):
        """Test organic city generation."""
        gen = CityGenerator(seed=42)
        blocks, roads = gen.generate_organic(num_main_roads=3, city_radius=200)
        
        self.assertGreater(len(roads), 0)


class TestBuildingGenerator(unittest.TestCase):
    """Test BuildingGenerator class."""
    
    def test_create_footprint(self):
        """Test footprint creation."""
        gen = BuildingGenerator(seed=42)
        footprint = gen.create_footprint(width=10, depth=20, corner_radius=0)
        
        self.assertEqual(len(footprint), 4)
    
    def test_create_rounded_footprint(self):
        """Test rounded footprint creation."""
        gen = BuildingGenerator(seed=42)
        footprint = gen.create_footprint(width=10, depth=20, corner_radius=2.0)
        
        self.assertGreater(len(footprint), 4)  # More vertices for rounded corners
    
    def test_generate_building(self):
        """Test building generation."""
        gen = BuildingGenerator(seed=42)
        params = BuildingParameters(
            width=15, depth=15, height=30, floors=10,
            style=BuildingStyle.MODERN, seed=42
        )
        building = gen.generate_building(params)
        
        self.assertIsInstance(building, BuildingData)
        self.assertEqual(building.params.style, BuildingStyle.MODERN)
        self.assertIn("roof_type", building.attributes)
    
    def test_populate_block(self):
        """Test block population."""
        gen = BuildingGenerator(seed=42)
        block = CityBlock(center=(0, 0), size=50)
        
        buildings = gen.populate_block(block, density=0.7, height_range=(10, 50))
        
        self.assertGreater(len(buildings), 0)
        self.assertEqual(len(block.buildings), len(buildings))


class TestStreetNetwork(unittest.TestCase):
    """Test StreetNetwork class."""
    
    def test_create_road_mesh(self):
        """Test road mesh creation."""
        vertices, indices = StreetNetwork.create_road_mesh(
            start=(0, 0), end=(10, 0), width=4.0, segments=5
        )
        
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(indices), 0)
    
    def test_generate_sidewalks(self):
        """Test sidewalk generation."""
        streets = [(0, 0, 10, 4), (10, 0, 4, 10)]
        sidewalks = StreetNetwork.generate_sidewalks(streets, sidewalk_width=1.5)
        
        self.assertEqual(len(sidewalks), len(streets))
        # Sidewalks should be larger than streets
        self.assertGreater(sidewalks[0][2], streets[0][2])


class TestGenerateCompleteCity(unittest.TestCase):
    """Test generate_complete_city function."""
    
    def test_complete_city(self):
        """Test complete city generation."""
        city = generate_complete_city(
            num_blocks_x=4, num_blocks_z=4,
            block_size=40, street_width=8, seed=42
        )
        
        self.assertIn("blocks", city)
        self.assertIn("streets", city)
        self.assertIn("buildings", city)
        self.assertIn("statistics", city)
        
        self.assertEqual(city["statistics"]["num_blocks"], 16)
        self.assertGreater(city["statistics"]["num_buildings"], 0)


class TestMeshBuilderIntegration(unittest.TestCase):
    """Test MeshBuilder integration with other Houdini modules."""
    
    def test_mesh_builder_with_cylinder_creation(self):
        """Test MeshBuilder with cylinder mesh creation."""
        # Create cylinder mesh data from houdini_nodes
        mesh = create_cylinder_mesh(radius=1.0, height=2.0, segments=8)
        
        # Build mesh using MeshBuilder
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        # Add vertices from cylinder
        for vertex in mesh["vertices"]:
            builder.add_vertex(vertex)
        
        # Add triangles (indices are tuples in the mesh data)
        for tri in mesh["indices"]:
            builder.add_triangle(0, tri[0], tri[1], tri[2])
        
        # Validate
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.vertex_count(), len(mesh["vertices"]))
    
    def test_mesh_builder_with_extruded_building(self):
        """Test MeshBuilder with extruded building footprint."""
        # Create building extrusion
        footprint = [(0, 0), (10, 0), (10, 10), (0, 10)]
        building = extrude_building(footprint, height=20.0, floors=5)
        
        # Create mesh from building walls
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add vertices from wall data (each wall has 4 vertices)
        vertex_map = {}  # Track unique vertices
        for wall in building["walls"]:
            for v in wall["vertices"]:
                # v is (x, y, z) tuple
                key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
                if key not in vertex_map:
                    vertex_map[key] = builder.add_vertex(v)
        
        # Add vertices from roof
        for v in building["roof"]["vertices"]:
            key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
            if key not in vertex_map:
                vertex_map[key] = builder.add_vertex(v)
        
        # Create a simple triangle fan from the roof vertices
        roof_vertices = building["roof"]["vertices"]
        if len(roof_vertices) >= 3:
            # Get indices for roof vertices
            indices = []
            for v in roof_vertices:
                key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
                indices.append(vertex_map[key])
            
            # Create triangle fan
            for i in range(1, len(indices) - 1):
                builder.add_triangle(0, indices[0], indices[i], indices[i + 1])
        
        # Validate
        if builder.vertex_count() > 0:
            error = builder.check()
            self.assertEqual(error, "")
    
    def test_mesh_builder_with_terrain(self):
        """Test MeshBuilder with terrain mesh conversion."""
        # Create terrain
        terrain = HeightFieldTerrain(width=16, height=16, cell_size=1.0, seed=42)
        terrain.generate_fractal_terrain()
        
        # Get mesh data from terrain
        vertices, indices = terrain.to_mesh_data()
        
        # Build mesh using MeshBuilder
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add vertices
        for vertex in vertices:
            builder.add_vertex(vertex)
        
        # Add triangles
        for i in range(0, len(indices), 3):
            builder.add_triangle(0, indices[i], indices[i+1], indices[i+2])
        
        # Validate
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.vertex_count(), vertices.shape[0])
    
    def test_mesh_builder_with_scatter_points(self):
        """Test MeshBuilder with scattered points as instanced markers."""
        # Scatter points on surface
        points = scatter_on_surface(bounds=(-10, 10), count=10, seed=42)
        
        # Create a simple marker mesh (triangle) for each point
        builder = MeshBuilder()
        
        # Create submesh for markers
        marker_submesh = builder.add_submesh()
        
        for point in points:
            pos = point.position
            scale = point.scale
            
            # Create a small triangle marker at each point position
            v0 = builder.add_vertex((pos[0] - scale, pos[1], pos[2] - scale))
            v1 = builder.add_vertex((pos[0] + scale, pos[1], pos[2] - scale))
            v2 = builder.add_vertex((pos[0], pos[1] + scale, pos[2]))
            
            builder.add_triangle(marker_submesh, v0, v1, v2)
        
        # Validate
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.vertex_count(), len(points) * 3)
    
    def test_mesh_builder_with_tree_structure(self):
        """Test MeshBuilder with L-system tree structure."""
        # Generate tree structure
        tree = generate_tree_structure(tree_type="oak", height=10.0, seed=42)
        
        builder = MeshBuilder()
        trunk_submesh = builder.add_submesh()
        branch_submesh = builder.add_submesh()
        
        # Add trunk vertices and triangles
        # trunk is a dict with "height" and "radius"
        trunk_height = tree["trunk"]["height"]
        trunk_radius = tree["trunk"]["radius"]
        
        # Create a simple pyramid trunk
        v0 = builder.add_vertex((-trunk_radius, 0, -trunk_radius))
        v1 = builder.add_vertex((trunk_radius, 0, -trunk_radius))
        v2 = builder.add_vertex((trunk_radius, 0, trunk_radius))
        v3 = builder.add_vertex((-trunk_radius, 0, trunk_radius))
        v_top = builder.add_vertex((0, trunk_height, 0))
        
        # Create triangles for pyramid trunk
        builder.add_triangle(trunk_submesh, v0, v1, v_top)
        builder.add_triangle(trunk_submesh, v1, v2, v_top)
        builder.add_triangle(trunk_submesh, v2, v3, v_top)
        builder.add_triangle(trunk_submesh, v3, v0, v_top)
        
        # Add foliage spheres as simple tetrahedrons to branch submesh
        for foliage in tree.get("foliage", [])[:3]:  # Limit to first 3
            pos = foliage["position"]
            radius = foliage["radius"]
            
            # Simple tetrahedron for foliage cluster
            f0 = builder.add_vertex((pos[0], pos[1] + radius, pos[2]))
            f1 = builder.add_vertex((pos[0] + radius, pos[1] - radius, pos[2] + radius))
            f2 = builder.add_vertex((pos[0] - radius, pos[1] - radius, pos[2] + radius))
            f3 = builder.add_vertex((pos[0], pos[1] - radius, pos[2] - radius))
            
            builder.add_triangle(branch_submesh, f0, f1, f2)
            builder.add_triangle(branch_submesh, f0, f2, f3)
            builder.add_triangle(branch_submesh, f0, f3, f1)
            builder.add_triangle(branch_submesh, f1, f2, f3)
        
        # Validate
        if builder.vertex_count() > 0:
            error = builder.check()
            self.assertEqual(error, "")
            self.assertEqual(builder.submesh_count(), 2)
    
    def test_mesh_builder_with_city_blocks(self):
        """Test MeshBuilder with city block visualization."""
        # Generate city layout
        city_gen = CityGenerator(block_size=50, street_width=10, seed=42)
        blocks, _ = city_gen.generate_grid(2, 2)
        
        builder = MeshBuilder()
        block_submesh = builder.add_submesh()
        
        for block in blocks:
            cx, cz = block.center
            size = block.size
            
            # Create a quad for each city block
            half_size = size / 2
            v0 = builder.add_vertex((cx - half_size, 0, cz - half_size))
            v1 = builder.add_vertex((cx + half_size, 0, cz - half_size))
            v2 = builder.add_vertex((cx + half_size, 0, cz + half_size))
            v3 = builder.add_vertex((cx - half_size, 0, cz + half_size))
            
            # Two triangles per block
            builder.add_triangle(block_submesh, v0, v1, v2)
            builder.add_triangle(block_submesh, v0, v2, v3)
        
        # Validate
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.submesh_count(), 1)
    
    def test_mesh_builder_with_hexagonal_packing(self):
        """Test MeshBuilder with hexagonal packing points."""
        # Generate hexagonal packing
        points = hexagonal_packing(spacing=2.0, rows=5, cols=5, center_origin=True)
        
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        # Create hexagonal prisms at each packing point
        for point in points:
            x, y, z = point
            
            # Hexagon vertices (top face)
            hex_vertices_top = []
            hex_vertices_bottom = []
            for i in range(6):
                angle = i * (2 * np.pi / 6)
                hx = x + 0.5 * np.cos(angle)
                hz = z + 0.5 * np.sin(angle)
                hex_vertices_bottom.append(builder.add_vertex((hx, y, hz)))
                hex_vertices_top.append(builder.add_vertex((hx, y + 1.0, hz)))
            
            # Create triangles for hexagonal prism
            # Top face (fan from center)
            center_top = builder.add_vertex((x, y + 1.0, z))
            for i in range(6):
                next_i = (i + 1) % 6
                builder.add_triangle(0, center_top, hex_vertices_top[i], hex_vertices_top[next_i])
            
            # Bottom face (fan from center)
            center_bottom = builder.add_vertex((x, y, z))
            for i in range(6):
                next_i = (i + 1) % 6
                builder.add_triangle(0, center_bottom, hex_vertices_bottom[next_i], hex_vertices_bottom[i])
        
        # Validate
        if builder.vertex_count() > 0:
            error = builder.check()
            self.assertEqual(error, "")
    
    def test_mesh_builder_multiple_uv_sets_with_terrain(self):
        """Test MeshBuilder with multiple UV sets for terrain blending."""
        # Create small terrain
        terrain = HeightFieldTerrain(width=8, height=8, cell_size=1.0, seed=42)
        terrain.generate_fractal_terrain()
        
        vertices, indices = terrain.to_mesh_data()
        
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add two UV sets (e.g., for base texture and detail texture)
        builder.add_uv_set()  # UV0: base texture coordinates
        builder.add_uv_set()  # UV1: detail texture coordinates
        
        # Add vertices
        for vertex in vertices:
            builder.add_vertex(vertex)
        
        # Add triangles
        for i in range(0, len(indices), 3):
            builder.add_triangle(0, indices[i], indices[i+1], indices[i+2])
        
        # Generate UVs for both sets
        # UV0: Standard planar mapping
        uv0_data = []
        for vertex in vertices:
            u = (vertex[0] + 4) / 8  # Normalize to 0-1
            v = (vertex[2] + 4) / 8
            uv0_data.append([u, v])
        builder.uvs[0] = np.array(uv0_data, dtype=np.float32)
        
        # UV1: Detail mapping (tiled 4x)
        uv1_data = []
        for vertex in vertices:
            u = ((vertex[0] + 4) / 8) * 4  # Tiled
            v = ((vertex[2] + 4) / 8) * 4
            uv1_data.append([u, v])
        builder.uvs[1] = np.array(uv1_data, dtype=np.float32)
        
        # Calculate tangents
        triangles_arr = np.array(indices, dtype=np.uint32).reshape(-1, 3)
        builder.tangent = MeshBuilder.calculate_tangent(
            builder.position, builder.uvs[0], triangles_arr
        )
        
        # Add normals (upward for flat terrain approximation)
        builder.normal = np.tile([0, 1, 0], (builder.vertex_count(), 1)).astype(np.float32)
        
        # Validate
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.uv_count(), 2)
        self.assertTrue(builder.contained_normal())
        self.assertTrue(builder.contained_tangent())
    
    def test_mesh_builder_submesh_per_building_style(self):
        """Test MeshBuilder with submeshes for different building styles."""
        # Generate buildings with different styles
        gen = BuildingGenerator(seed=42)
        block = CityBlock(center=(0, 0), size=100)
        buildings = gen.populate_block(block, density=0.5)
        
        builder = MeshBuilder()
        
        # Create a submesh for each building style
        style_submeshes = {}
        for style in BuildingStyle:
            style_submeshes[style] = builder.add_submesh()
        
        # Add simplified building geometry
        for building in buildings[:5]:  # Limit to first 5
            params = building.params
            submesh_idx = style_submeshes.get(params.style, 0)
            
            # Calculate building center from footprint
            footprint = building.footprint
            cx = sum(p[0] for p in footprint) / len(footprint)
            cz = sum(p[1] for p in footprint) / len(footprint)
            
            width = params.width
            depth = params.depth
            height = params.height
            
            # Bottom face vertices
            v0 = builder.add_vertex((cx - width/2, 0, cz - depth/2))
            v1 = builder.add_vertex((cx + width/2, 0, cz - depth/2))
            v2 = builder.add_vertex((cx + width/2, 0, cz + depth/2))
            v3 = builder.add_vertex((cx - width/2, 0, cz + depth/2))
            
            # Top face vertices
            v4 = builder.add_vertex((cx - width/2, height, cz - depth/2))
            v5 = builder.add_vertex((cx + width/2, height, cz - depth/2))
            v6 = builder.add_vertex((cx + width/2, height, cz + depth/2))
            v7 = builder.add_vertex((cx - width/2, height, cz + depth/2))
            
            # Top face triangles
            builder.add_triangle(submesh_idx, v4, v5, v6)
            builder.add_triangle(submesh_idx, v4, v6, v7)
        
        # Validate
        if builder.vertex_count() > 0:
            error = builder.check()
            self.assertEqual(error, "")
            self.assertEqual(builder.submesh_count(), len(BuildingStyle))
    
    def test_mesh_builder_building_with_floors(self):
        """Test MeshBuilder creating detailed building with floor subdivisions."""
        gen = BuildingGenerator(seed=42)
        params = BuildingParameters(
            width=20, depth=15, height=30, floors=10,
            style=BuildingStyle.MODERN, seed=42
        )
        building = gen.generate_building(params)
        
        builder = MeshBuilder()
        walls_submesh = builder.add_submesh()
        roof_submesh = builder.add_submesh()
        
        # Create footprint from building data
        footprint = building.footprint
        height = building.params.height
        floors = building.params.floors
        floor_height = height / floors
        
        # Generate wall vertices with floor subdivisions
        wall_vertices_bottom = []
        wall_vertices_top = []
        
        for i, (x, z) in enumerate(footprint):
            # Bottom vertices
            v_bottom = builder.add_vertex((x, 0, z))
            wall_vertices_bottom.append(v_bottom)
            
            # Top vertices
            v_top = builder.add_vertex((x, height, z))
            wall_vertices_top.append(v_top)
        
        # Create wall faces (with floor lines as additional geometry)
        n = len(footprint)
        for i in range(n):
            next_i = (i + 1) % n
            
            # Wall quad: bottom[i], bottom[next], top[next], top[i]
            v0 = wall_vertices_bottom[i]
            v1 = wall_vertices_bottom[next_i]
            v2 = wall_vertices_top[next_i]
            v3 = wall_vertices_top[i]
            
            # Two triangles per wall
            builder.add_triangle(walls_submesh, v0, v1, v2)
            builder.add_triangle(walls_submesh, v0, v2, v3)
        
        # Create roof (fan triangulation)
        # Calculate roof center
        cx = sum(p[0] for p in footprint) / len(footprint)
        cz = sum(p[1] for p in footprint) / len(footprint)
        roof_center = builder.add_vertex((cx, height, cz))
        
        for i in range(n):
            next_i = (i + 1) % n
            builder.add_triangle(roof_submesh, roof_center, wall_vertices_top[i], wall_vertices_top[next_i])
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertEqual(builder.submesh_count(), 2)
    
    def test_mesh_builder_complete_city_meshes(self):
        """Test MeshBuilder generating meshes for complete city."""
        city = generate_complete_city(
            num_blocks_x=2, num_blocks_z=2,
            block_size=40, street_width=8, seed=42
        )
        
        builder = MeshBuilder()
        building_submesh = builder.add_submesh()
        street_submesh = builder.add_submesh()
        
        # Generate building meshes
        for building in city["buildings"][:10]:  # Limit for performance
            params = building.params
            footprint = building.footprint
            
            # Calculate center
            cx = sum(p[0] for p in footprint) / len(footprint)
            cz = sum(p[1] for p in footprint) / len(footprint)
            
            width = params.width
            depth = params.depth
            height = params.height
            
            # Simple box for each building
            base_vertices = []
            top_vertices = []
            
            # Create 4 corners
            corners = [
                (cx - width/2, cz - depth/2),
                (cx + width/2, cz - depth/2),
                (cx + width/2, cz + depth/2),
                (cx - width/2, cz + depth/2),
            ]
            
            for x, z in corners:
                base_vertices.append(builder.add_vertex((x, 0, z)))
                top_vertices.append(builder.add_vertex((x, height, z)))
            
            # Create walls
            for i in range(4):
                next_i = (i + 1) % 4
                builder.add_triangle(building_submesh, base_vertices[i], base_vertices[next_i], top_vertices[next_i])
                builder.add_triangle(building_submesh, base_vertices[i], top_vertices[next_i], top_vertices[i])
            
            # Create roof
            for i in range(1, 3):
                builder.add_triangle(building_submesh, top_vertices[0], top_vertices[i], top_vertices[i + 1])
        
        # Generate street meshes (simple quads)
        for street in city["streets"][:5]:  # Limit streets
            x, z, w, d = street
            
            v0 = builder.add_vertex((x, 0.1, z))  # Slightly above ground
            v1 = builder.add_vertex((x + w, 0.1, z))
            v2 = builder.add_vertex((x + w, 0.1, z + d))
            v3 = builder.add_vertex((x, 0.1, z + d))
            
            builder.add_triangle(street_submesh, v0, v1, v2)
            builder.add_triangle(street_submesh, v0, v2, v3)
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertGreaterEqual(builder.submesh_count(), 2)
    
    def test_mesh_builder_heightfield_with_normals(self):
        """Test MeshBuilder with heightfield including normal calculation."""
        # Create terrain with specific features
        terrain = HeightFieldTerrain(width=32, height=32, cell_size=1.0, seed=42)
        terrain.generate_fractal_terrain(roughness=0.5)
        terrain.thermal_erosion(iterations=10)
        
        # Get mesh data
        vertices, indices = terrain.to_mesh_data()
        
        builder = MeshBuilder()
        builder.add_submesh()
        
        # Add all vertices
        for vertex in vertices:
            builder.add_vertex(vertex)
        
        # Add triangles
        indices_arr = np.array(indices, dtype=np.uint32)
        for i in range(0, len(indices_arr), 3):
            builder.add_triangle(0, indices_arr[i], indices_arr[i+1], indices_arr[i+2])
        
        # Calculate normals from heightfield (more accurate than flat)
        normals = []
        for y in range(terrain.height):
            for x in range(terrain.width):
                normal = terrain.get_normal_at(x, y)
                normals.append(normal)
        
        builder.normal = np.array(normals, dtype=np.float32)
        
        # Add UVs (planar mapping)
        builder.add_uv_set()
        uvs = []
        for y in range(terrain.height):
            for x in range(terrain.width):
                u = x / (terrain.width - 1)
                v = y / (terrain.height - 1)
                uvs.append([u, v])
        builder.uvs[0] = np.array(uvs, dtype=np.float32)
        
        # Calculate tangents
        triangles_arr = indices_arr.reshape(-1, 3)
        builder.tangent = MeshBuilder.calculate_tangent(
            builder.position, builder.uvs[0], triangles_arr
        )
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertTrue(builder.contained_normal())
        self.assertTrue(builder.contained_tangent())
    
    def test_mesh_builder_heightfield_lod(self):
        """Test MeshBuilder creating LOD meshes from heightfield."""
        terrain = HeightFieldTerrain(width=64, height=64, cell_size=1.0, seed=42)
        terrain.generate_fractal_terrain()
        
        # Create LOD levels by sampling
        lod_levels = [1, 2, 4]  # Full, half, quarter resolution
        
        for lod in lod_levels:
            builder = MeshBuilder()
            builder.add_submesh()
            
            step = lod
            lod_width = terrain.width // step
            lod_height = terrain.height // step
            
            # Sample vertices at LOD resolution
            vertex_map = {}  # (x, y) -> vertex index
            for y in range(0, terrain.height, step):
                for x in range(0, terrain.width, step):
                    height = terrain.get_height_at(x, y)
                    vx = x * terrain.cell_size
                    vz = y * terrain.cell_size
                    idx = builder.add_vertex((vx, height, vz))
                    vertex_map[(x // step, y // step)] = idx
            
            # Create triangles
            for y in range(lod_height - 1):
                for x in range(lod_width - 1):
                    v0 = vertex_map[(x, y)]
                    v1 = vertex_map[(x + 1, y)]
                    v2 = vertex_map[(x + 1, y + 1)]
                    v3 = vertex_map[(x, y + 1)]
                    
                    builder.add_triangle(0, v0, v1, v2)
                    builder.add_triangle(0, v0, v2, v3)
            
            error = builder.check()
            self.assertEqual(error, "")
            
            # Verify vertex count decreases with higher LOD
            expected_vertices = lod_width * lod_height
            self.assertEqual(builder.vertex_count(), expected_vertices)
    
    def test_mesh_builder_extrude_building_detailed(self):
        """Test MeshBuilder with detailed extruded building."""
        # Create complex footprint with more points
        footprint = [
            (0, 0), (10, 0), (15, 5), (10, 10), (5, 12), (0, 10)
        ]
        
        building = extrude_building(footprint, height=25.0, floors=8)
        
        builder = MeshBuilder()
        walls_submesh = builder.add_submesh()
        roof_submesh = builder.add_submesh()
        
        # Track unique vertices to avoid duplicates
        vertex_cache = {}
        
        def get_vertex(x, y, z):
            key = (round(x, 6), round(y, 6), round(z, 6))
            if key not in vertex_cache:
                vertex_cache[key] = builder.add_vertex((x, y, z))
            return vertex_cache[key]
        
        # Create walls from building data
        for wall in building["walls"]:
            vertices = wall["vertices"]  # List of (x, y, z) tuples
            
            # Wall should have 4 vertices (bottom-left, bottom-right, top-right, top-left)
            if len(vertices) >= 4:
                v0 = get_vertex(*vertices[0])
                v1 = get_vertex(*vertices[1])
                v2 = get_vertex(*vertices[2])
                v3 = get_vertex(*vertices[3])
                
                # Two triangles per wall
                builder.add_triangle(walls_submesh, v0, v1, v2)
                builder.add_triangle(walls_submesh, v0, v2, v3)
        
        # Create roof
        roof_vertices = building["roof"]["vertices"]
        roof_indices = []
        for v in roof_vertices:
            roof_indices.append(get_vertex(*v))
        
        # Fan triangulation for roof
        if len(roof_indices) >= 3:
            for i in range(1, len(roof_indices) - 1):
                builder.add_triangle(roof_submesh, roof_indices[0], roof_indices[i], roof_indices[i + 1])
        
        # Create floor slabs as separate submesh
        floors_submesh = builder.add_submesh()
        for floor_slab in building["floor_slabs"][::4]:  # Every 4th floor
            floor_vertices = floor_slab["vertices"]
            floor_indices = []
            for v in floor_vertices:
                floor_indices.append(get_vertex(*v))
            
            if len(floor_indices) >= 3:
                for i in range(1, len(floor_indices) - 1):
                    builder.add_triangle(floors_submesh, floor_indices[0], floor_indices[i], floor_indices[i + 1])
        
        error = builder.check()
        self.assertEqual(error, "")
        self.assertGreaterEqual(builder.submesh_count(), 2)
    
    def test_mesh_builder_extrude_building_rounded(self):
        """Test MeshBuilder with rounded building extrusion."""
        # Create rounded footprint using many points
        import math
        radius = 10.0
        segments = 16
        footprint = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            footprint.append((x, z))
        
        building = extrude_building(footprint, height=20.0, floors=5)
        
        builder = MeshBuilder()
        builder.add_submesh()
        builder.add_uv_set()
        
        # Add vertices
        vertex_cache = {}
        
        def get_vertex(x, y, z):
            key = (round(x, 6), round(y, 6), round(z, 6))
            if key not in vertex_cache:
                vertex_cache[key] = builder.add_vertex((x, y, z))
            return vertex_cache[key]
        
        # Create cylindrical walls
        for wall in building["walls"]:
            vertices = wall["vertices"]
            if len(vertices) >= 4:
                v0 = get_vertex(*vertices[0])
                v1 = get_vertex(*vertices[1])
                v2 = get_vertex(*vertices[2])
                v3 = get_vertex(*vertices[3])
                
                builder.add_triangle(0, v0, v1, v2)
                builder.add_triangle(0, v0, v2, v3)
        
        # Create top cap (circle fan)
        roof_vertices = building["roof"]["vertices"]
        center_x = sum(v[0] for v in roof_vertices) / len(roof_vertices)
        center_z = sum(v[2] for v in roof_vertices) / len(roof_vertices)
        center_y = building["height"]
        
        center_idx = get_vertex(center_x, center_y, center_z)
        roof_indices = [get_vertex(*v) for v in roof_vertices]
        
        for i in range(len(roof_indices)):
            next_i = (i + 1) % len(roof_indices)
            builder.add_triangle(0, center_idx, roof_indices[i], roof_indices[next_i])
        
        # Generate UVs for cylindrical mapping
        uvs = []
        for i in range(builder.vertex_count()):
            pos = builder.position[i]
            # Cylindrical UV mapping
            angle = math.atan2(pos[2], pos[0])
            u = (angle / (2 * math.pi) + 0.5)
            v = pos[1] / building["height"]
            uvs.append([u, v])
        
        builder.uvs[0] = np.array(uvs, dtype=np.float32)
        
        error = builder.check()
        self.assertEqual(error, "")


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
