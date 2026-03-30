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
