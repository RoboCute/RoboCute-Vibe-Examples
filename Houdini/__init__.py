"""
Houdini-Style Procedural Scene Generation for RoboCute
======================================================

This package implements Houdini's procedural scene generation nodes
using the RoboCute Python API.

Modules:
--------
houdini_nodes.py        - Core scattering, distribution, and constraint systems
procedural_terrain.py   - HeightField terrain generation and erosion
city_generator.py       - City layout and building generation

Examples:
---------
examples/forest_scene.py    - Complete forest generation workflow
examples/city_scene.py      - Complete city generation workflow
examples/scatter_example.py - Basic scattering techniques

Usage:
------
    from houdini_nodes import scatter_on_surface, ScenePartitioner
    from procedural_terrain import HeightFieldTerrain
    from city_generator import CityGenerator

Houdini Node Equivalents:
-------------------------
Scatter                 → scatter_on_surface, poisson_disk_sampling
Copy to Points          → create_instances_at_points
Copy and Transform      → create_grid_instances_data
Hexagonal Packing       → hexagonal_packing
Relax                   → relax_points
Group by Bounding Box   → ScenePartitioner.group_by_bounding_box
Attribute Wrangle       → ScenePartitioner.apply_condition
L-System                → LSystem
For-Each                → foreach_batch_process
PolyExtrude             → BuildingGenerator.create_footprint
Variant LOP             → VariantSet
Component Builder       → ComponentBuilder
HeightField             → HeightFieldTerrain
HeightField Erode       → thermal_erosion, hydraulic_erosion
HeightField Scatter     → scatter_on_terrain
Labs Building Generator → BuildingGenerator
Labs City Generator     → CityGenerator
"""

__version__ = "1.0.0"
__author__ = "RoboCute Project"

from .houdini_nodes import (
    # Scattering
    ScatterPoint,
    scatter_on_surface,
    poisson_disk_sampling,
    hexagonal_packing,
    rectangular_packing,
    relax_points,
    create_grid_instances_data,
    
    # Constraints & Partitioning
    ScenePartitioner,
    
    # Procedural Modeling
    LSystem,
    foreach_batch_process,
    
    # Solaris/Stage
    VariantSet,
    ComponentBuilder,
    
    # Procedural Modeling
    extrude_building,
    create_cylinder_mesh,
    
    # Utilities
    generate_city_block_points,
    generate_tree_structure,
    sample_height_from_function,
    calculate_slope,
)

from .procedural_terrain import (
    HeightFieldTerrain,
    TerrainScatterResult,
)

from .city_generator import (
    CityGenerator,
    BuildingGenerator,
    BuildingParameters,
    BuildingStyle,
    BuildingData,
    CityBlock,
    StreetNetwork,
    generate_complete_city,
)

__all__ = [
    # Version
    "__version__",
    
    # Core types
    "ScatterPoint",
    "TerrainScatterResult",
    "BuildingParameters",
    "BuildingStyle",
    "BuildingData",
    "CityBlock",
    
    # Scattering functions
    "scatter_on_surface",
    "poisson_disk_sampling",
    "hexagonal_packing",
    "rectangular_packing",
    "relax_points",
    "create_grid_instances_data",
    
    # Classes
    "ScenePartitioner",
    "LSystem",
    "VariantSet",
    "ComponentBuilder",
    "HeightFieldTerrain",
    "CityGenerator",
    "BuildingGenerator",
    "StreetNetwork",
    
    # Procedural Modeling
    "extrude_building",
    "create_cylinder_mesh",
    
    # Utility functions
    "foreach_batch_process",
    "generate_city_block_points",
    "generate_tree_structure",
    "generate_complete_city",
    "sample_height_from_function",
    "calculate_slope",
]
