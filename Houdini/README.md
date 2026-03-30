# Houdini-Style Procedural Scene Generation for RoboCute

A comprehensive Python implementation of Houdini's procedural scene generation nodes, adapted for the RoboCute rendering engine.

## Overview

This project translates Houdini's powerful procedural workflows into Python code compatible with RoboCute's Python API. It enables artists and developers to create complex, data-driven scenes using familiar Houdini concepts.

### What is Procedural Generation?

Procedural generation is a method of creating content algorithmically rather than manually. Houdini excels at this through its node-based workflow where each operation (node) transforms data that flows to the next operation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Houdini Node Equivalents                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SOPs (Surface Operators)                                       │
│  ├── Scatter          →  scatter_on_surface()                   │
│  ├── Copy to Points   →  create_instances_at_points()           │
│  ├── Copy & Transform →  create_grid_instances_data()           │
│  ├── Hexagonal Pack   →  hexagonal_packing()                    │
│  ├── Relax            →  relax_points()                         │
│  ├── Group            →  ScenePartitioner                       │
│  ├── Attribute Wrangle→  ScenePartitioner.apply_condition()     │
│  ├── L-System         →  LSystem                                │
│  ├── For-Each         →  foreach_batch_process()                │
│  └── PolyExtrude      →  BuildingGenerator                      │
│                                                                  │
│  HeightField System                                             │
│  ├── HeightField      →  HeightFieldTerrain                     │
│  ├── Thermal Erosion  →  thermal_erosion()                      │
│  ├── Hydraulic Erosion→  hydraulic_erosion()                    │
│  ├── Scatter          →  scatter_on_terrain()                   │
│  └── Mask by Feature  →  create_mask_by_feature()               │
│                                                                  │
│  Solaris / Stage (USD)                                          │
│  ├── Variant LOP      →  VariantSet                             │
│  └── Component Builder→  ComponentBuilder                       │
│                                                                  │
│  Labs Tools                                                     │
│  ├── Building Generator→ BuildingGenerator                      │
│  └── City Generator   →  CityGenerator                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
samples/Houdini/
├── README.md                   # This file
├── __init__.py                 # Package initialization
│
├── houdini_nodes.py            # Core procedural generation modules
│   ├── Scattering functions    # Scatter, Poisson disk, packing
│   ├── ScenePartitioner        # Grouping and filtering
│   ├── LSystem                 # Fractal generation
│   ├── VariantSet              # Asset variants
│   └── ComponentBuilder        # Reusable components
│
├── procedural_terrain.py       # HeightField terrain system
│   ├── HeightFieldTerrain      # Main terrain class
│   ├── generate_noise_terrain  # Fractal terrain
│   ├── thermal_erosion         # Thermal erosion simulation
│   ├── hydraulic_erosion       # Hydraulic erosion simulation
│   └── scatter_on_terrain      # Feature-based scattering
│
├── city_generator.py           # Urban generation system
│   ├── CityGenerator           # City layout generation
│   ├── BuildingGenerator       # Procedural buildings
│   ├── BuildingParameters      # Building configuration
│   └── StreetNetwork           # Road system utilities
│
└── examples/
    ├── forest_scene.py         # Forest generation example
    ├── city_scene.py           # City generation example
    └── scatter_example.py      # Scattering techniques demo
```

## Installation

No additional installation required beyond the RoboCute Python environment with NumPy.

```python
# Add to your Python path
import sys
sys.path.insert(0, 'samples/Houdini')

# Import modules
from houdini_nodes import scatter_on_surface, ScenePartitioner
from procedural_terrain import HeightFieldTerrain
from city_generator import CityGenerator
```

## Quick Start

### 1. Basic Scattering

```python
from houdini_nodes import scatter_on_surface

# Scatter 100 objects in a 100x100 area
points = scatter_on_surface(
    bounds=(-50, 50),
    count=100,
    scale_range=(0.5, 1.5),
    seed=42
)

for point in points:
    print(f"Position: {point.position}, Scale: {point.scale}")
```

### 2. Terrain Generation

```python
from procedural_terrain import HeightFieldTerrain

# Create terrain
terrain = HeightFieldTerrain(width=256, height=256, cell_size=0.5)
terrain.generate_fractal_terrain(roughness=0.5)
terrain.thermal_erosion(iterations=100, talus_angle=45.0)

# Scatter trees based on slope
result = terrain.scatter_on_terrain(
    count=500,
    min_slope=0,
    max_slope=20,  # Flat areas only
    min_height=10,
    max_height=50
)

print(f"Placed {result.placed_count} trees")
```

### 3. City Generation

```python
from city_generator import generate_complete_city

# Generate complete city
city = generate_complete_city(
    num_blocks_x=8,
    num_blocks_z=8,
    block_size=40.0,
    street_width=8.0,
    seed=42
)

print(f"Generated {city['statistics']['num_buildings']} buildings")
```

## Detailed Usage

### Scattering Techniques

#### Poisson Disk Sampling (Blue Noise)

Creates natural point distribution with guaranteed minimum distance:

```python
from houdini_nodes import poisson_disk_sampling

points = poisson_disk_sampling(
    radius=5.0,      # Minimum distance between points
    width=100.0,
    height=100.0,
    seed=42
)
```

#### Hexagonal Packing

Efficient hexagonal grid for regular arrangements:

```python
from houdini_nodes import hexagonal_packing

points = hexagonal_packing(
    spacing=2.0,
    rows=10,
    cols=10,
    center_origin=True
)
```

#### Point Relaxation

Pushes overlapping points apart:

```python
from houdini_nodes import relax_points

relaxed = relax_points(
    points=initial_points,
    iterations=50,
    repulsion_radius=5.0,
    damping=0.5
)
```

### Terrain Operations

#### Feature-Based Scattering

```python
# Place trees only on flat, mid-elevation areas
trees = terrain.scatter_on_terrain(
    count=500,
    min_slope=0,
    max_slope=15,
    min_height=mean_height - 10,
    max_height=mean_height + 20
)

# Place rocks on steep slopes
rocks = terrain.scatter_on_terrain(
    count=200,
    min_slope=30,
    max_slope=90
)
```

#### Creating Masks

```python
# Create slope mask for steep areas
steep_mask = terrain.create_mask_by_feature(
    feature="slope",
    min_value=30,
    max_value=90
)

# Create height mask for high elevation
high_mask = terrain.create_mask_by_feature(
    feature="height",
    min_value=terrain.height_map.mean(),
    max_value=terrain.height_map.max()
)
```

### Scene Partitioning

```python
from houdini_nodes import ScenePartitioner, scatter_on_surface

# Generate points
points = scatter_on_surface(count=100)

# Create partitioner
partitioner = ScenePartitioner()

# Group by bounding box
center_area = partitioner.group_by_bounding_box(
    points,
    min_point=(-20, 0, -20),
    max_point=(20, 100, 20),
    group_name="center"
)

# Partition by condition
large, small = partitioner.partition(
    points,
    condition_func=lambda p: p.scale > 1.0
)

# Apply transformation (Attribute Wrangle style)
partitioner.apply_condition(
    points,
    condition_func=lambda p: p.attributes.get("type") == "tree",
    action_func=lambda p: setattr(p, "scale", p.scale * 1.5)
)
```

### Component System

```python
from houdini_nodes import ComponentBuilder

# Create reusable building component
building = ComponentBuilder("ProceduralBuilding")
building.add_parameter("width", 15.0, min_val=5.0, max_val=50.0)
building.add_parameter("height", 20.0, min_val=5.0, max_val=100.0)
building.add_parameter("floors", 5, min_val=1, max_val=20)
building.add_parameter("style", "modern")

def build_building(width, height, floors, style):
    return {
        "dimensions": (width, height),
        "floors": floors,
        "style": style
    }

building.set_build_function(build_building)

# Instantiate with parameters
office = building.build(width=20, height=40, floors=10, style="modern")
house = building.build(width=10, height=12, floors=2, style="classic")
```

### Variant System

```python
from houdini_nodes import VariantSet

# Create variant set for a building location
building_variants = VariantSet("Building_Variants")
building_variants.add_variant("modern", modern_mesh)
building_variants.add_variant("classic", classic_mesh)
building_variants.add_variant("industrial", industrial_mesh)

# Switch variants
building_variants.switch_variant("modern")
current = building_variants.get_active()
```

## Examples

### Forest Scene Generation

Complete workflow replicating Houdini's forest generation:

```python
from examples.forest_scene import generate_forest_scene

scene = generate_forest_scene(
    terrain_size=512,
    tree_count=500,
    rock_count=200,
    seed=42
)

# Access generated data
terrain = scene["terrain"]
trees = scene["trees"]
rocks = scene["rocks"]
```

**Houdini Equivalent:**
```
HeightField → Erode → Scatter@pts → Copy to Points → Merge
```

### City Scene Generation

Complete city generation with districts:

```python
from examples.city_scene import generate_city_scene

city = generate_city_scene(
    num_blocks_x=8,
    num_blocks_z=8,
    seed=123
)

# Access city data
blocks = city["blocks"]
buildings = city["buildings"]
streets = city["streets"]
```

**Houdini Equivalent:**
```
Grid → Connectivity → For-Each → Copy to Points → PolyExtrude → Merge
```

## API Reference

### houdini_nodes.py

| Function/Class | Houdini Equivalent | Description |
|----------------|-------------------|-------------|
| `scatter_on_surface()` | Scatter | Random point distribution |
| `poisson_disk_sampling()` | Labs Poisson Disk | Blue noise sampling |
| `hexagonal_packing()` | Hexagonal Packing | Hexagonal grid |
| `rectangular_packing()` | Grid | Rectangular grid |
| `relax_points()` | Relax | Point separation |
| `ScenePartitioner` | Group + Blast | Selection and filtering |
| `LSystem` | L-System | Fractal generation |
| `VariantSet` | Variant LOP | Asset variants |
| `ComponentBuilder` | Component Builder | Reusable components |

### procedural_terrain.py

| Method | Houdini Equivalent | Description |
|--------|-------------------|-------------|
| `generate_noise_terrain()` | HeightField | Noise-based terrain |
| `generate_fractal_terrain()` | HeightField | Diamond-square terrain |
| `thermal_erosion()` | HeightField Erode (Thermal) | Thermal weathering |
| `hydraulic_erosion()` | HeightField Erode (Hydraulic) | Water erosion |
| `scatter_on_terrain()` | HeightField Scatter | Terrain-based scattering |
| `create_mask_by_feature()` | HeightField Mask | Feature masks |
| `get_slope_at()` | - | Slope calculation |
| `get_height_at()` | - | Height sampling |

### city_generator.py

| Class/Function | Houdini Equivalent | Description |
|----------------|-------------------|-------------|
| `CityGenerator` | Labs City Generator | City layout |
| `BuildingGenerator` | Labs Building Generator | Procedural buildings |
| `StreetNetwork` | - | Road system |
| `generate_complete_city()` | - | Full city generation |

## Houdini to RoboCute Mapping

### Data Types

| Houdini | RoboCute Python |
|---------|-----------------|
| Points | `ScatterPoint` |
| Primitives | `BuildingData` |
| Attributes | `dict` in `attributes` |
| Groups | `ScenePartitioner.groups` |
| VEX | Python functions |

### Operations

| Houdini Operation | Python Equivalent |
|-------------------|-------------------|
| `@P` (position) | `point.position` |
| `@pscale` | `point.scale` |
| `@orient` | `point.rotation` |
| `rand(@ptnum)` | `np.random.uniform()` |
| `fit()` | Linear interpolation |
| `chramp()` | Custom curves |

## Best Practices

### 1. Seed Management

Always use seeds for reproducible results:

```python
np.random.seed(seed)
points = scatter_on_surface(seed=seed)
```

### 2. Performance

For large scenes, batch operations:

```python
# Good: Batch process
results = foreach_batch_process(items, process_func)

# Avoid: Individual processing in loop
for item in items:
    process_single(item)  # Slow for large datasets
```

### 3. Memory Management

Large terrain operations:

```python
# Process in chunks for very large terrains
chunk_size = 256
for y in range(0, height, chunk_size):
    for x in range(0, width, chunk_size):
        process_chunk(x, y, chunk_size)
```

### 4. Error Handling

Always validate inputs:

```python
def scatter_on_surface(bounds, count, **kwargs):
    if bounds[0] >= bounds[1]:
        raise ValueError("Invalid bounds: min >= max")
    if count < 0:
        raise ValueError("Count must be non-negative")
    # ...
```

## Troubleshooting

### Common Issues

**Issue:** Points clumping together
```python
# Solution: Use Poisson disk sampling instead of random
points = poisson_disk_sampling(radius=5.0, ...)
```

**Issue:** Terrain erosion looks unnatural
```python
# Solution: Add smoothing before erosion
terrain.smooth(iterations=2)
terrain.thermal_erosion(iterations=100)
```

**Issue:** City buildings overlapping
```python
# Solution: Use smaller density or larger blocks
buildings = building_gen.populate_block(
    block,
    density=0.5,  # Reduce from 0.8
    # ...
)
```

## Future Extensions

Potential additions to the system:

- **Voronoi Fracture**: Destruction geometry generation
- **VDB/Volumes**: Volumetric operations
- **Particles**: Particle system integration
- **Constraints**: Physics-based layout
- **HDA Integration**: Houdini Digital Asset compatibility

## References

### Houdini Documentation
- [Houdini SOPs](https://www.sidefx.com/docs/houdini/nodes/sop/)
- [HeightField Tools](https://www.sidefx.com/docs/houdini/terrain/)
- [Solaris/USD](https://www.sidefx.com/docs/houdini/solaris/)

### Related Papers
- Fast Poisson Disk Sampling
- Multi-scale Terrain Erosion
- Procedural City Generation (Mueller et al.)

## Conclusion

This project demonstrates how Houdini's procedural workflows can be adapted to game engines like RoboCute. By mapping familiar node concepts to Python classes and functions, artists can leverage their Houdini knowledge while benefiting from real-time rendering capabilities.

The key insight is that Houdini's node graphs represent data flow transformations, which translate naturally to Python function compositions. Each Houdini node becomes a function or class method, and the node graph becomes a Python script.

### Key Takeaways

1. **Node → Function**: Houdini nodes map directly to Python functions
2. **Geometry → Data**: Point/Primitive data becomes Python objects
3. **Attributes → Properties**: `@P`, `@pscale` become object attributes
4. **VEX → Python**: Expression logic becomes Python code
5. **Network → Script**: Node graphs become Python scripts

### Workflow Comparison

| Stage | Houdini | RoboCute Python |
|-------|---------|-----------------|
| Setup | Create nodes | Import modules |
| Configuration | Set parameters | Function arguments |
| Connection | Wire nodes | Function calls |
| Execution | Cook network | Run script |
| Output | Viewport/GDisk | Scene entities |

This implementation provides a foundation for procedural content generation in RoboCute, bringing the power of Houdini's workflows to real-time rendering.

---

**Version:** 1.0.0  
**License:** MIT (same as RoboCute)  
**Maintainer:** RoboCute Project Team
