"""
Forest Scene Generation Example
===============================

Houdini Node Graph:
-------------------
HeightField (地形) → Erode (侵蚀) → Scatter@pts (散布点) → 
Copy to Points (复制树木) + Attribute Wrangle (坡度过滤)

This example demonstrates:
- HeightField terrain generation with erosion
- Feature-based scattering (slope, height constraints)
- Multi-layer distribution (trees on flat areas, rocks on slopes)
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from procedural_terrain import HeightFieldTerrain
from houdini_nodes import (
    ScatterPoint, ScenePartitioner,
    foreach_batch_process, ComponentBuilder
)


def generate_forest_scene(
    terrain_size: int = 512,
    tree_count: int = 500,
    rock_count: int = 200,
    seed: int = 42
) -> dict:
    """
    Generate a complete forest scene.
    
    This replicates the Houdini workflow:
    1. HeightField - Create base terrain
    2. Erode - Apply thermal erosion
    3. Scatter@pts - Place points based on terrain features
    4. Copy to Points - Assign assets to points
    
    Args:
        terrain_size: Size of terrain grid
        tree_count: Number of trees to place
        rock_count: Number of rocks to place
        seed: Random seed
        
    Returns:
        Dictionary with scene data
    """
    print("=" * 60)
    print("Forest Scene Generation")
    print("=" * 60)
    
    np.random.seed(seed)
    
    # =========================================================================
    # 1. HeightField (地形生成)
    # =========================================================================
    print("\n[1/4] Generating HeightField terrain...")
    
    terrain = HeightFieldTerrain(
        width=terrain_size,
        height=terrain_size,
        cell_size=0.5,
        seed=seed
    )
    
    # Generate fractal terrain for more natural look
    terrain.generate_fractal_terrain(roughness=0.5, initial_height=100.0)
    
    stats = terrain.get_statistics()
    print(f"  Terrain size: {stats['world_size']}")
    print(f"  Height range: {stats['min_height']:.2f} - {stats['max_height']:.2f}")
    
    # =========================================================================
    # 2. Erode (侵蚀模拟)
    # =========================================================================
    print("\n[2/4] Applying thermal erosion...")
    
    terrain.thermal_erosion(
        iterations=100,
        talus_angle=45.0,
        fraction=0.5
    )
    
    # Additional smoothing
    try:
        terrain.smooth(iterations=2, strength=0.3)
    except ImportError:
        print("  (scipy not available, skipping smooth)")
    
    eroded_stats = terrain.get_statistics()
    print(f"  Height range after erosion: {eroded_stats['min_height']:.2f} - {eroded_stats['max_height']:.2f}")
    
    # =========================================================================
    # 3. Scatter@pts (基于地形特征散布)
    # =========================================================================
    print("\n[3/4] Scattering objects on terrain...")
    
    # Create masks for different regions
    slope_mask = terrain.create_mask_by_feature(
        feature="slope",
        min_value=0,
        max_value=20
    )
    
    # Place trees on flat areas (low slope, medium height)
    print("  Placing trees on flat areas...")
    tree_result = terrain.scatter_on_terrain(
        count=tree_count,
        min_slope=0,
        max_slope=20,       # Flat to gentle slopes
        min_height=stats['mean_height'] - 10,
        max_height=stats['mean_height'] + 20,
        seed=seed
    )
    
    print(f"  Placed {tree_result.placed_count}/{tree_count} trees")
    print(f"  Attempts: {tree_result.attempts}")
    
    # Place rocks on steep areas
    print("  Placing rocks on steep slopes...")
    rock_result = terrain.scatter_on_terrain(
        count=rock_count,
        min_slope=30,       # Steep slopes
        max_slope=90,
        min_height=0,
        max_height=float('inf'),
        seed=seed + 1
    )
    
    print(f"  Placed {rock_result.placed_count}/{rock_count} rocks")
    
    # =========================================================================
    # 4. Copy to Points + Attribute Wrangle (处理分布)
    # =========================================================================
    print("\n[4/4] Processing scattered objects...")
    
    partitioner = ScenePartitioner()
    
    # Convert to ScatterPoint objects with attributes
    tree_points = []
    for i, pos in enumerate(tree_result.points):
        # Calculate slope for each tree
        slope = terrain.get_slope_at(pos[0], pos[1])
        normal = terrain.get_normal_at(pos[0], pos[1])
        
        point = ScatterPoint(
            position=pos,
            scale=np.random.uniform(0.7, 1.3),
            rotation=(0, 1, 0, np.random.uniform(0, 2 * np.pi)),
            attributes={
                "type": "tree",
                "id": i,
                "slope": slope,
                "normal": normal
            }
        )
        tree_points.append(point)
    
    rock_points = []
    for i, pos in enumerate(rock_result.points):
        slope = terrain.get_slope_at(pos[0], pos[1])
        
        point = ScatterPoint(
            position=pos,
            scale=np.random.uniform(0.5, 1.5),
            rotation=(0, 1, 0, np.random.uniform(0, 2 * np.pi)),
            attributes={
                "type": "rock",
                "id": i,
                "slope": slope
            }
        )
        rock_points.append(point)
    
    # Group by bounding boxes (create forest clearings)
    print("  Creating forest clearings...")
    center_clearing = partitioner.group_by_bounding_box(
        tree_points,
        min_point=(terrain_size * 0.3, 0, terrain_size * 0.3),
        max_point=(terrain_size * 0.7, 100, terrain_size * 0.7),
        group_name="clearing"
    )
    
    print(f"  Trees in clearing: {len(center_clearing)}")
    
    # Filter trees by slope using Attribute Wrangle style
    steep_trees = partitioner.filter_by_attribute(
        tree_points, "slope", 15, 20
    )
    print(f"  Trees on moderate slopes: {len(steep_trees)}")
    
    # Create ComponentBuilder for tree variants
    tree_component = ComponentBuilder("ProceduralTree")
    tree_component.add_parameter("height", 10.0, 5.0, 20.0)
    tree_component.add_parameter("type", "oak")
    tree_component.add_parameter("seed", 0)
    
    def build_tree(height, type, seed):
        return {
            "height": height,
            "type": type,
            "seed": seed,
            "foliage_density": np.random.uniform(0.5, 1.0)
        }
    
    tree_component.set_build_function(build_tree)
    
    # Generate tree variants
    tree_variants = []
    for point in tree_points[:20]:  # Just first 20 for demo
        variant = tree_component.build(
            height=np.random.uniform(8, 15),
            type=np.random.choice(["oak", "pine", "birch"]),
            seed=point.attributes["id"]
        )
        variant["position"] = point.position
        tree_variants.append(variant)
    
    print(f"  Generated {len(tree_variants)} tree variants")
    
    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("Scene Generation Complete!")
    print("=" * 60)
    
    scene_data = {
        "terrain": terrain,
        "trees": {
            "points": tree_points,
            "count": len(tree_points),
            "clearing": center_clearing
        },
        "rocks": {
            "points": rock_points,
            "count": len(rock_points)
        },
        "statistics": {
            "total_objects": len(tree_points) + len(rock_points),
            "terrain_stats": eroded_stats
        }
    }
    
    return scene_data


def export_scene_data(scene_data: dict, output_dir: str = "./output"):
    """
    Export scene data to files.
    
    Args:
        scene_data: Scene data dictionary
        output_dir: Output directory
    """
    import json
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Export terrain heightmap
    terrain = scene_data["terrain"]
    terrain.export_heightmap(f"{output_dir}/forest_terrain.png")
    
    # Export object positions
    objects_data = []
    
    for point in scene_data["trees"]["points"]:
        objects_data.append({
            "type": "tree",
            "position": point.position,
            "scale": point.scale,
            "rotation": point.rotation,
            "attributes": {k: v for k, v in point.attributes.items() if k != "normal"}
        })
    
    for point in scene_data["rocks"]["points"]:
        objects_data.append({
            "type": "rock",
            "position": point.position,
            "scale": point.scale,
            "rotation": point.rotation,
            "attributes": point.attributes
        })
    
    with open(f"{output_dir}/forest_objects.json", "w") as f:
        json.dump(objects_data, f, indent=2)
    
    print(f"\nScene exported to {output_dir}/")
    print(f"  - forest_terrain.png")
    print(f"  - forest_objects.json")


def main():
    """Main execution."""
    # Generate scene
    scene = generate_forest_scene(
        terrain_size=256,
        tree_count=200,
        rock_count=50,
        seed=42
    )
    
    # Print summary
    print("\nScene Summary:")
    print(f"  Total objects: {scene['statistics']['total_objects']}")
    print(f"  Trees: {scene['trees']['count']}")
    print(f"  Rocks: {scene['rocks']['count']}")
    
    # Export (optional)
    try:
        export_scene_data(scene)
    except ImportError:
        print("\n(imageio not available, skipping export)")
    
    return scene


if __name__ == "__main__":
    main()
