"""
Scatter Example - Basic to Advanced
====================================

This example demonstrates various scattering techniques:
- Basic random scatter
- Poisson disk sampling (blue noise)
- Hexagonal packing
- Grid distribution with variations
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from houdini_nodes import (
    scatter_on_surface,
    poisson_disk_sampling,
    hexagonal_packing,
    rectangular_packing,
    relax_points,
    create_grid_instances_data,
    relax_points,
    ScatterPoint,
    ScenePartitioner
)


def example_basic_scatter():
    """
    Example 1: Basic Scatter
    
    Houdini: Scatter SOP - Random point distribution
    """
    print("=" * 60)
    print("Example 1: Basic Scatter")
    print("=" * 60)
    
    # Simple random scatter in 100x100 area
    points = scatter_on_surface(
        bounds=(-50, 50),
        count=50,
        y_height=0.0,
        scale_range=(0.5, 1.5),
        seed=42
    )
    
    print(f"\nGenerated {len(points)} random points")
    print(f"Bounds: (-50, 50) on X and Z axes")
    
    # Show first 5 points
    print("\nFirst 5 points:")
    for i, p in enumerate(points[:5]):
        print(f"  Point {i}: pos={p.position}, scale={p.scale:.2f}")
    
    return points


def example_poisson_disk():
    """
    Example 2: Poisson Disk Sampling
    
    Houdini: Labs Poisson Disk - Blue noise distribution
    
    Blue noise ensures minimum distance between points,
    creating natural, non-overlapping distributions.
    """
    print("\n" + "=" * 60)
    print("Example 2: Poisson Disk Sampling (Blue Noise)")
    print("=" * 60)
    
    # Generate points with minimum radius of 5 units
    radius = 5.0
    points = poisson_disk_sampling(
        radius=radius,
        width=100.0,
        height=100.0,
        max_attempts=30,
        seed=42
    )
    
    print(f"\nGenerated {len(points)} points")
    print(f"Minimum radius: {radius}")
    print(f"Coverage: 100x100 area")
    print(f"Density: {len(points) / 100:.2f} points per unit length")
    
    # Verify minimum distance
    min_dist = float('inf')
    for i, p1 in enumerate(points):
        for p2 in points[i+1:]:
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            min_dist = min(min_dist, dist)
    
    print(f"Actual minimum distance: {min_dist:.2f} (should be >= {radius})")
    
    # Show first 5 points
    print("\nFirst 5 points:")
    for i, p in enumerate(points[:5]):
        print(f"  Point {i}: ({p[0]:.2f}, {p[1]:.2f})")
    
    return points


def example_hexagonal_packing():
    """
    Example 3: Hexagonal Packing
    
    Houdini: Hexagonal Packing SOP
    
    Creates tightly packed hexagonal grid, useful for:
    - Honeycomb structures
    - Efficient circular packing
    - Organic pattern bases
    """
    print("\n" + "=" * 60)
    print("Example 3: Hexagonal Packing")
    print("=" * 60)
    
    points = hexagonal_packing(
        spacing=5.0,
        rows=8,
        cols=8,
        center_origin=True
    )
    
    print(f"\nGenerated {len(points)} points in hexagonal grid")
    print(f"Spacing: 5.0 units")
    print(f"Grid: 8x8")
    
    # Calculate actual spacing
    if len(points) > 1:
        p0 = np.array(points[0])
        p1 = np.array(points[1])
        actual_spacing = np.linalg.norm(p1 - p0)
        print(f"Actual horizontal spacing: {actual_spacing:.2f}")
    
    # Show sample points
    print("\nSample points (corner, edge, center):")
    print(f"  Corner: {points[0]}")
    print(f"  Edge: {points[7]}")
    print(f"  Center-ish: {points[32] if len(points) > 32 else points[len(points)//2]}")
    
    return points


def example_grid_variations():
    """
    Example 4: Grid with Variations
    
    Houdini: Copy and Transform + Random
    
    Creates grid instances with random scale and rotation.
    """
    print("\n" + "=" * 60)
    print("Example 4: Grid with Variations (Copy and Transform)")
    print("=" * 60)
    
    points = create_grid_instances_data(
        rows=5,
        cols=5,
        spacing=4.0,
        scale_range=(0.8, 1.2)
    )
    
    print(f"\nGenerated {len(points)} grid points")
    print(f"Grid: 5x5 with 4.0 unit spacing")
    
    # Show distribution of scales
    scales = [p.scale for p in points]
    print(f"\nScale statistics:")
    print(f"  Min: {min(scales):.2f}")
    print(f"  Max: {max(scales):.2f}")
    print(f"  Mean: {np.mean(scales):.2f}")
    
    # Show first few points
    print("\nFirst 5 points:")
    for i, p in enumerate(points[:5]):
        print(f"  Grid[{p.attributes['row']},{p.attributes['col']}]: "
              f"pos=({p.position[0]:.1f}, {p.position[2]:.1f}), "
              f"scale={p.scale:.2f}")
    
    return points


def example_relaxation():
    """
    Example 5: Point Relaxation
    
    Houdini: Relax SOP
    
    Pushes points apart to avoid overlaps.
    """
    print("\n" + "=" * 60)
    print("Example 5: Point Relaxation")
    print("=" * 60)
    
    # Start with random points that may overlap
    np.random.seed(42)
    initial_points = [
        (np.random.uniform(0, 50), np.random.uniform(0, 50))
        for _ in range(30)
    ]
    
    print(f"\nInitial points: {len(initial_points)}")
    
    # Calculate initial minimum distance
    min_dist_initial = float('inf')
    for i, p1 in enumerate(initial_points):
        for p2 in initial_points[i+1:]:
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            min_dist_initial = min(min_dist_initial, dist)
    
    print(f"Initial minimum distance: {min_dist_initial:.2f}")
    
    # Relax points
    relaxed_points = relax_points(
        points=initial_points,
        iterations=50,
        repulsion_radius=5.0,
        damping=0.5
    )
    
    print(f"Relaxed points: {len(relaxed_points)}")
    
    # Calculate relaxed minimum distance
    min_dist_relaxed = float('inf')
    for i, p1 in enumerate(relaxed_points):
        for p2 in relaxed_points[i+1:]:
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            min_dist_relaxed = min(min_dist_relaxed, dist)
    
    print(f"Relaxed minimum distance: {min_dist_relaxed:.2f}")
    print(f"Improvement: {min_dist_relaxed - min_dist_initial:.2f}")
    
    return relaxed_points


def example_partitioning():
    """
    Example 6: Scene Partitioning
    
    Houdini: Group SOP + Blast SOP
    
    Creates selection groups based on conditions.
    """
    print("\n" + "=" * 60)
    print("Example 6: Scene Partitioning (Group + Blast)")
    print("=" * 60)
    
    # Generate some points
    points = scatter_on_surface(
        bounds=(-50, 50),
        count=100,
        seed=42
    )
    
    print(f"\nTotal points: {len(points)}")
    
    # Create partitioner
    partitioner = ScenePartitioner()
    
    # Group by bounding box (center region)
    center_group = partitioner.group_by_bounding_box(
        points,
        min_point=(-20, 0, -20),
        max_point=(20, 100, 20),
        group_name="center"
    )
    
    print(f"Points in center region: {len(center_group)}")
    
    # Partition by scale (large vs small)
    large_points, small_points = partitioner.partition(
        points,
        condition_func=lambda p: p.scale > 1.0
    )
    
    print(f"Large points (scale > 1.0): {len(large_points)}")
    print(f"Small points (scale <= 1.0): {len(small_points)}")
    
    # Apply condition (Attribute Wrangle style)
    print("\nApplying condition to double large point scales...")
    
    def double_scale(p):
        p.scale *= 2.0
        p.attributes["doubled"] = True
    
    partitioner.apply_condition(
        large_points,
        condition_func=lambda p: True,  # Apply to all
        action_func=double_scale
    )
    
    print(f"Updated {len(large_points)} large points")
    print(f"New scale range: {min(p.scale for p in large_points):.2f} - "
          f"{max(p.scale for p in large_points):.2f}")
    
    return partitioner.groups


def example_combined_techniques():
    """
    Example 7: Combined Techniques
    
    Demonstrates combining multiple scattering techniques
    for a complex distribution.
    """
    print("\n" + "=" * 60)
    print("Example 7: Combined Techniques")
    print("=" * 60)
    
    print("\nCreating layered distribution...")
    
    # Layer 1: Base grid for structures
    print("  Layer 1: Grid distribution for buildings")
    grid_points = create_grid_instances_data(
        rows=6,
        cols=6,
        spacing=10.0,
        scale_range=(0.9, 1.1)
    )
    
    # Layer 2: Poisson disk for trees (natural distribution)
    print("  Layer 2: Poisson disk for trees")
    tree_points_2d = poisson_disk_sampling(
        radius=3.0,
        width=60.0,
        height=60.0,
        seed=42
    )
    
    # Center the tree points
    tree_points_2d = [(x - 30, z - 30) for x, z in tree_points_2d]
    
    # Layer 3: Hexagonal packing for decorative elements
    print("  Layer 3: Hexagonal packing for decorations")
    decor_points = hexagonal_packing(
        spacing=4.0,
        rows=4,
        cols=4,
        center_origin=True
    )
    
    # Filter out points too close to grid points (buildings)
    print("  Filtering: Removing trees near buildings...")
    
    filtered_trees = []
    for tx, tz in tree_points_2d:
        too_close = False
        for grid_p in grid_points:
            gx, gz = grid_p.position[0], grid_p.position[2]
            dist = np.sqrt((tx - gx)**2 + (tz - gz)**2)
            if dist < 5.0:  # Keep trees 5 units away from buildings
                too_close = True
                break
        
        if not too_close:
            filtered_trees.append((tx, tz))
    
    print(f"\nResults:")
    print(f"  Grid points (buildings): {len(grid_points)}")
    print(f"  Original trees: {len(tree_points_2d)}")
    print(f"  Filtered trees: {len(filtered_trees)}")
    print(f"  Decor points: {len(decor_points)}")
    
    total = len(grid_points) + len(filtered_trees) + len(decor_points)
    print(f"  Total objects: {total}")
    
    return {
        "buildings": grid_points,
        "trees": filtered_trees,
        "decorations": decor_points
    }


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Scatter Examples - Houdini to RoboCute")
    print("=" * 60)
    
    # Run all examples
    example_basic_scatter()
    example_poisson_disk()
    example_hexagonal_packing()
    example_grid_variations()
    example_relaxation()
    example_partitioning()
    example_combined_techniques()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
