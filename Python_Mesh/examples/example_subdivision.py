"""
Example: Mesh Subdivision (细分示例)

This example demonstrates mesh subdivision algorithms:
- Loop Subdivision (for triangle meshes)
- Midpoint Subdivision
- Catmull-Clark Subdivision (pure Python implementation)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.subdivision import (
    loop_subdivide,
    midpoint_subdivide,
    catmull_clark_subdivision,
    save_catmull_clark_mesh,
    check_manifold
)
from core.utils import create_test_mesh
import open3d as o3d


def example_loop_subdivision():
    """Loop 细分示例"""
    print("=" * 60)
    print("Example: Loop Subdivision")
    print("=" * 60)
    print("\nLoop subdivision increases face count by 4x per iteration")
    print("Note: Requires manifold mesh!\n")
    
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Create test mesh
    mesh = create_test_mesh('cube', size=2.0)
    input_path = output_dir / 'test_cube.ply'
    o3d.io.write_triangle_mesh(str(input_path), mesh)
    
    print(f"Original mesh: {len(mesh.triangles)} faces")
    
    # Check manifold before subdivision
    print("\nManifold check:")
    check_manifold(mesh)
    
    # Subdivide with different iterations
    for iterations in [1, 2]:
        print(f"\n--- Iteration {iterations} ---")
        output_path = output_dir / f'cube_loop_subdiv_{iterations}.ply'
        try:
            result = loop_subdivide(input_path, output_path, iterations=iterations)
            print(f"Saved to: {output_path}")
        except Exception as e:
            print(f"Error: {e}")


def example_midpoint_subdivision():
    """中点细分示例"""
    print("\n" + "=" * 60)
    print("Example: Midpoint Subdivision")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / 'output'
    
    mesh = create_test_mesh('sphere', size=2.0)
    input_path = output_dir / 'test_sphere_subdiv.ply'
    o3d.io.write_triangle_mesh(str(input_path), mesh)
    
    print(f"Original mesh: {len(mesh.triangles)} faces")
    
    output_path = output_dir / 'sphere_midpoint_subdiv.ply'
    result = midpoint_subdivide(input_path, output_path, iterations=1)
    print(f"Saved to: {output_path}")


def example_catmull_clark():
    """Catmull-Clark 细分示例"""
    print("\n" + "=" * 60)
    print("Example: Catmull-Clark Subdivision (Pure Python)")
    print("=" * 60)
    print("\nCatmull-Clark works with any polygon type")
    print("Output is always quads (4-sided polygons)\n")
    
    output_dir = Path(__file__).parent.parent / 'output'
    
    # Create a simple cube as quad mesh
    # Cube vertices
    points = [
        [-1, -1, -1],  # 0
        [ 1, -1, -1],  # 1
        [ 1,  1, -1],  # 2
        [-1,  1, -1],  # 3
        [-1, -1,  1],  # 4
        [ 1, -1,  1],  # 5
        [ 1,  1,  1],  # 6
        [-1,  1,  1],  # 7
    ]
    
    # Cube faces (quads)
    faces = [
        [0, 1, 2, 3],  # front
        [1, 5, 6, 2],  # right
        [5, 4, 7, 6],  # back
        [4, 0, 3, 7],  # left
        [3, 2, 6, 7],  # top
        [4, 5, 1, 0],  # bottom
    ]
    
    print(f"Input: {len(points)} vertices, {len(faces)} quads")
    
    # Subdivide
    print("\nSubdividing...")
    new_points, new_faces = catmull_clark_subdivision(points, faces, iterations=1)
    
    # Save result
    output_path = output_dir / 'cube_catmull_clark.obj'
    save_catmull_clark_mesh(new_points, new_faces, output_path)
    
    # Do another iteration
    print("\nSecond iteration...")
    new_points2, new_faces2 = catmull_clark_subdivision(new_points, new_faces, iterations=1)
    output_path2 = output_dir / 'cube_catmull_clark_2iter.obj'
    save_catmull_clark_mesh(new_points2, new_faces2, output_path2)


if __name__ == '__main__':
    example_loop_subdivision()
    example_midpoint_subdivision()
    example_catmull_clark()
    
    print("\n" + "=" * 60)
    print("Subdivision examples completed!")
    print("=" * 60)
