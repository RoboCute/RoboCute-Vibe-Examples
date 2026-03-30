"""
Example: Mesh Decimation (减面示例)

This example demonstrates how to use the decimation module
to reduce mesh complexity using QEM algorithm.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.decimation import (
    decimate_mesh_pymeshlab,
    decimate_mesh_open3d,
    decimate_mesh_percentage
)
from core.utils import create_test_mesh, compare_meshes


def example_basic_decimation():
    """基本减面示例"""
    print("=" * 60)
    print("Example: Basic Mesh Decimation")
    print("=" * 60)
    
    # Create a test mesh
    print("\n1. Creating test sphere mesh...")
    mesh = create_test_mesh('sphere', size=2.0)
    
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    input_path = output_dir / 'test_sphere.ply'
    import open3d as o3d
    o3d.io.write_triangle_mesh(str(input_path), mesh)
    
    original_faces = len(mesh.triangles)
    print(f"   Original: {len(mesh.vertices)} vertices, {original_faces} faces")
    
    # Method 1: PyMeshLab (recommended for production)
    print("\n2. Decimating with PyMeshLab (QEM)...")
    try:
        output_path1 = output_dir / 'sphere_decimated_pymeshlab.ply'
        result1 = decimate_mesh_pymeshlab(
            input_path,
            output_path1,
            target_faces=500,
            preservenormal=True,
            preservetopology=True
        )
        print(f"   Saved to: {output_path1}")
    except ImportError as e:
        print(f"   Skipped: {e}")
    
    # Method 2: Open3D (lightweight)
    print("\n3. Decimating with Open3D...")
    try:
        output_path2 = output_dir / 'sphere_decimated_open3d.ply'
        result2 = decimate_mesh_open3d(
            input_path,
            output_path2,
            target_faces=500
        )
        print(f"   Saved to: {output_path2}")
    except ImportError as e:
        print(f"   Skipped: {e}")
    
    # Method 3: Percentage-based
    print("\n4. Decimating by percentage (25%)...")
    try:
        output_path3 = output_dir / 'sphere_decimated_25percent.ply'
        result3 = decimate_mesh_percentage(
            input_path,
            output_path3,
            target_percentage=0.25,
            method='open3d'
        )
        print(f"   Saved to: {output_path3}")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("Decimation examples completed!")
    print(f"Check output directory: {output_dir}")
    print("=" * 60)


def example_compare_methods():
    """比较不同减面方法"""
    print("\n" + "=" * 60)
    print("Example: Comparing Decimation Methods")
    print("=" * 60)
    
    mesh = create_test_mesh('torus', size=2.0)
    output_dir = Path(__file__).parent.parent / 'output'
    
    input_path = output_dir / 'test_torus.ply'
    import open3d as o3d
    o3d.io.write_triangle_mesh(str(input_path), mesh)
    
    # Decimate with different target face counts
    targets = [2000, 1000, 500, 200]
    
    print(f"\nOriginal: {len(mesh.triangles)} faces")
    print("\nOpen3D QEM Results:")
    print("-" * 40)
    print(f"{'Target':<12} {'Result':<12} {'Ratio':<10}")
    print("-" * 40)
    
    for target in targets:
        output_path = output_dir / f'torus_decimated_{target}.ply'
        try:
            result = decimate_mesh_open3d(input_path, output_path, target_faces=target)
            actual = len(result.triangles)
            ratio = actual / len(mesh.triangles)
            print(f"{target:<12} {actual:<12} {ratio:<10.2%}")
        except Exception as e:
            print(f"{target:<12} Error: {e}")
    
    print("-" * 40)


if __name__ == '__main__':
    example_basic_decimation()
    example_compare_methods()
