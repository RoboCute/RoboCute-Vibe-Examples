"""
Example: Mesh Topology Repair (拓扑修复示例)

This example demonstrates how to repair common mesh issues:
- Duplicate vertices
- Unreferenced vertices
- Holes
- Non-manifold edges
- Inconsistent normals
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.repair import (
    repair_topology_pymeshlab,
    diagnose_mesh,
    laplacian_smooth,
    fix_winding_order
)
from core.utils import create_test_mesh
import open3d as o3d
import numpy as np


def example_diagnose():
    """网格诊断示例"""
    print("=" * 60)
    print("Example: Mesh Diagnosis")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    
    # Create test meshes with different properties
    print("\n1. Creating test meshes...")
    
    # Healthy mesh (sphere)
    sphere = create_test_mesh('sphere', size=2.0)
    sphere_path = output_dir / 'healthy_sphere.ply'
    o3d.io.write_triangle_mesh(str(sphere_path), sphere)
    
    # Mesh with holes (remove some triangles)
    sphere_with_holes = create_test_mesh('sphere', size=2.0)
    triangles = np.asarray(sphere_with_holes.triangles)
    # Remove first 100 triangles to create holes
    sphere_with_holes.triangles = o3d.utility.Vector3iVector(triangles[100:])
    holes_path = output_dir / 'sphere_with_holes.ply'
    o3d.io.write_triangle_mesh(str(holes_path), sphere_with_holes)
    
    print("\n2. Diagnosing healthy sphere...")
    diagnose_mesh(sphere_path)
    
    print("\n3. Diagnosing sphere with holes...")
    diagnose_mesh(holes_path)


def example_repair_workflow():
    """完整修复工作流示例"""
    print("\n" + "=" * 60)
    print("Example: Complete Repair Workflow")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / 'output'
    
    # Create a problematic mesh
    print("\n1. Creating mesh with issues...")
    mesh = create_test_mesh('torus', size=2.0)
    
    # Add duplicate vertices by merging two meshes
    mesh2 = create_test_mesh('torus', size=1.0)
    vertices = np.vstack([
        np.asarray(mesh.vertices),
        np.asarray(mesh2.vertices) + [3, 0, 0]
    ])
    triangles = np.vstack([
        np.asarray(mesh.triangles),
        np.asarray(mesh2.triangles) + len(mesh.vertices)
    ])
    
    problem_mesh = o3d.geometry.TriangleMesh()
    problem_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    problem_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    input_path = output_dir / 'problem_mesh.ply'
    o3d.io.write_triangle_mesh(str(input_path), problem_mesh)
    
    print(f"   Input: {len(problem_mesh.vertices)} vertices")
    
    # Diagnose before
    print("\n2. Before repair:")
    diagnose_mesh(input_path)
    
    # Repair
    print("\n3. Repairing...")
    output_path = output_dir / 'repaired_mesh.ply'
    try:
        result = repair_topology_pymeshlab(
            input_path,
            output_path,
            remove_duplicates=True,
            remove_unreferenced=True,
            close_holes=True,
            remove_isolated=True
        )
        print(f"   Saved to: {output_path}")
    except ImportError as e:
        print(f"   Skipped: {e}")
        return
    
    # Diagnose after
    print("\n4. After repair:")
    diagnose_mesh(output_path)


def example_smoothing():
    """平滑处理示例"""
    print("\n" + "=" * 60)
    print("Example: Laplacian Smoothing")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / 'output'
    
    # Create a noisy mesh by perturbing vertices
    print("\n1. Creating noisy mesh...")
    mesh = create_test_mesh('sphere', size=2.0)
    vertices = np.asarray(mesh.vertices)
    noise = np.random.normal(0, 0.05, vertices.shape)
    noisy_vertices = vertices + noise
    mesh.vertices = o3d.utility.Vector3dVector(noisy_vertices)
    
    input_path = output_dir / 'noisy_sphere.ply'
    o3d.io.write_triangle_mesh(str(input_path), mesh)
    print(f"   Saved noisy mesh to: {input_path}")
    
    # Smooth
    print("\n2. Applying Laplacian smoothing...")
    output_path = output_dir / 'smoothed_sphere.ply'
    try:
        result = laplacian_smooth(input_path, output_path, steps=5)
        print(f"   Saved to: {output_path}")
    except ImportError as e:
        print(f"   Skipped: {e}")


if __name__ == '__main__':
    example_diagnose()
    example_repair_workflow()
    example_smoothing()
    
    print("\n" + "=" * 60)
    print("Repair examples completed!")
    print("=" * 60)
