#!/usr/bin/env python3
"""
Demo Script for Python Mesh Processing Toolkit

This script demonstrates all features of the toolkit:
1. Creates sample meshes
2. Runs decimation, subdivision, repair
3. Generates LOD chain
4. Shows results summary
"""

import sys
from pathlib import Path

def check_dependencies():
    """检查依赖是否安装"""
    deps = {
        'open3d': False,
        'pymeshlab': False,
        'numpy': False,
    }
    
    try:
        import open3d
        deps['open3d'] = True
    except ImportError:
        pass
    
    try:
        import pymeshlab
        deps['pymeshlab'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        deps['numpy'] = True
    except ImportError:
        pass
    
    return deps


def print_banner():
    """打印欢迎横幅"""
    print("=" * 70)
    print("   Python 3D Mesh Processing Toolkit - Demo")
    print("=" * 70)
    print()


def print_deps_status(deps):
    """打印依赖状态"""
    print("Dependencies Status:")
    print("-" * 40)
    for name, available in deps.items():
        status = "✓ Available" if available else "✗ Not installed"
        print(f"  {name:<15} {status}")
    print()
    
    if not any(deps.values()):
        print("ERROR: No dependencies installed!")
        print("Please run: pip install open3d pymeshlab numpy")
        return False
    
    if not deps['open3d']:
        print("WARNING: Open3D is required for most demos.")
        print("Install with: pip install open3d")
        return False
    
    return True


def demo_create_meshes():
    """演示：创建测试网格"""
    print("-" * 70)
    print("DEMO 1: Creating Test Meshes")
    print("-" * 70)
    
    from core.utils import create_test_mesh
    import open3d as o3d
    
    output_dir = Path(__file__).parent / 'output' / 'demo'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mesh_types = ['cube', 'sphere', 'torus']
    
    for mesh_type in mesh_types:
        print(f"\nCreating {mesh_type}...")
        mesh = create_test_mesh(mesh_type, size=2.0)
        
        # Save original
        path = output_dir / f'{mesh_type}_original.ply'
        o3d.io.write_triangle_mesh(str(path), mesh)
        print(f"  Saved: {path.name} ({len(mesh.vertices)} vertices, {len(mesh.triangles)} faces)")
    
    print("\n✓ Test meshes created!")
    return output_dir


def demo_decimation(output_dir):
    """演示：减面"""
    print("\n" + "-" * 70)
    print("DEMO 2: Mesh Decimation (QEM Algorithm)")
    print("-" * 70)
    
    from core.decimation import decimate_mesh_open3d
    import open3d as o3d
    
    input_path = output_dir / 'sphere_original.ply'
    
    targets = [2000, 1000, 500]
    
    print(f"\nInput: {input_path.name}")
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    print(f"Original faces: {len(mesh.triangles)}")
    
    print("\nDecimation Results:")
    print(f"{'Target':<12} {'Actual':<12} {'Reduction':<12}")
    print("-" * 40)
    
    for target in targets:
        output_path = output_dir / f'sphere_decimated_{target}.ply'
        result = decimate_mesh_open3d(input_path, output_path, target_faces=target)
        actual = len(result.triangles)
        reduction = 1 - (actual / len(mesh.triangles))
        print(f"{target:<12} {actual:<12} {reduction:<12.1%}")
    
    print("\n✓ Decimation demo complete!")


def demo_subdivision(output_dir):
    """演示：细分"""
    print("\n" + "-" * 70)
    print("DEMO 3: Mesh Subdivision")
    print("-" * 70)
    
    from core.subdivision import loop_subdivide, check_manifold
    import open3d as o3d
    
    input_path = output_dir / 'cube_original.ply'
    
    print(f"\nInput: {input_path.name}")
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    print(f"Original faces: {len(mesh.triangles)}")
    
    # Check manifold
    print("\nManifold check:")
    check_manifold(mesh)
    
    # Subdivide
    print("\nLoop Subdivision:")
    for i in [1, 2]:
        output_path = output_dir / f'cube_subdiv_{i}.ply'
        result = loop_subdivide(input_path, output_path, iterations=i)
        print(f"  {i} iteration(s): {len(result.triangles)} faces (x{4**i})")
    
    print("\n✓ Subdivision demo complete!")


def demo_repair(output_dir):
    """演示：修复"""
    print("\n" + "-" * 70)
    print("DEMO 4: Mesh Diagnosis")
    print("-" * 70)
    
    from core.repair import diagnose_mesh
    
    # Diagnose healthy mesh
    print("\n1. Diagnosing healthy sphere:")
    sphere_path = output_dir / 'sphere_original.ply'
    diagnose_mesh(sphere_path)
    
    print("\n✓ Diagnosis demo complete!")


def demo_lod(output_dir):
    """演示：LOD 生成"""
    print("\n" + "-" * 70)
    print("DEMO 5: LOD Generation")
    print("-" * 70)
    
    from core.lod import LODGenerator, LODConfig
    from core.utils import create_test_mesh
    import open3d as o3d
    
    # Create high-poly mesh
    print("\nCreating high-poly input mesh...")
    mesh = create_test_mesh('torus', size=2.0)
    mesh = mesh.subdivide_loop(number_of_iterations=2)
    
    input_path = output_dir / 'lod_input.ply'
    o3d.io.write_triangle_mesh(str(input_path), mesh)
    print(f"Input: {len(mesh.triangles)} faces")
    
    # Generate LODs
    lod_dir = output_dir / 'lod_demo'
    configs = [
        LODConfig('LOD0_High', 5000, 0),
        LODConfig('LOD1_Medium', 2000, 0),
        LODConfig('LOD2_Low', 500, 0),
    ]
    
    print("\nGenerating LOD chain...")
    generator = LODGenerator(lod_dir)
    results = generator.generate(input_path, configs)
    
    print("\nLOD Report:")
    generator.print_report()
    
    print("\n✓ LOD demo complete!")


def demo_summary(output_dir):
    """演示总结"""
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    
    print("\nGenerated Files:")
    print("-" * 40)
    
    for file in sorted(output_dir.rglob('*.ply')):
        size_kb = file.stat().st_size / 1024
        rel_path = file.relative_to(output_dir)
        print(f"  {rel_path}: {size_kb:.1f} KB")
    
    print("\n" + "=" * 70)
    print("All demos completed successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


def main():
    print_banner()
    
    # Check dependencies
    deps = check_dependencies()
    if not print_deps_status(deps):
        return 1
    
    try:
        # Run demos
        output_dir = demo_create_meshes()
        
        if deps['open3d']:
            demo_decimation(output_dir)
            demo_subdivision(output_dir)
            demo_repair(output_dir)
            demo_lod(output_dir)
            demo_summary(output_dir)
        
        print("\n✓ Demo completed! Check the output directory for results.")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
