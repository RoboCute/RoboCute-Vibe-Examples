"""
Example: Complete Mesh Processing Workflow
完整网格处理工作流示例

This example demonstrates a complete workflow:
1. Load mesh
2. Diagnose issues
3. Repair topology
4. Generate LOD chain
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.repair import diagnose_mesh, repair_topology_pymeshlab
from core.lod import LODGenerator, LODConfig
from core.utils import create_test_mesh, compare_meshes
import open3d as o3d


def complete_workflow():
    """完整工作流示例"""
    print("=" * 70)
    print("Complete Mesh Processing Workflow")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent / 'output' / 'complete_workflow'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create input mesh
    print("\n" + "-" * 70)
    print("STEP 1: Create Input Mesh")
    print("-" * 70)
    
    mesh = create_test_mesh('sphere', size=2.0)
    # Make it high-poly
    mesh = mesh.subdivide_loop(number_of_iterations=3)
    
    raw_path = output_dir / '01_raw_input.ply'
    o3d.io.write_triangle_mesh(str(raw_path), mesh)
    print(f"Created mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")
    
    # Step 2: Diagnose
    print("\n" + "-" * 70)
    print("STEP 2: Diagnose Mesh")
    print("-" * 70)
    diagnose_mesh(raw_path)
    
    # Step 3: Repair (if needed)
    print("\n" + "-" * 70)
    print("STEP 3: Repair Topology")
    print("-" * 70)
    
    repaired_path = output_dir / '02_repaired.ply'
    try:
        repair_topology_pymeshlab(
            raw_path,
            repaired_path,
            close_holes=True,
            remove_isolated=True
        )
        print(f"Repaired mesh saved to: {repaired_path}")
    except ImportError:
        print("PyMeshLab not available, using raw mesh")
        repaired_path = raw_path
    
    # Step 4: Generate LODs
    print("\n" + "-" * 70)
    print("STEP 4: Generate LOD Chain")
    print("-" * 70)
    
    lod_configs = [
        LODConfig('LOD0_High', 10000, 0),
        LODConfig('LOD1_Medium', 3000, 0),
        LODConfig('LOD2_Low', 1000, 0),
    ]
    
    lod_dir = output_dir / 'LOD'
    generator = LODGenerator(lod_dir)
    results = generator.generate(repaired_path, lod_configs)
    
    # Step 5: Final report
    print("\n" + "-" * 70)
    print("STEP 5: Final Report")
    print("-" * 70)
    
    print("\nGenerated files:")
    for file in sorted(output_dir.rglob('*')):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"  {file.relative_to(output_dir)}: {size_kb:.1f} KB")
    
    print("\nLOD Statistics:")
    generator.print_report()
    
    print("\n" + "=" * 70)
    print("Workflow Complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


def batch_processing_example():
    """批处理示例"""
    print("\n" + "=" * 70)
    print("Batch Processing Example")
    print("=" * 70)
    
    output_dir = Path(__file__).parent.parent / 'output' / 'batch'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple test meshes
    mesh_types = ['cube', 'sphere', 'torus']
    
    print("\nCreating test meshes...")
    for mesh_type in mesh_types:
        mesh = create_test_mesh(mesh_type, size=2.0)
        # Make them high-poly
        mesh = mesh.subdivide_loop(number_of_iterations=2)
        
        path = output_dir / f'{mesh_type}_input.ply'
        o3d.io.write_triangle_mesh(str(path), mesh)
        print(f"  {mesh_type}: {len(mesh.triangles)} faces")
    
    print("\nProcessing each mesh...")
    for mesh_type in mesh_types:
        print(f"\n--- Processing {mesh_type} ---")
        input_path = output_dir / f'{mesh_type}_input.ply'
        
        # Repair
        repaired_path = output_dir / f'{mesh_type}_repaired.ply'
        try:
            repair_topology_pymeshlab(input_path, repaired_path)
        except ImportError:
            repaired_path = input_path
        
        # Generate LOD
        lod_dir = output_dir / f'{mesh_type}_lod'
        configs = [
            LODConfig('High', 5000, 0),
            LODConfig('Low', 1000, 0),
        ]
        
        generator = LODGenerator(lod_dir)
        generator.generate(repaired_path, configs)
    
    print("\n" + "=" * 70)
    print("Batch processing complete!")
    print("=" * 70)


if __name__ == '__main__':
    complete_workflow()
    batch_processing_example()
