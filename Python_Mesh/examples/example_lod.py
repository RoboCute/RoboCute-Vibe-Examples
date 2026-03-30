"""
Example: LOD Generation (LOD 生成示例)

This example demonstrates generating Level-of-Detail (LOD) chains
for game assets or real-time rendering.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.lod import (
    LODGenerator,
    LODConfig,
    generate_lod_chain,
    generate_preset_lod,
    LOD_PRESETS
)
from core.utils import create_test_mesh
import open3d as o3d


def example_basic_lod():
    """基本 LOD 生成示例"""
    print("=" * 60)
    print("Example: Basic LOD Generation")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / 'output' / 'lod_basic'
    
    # Create test mesh
    print("\n1. Creating test mesh (high-poly sphere)...")
    mesh = create_test_mesh('sphere', size=2.0)
    
    # Subdivide to make it high-poly
    mesh = mesh.subdivide_loop(number_of_iterations=2)
    
    input_path = output_dir / 'input_highpoly.ply'
    o3d.io.write_triangle_mesh(str(input_path), mesh)
    print(f"   Input: {len(mesh.triangles)} faces")
    
    # Define LOD configs
    configs = [
        LODConfig('LOD0_High', 10000, 0),
        LODConfig('LOD1_Medium', 3000, 0),
        LODConfig('LOD2_Low', 1000, 0),
        LODConfig('LOD3_UltraLow', 300, 0),
    ]
    
    print("\n2. Generating LOD chain...")
    generator = LODGenerator(output_dir)
    results = generator.generate(input_path, configs)
    
    print("\n3. Results:")
    generator.print_report()


def example_preset_lod():
    """使用预设配置生成 LOD"""
    print("\n" + "=" * 60)
    print("Example: Using LOD Presets")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / 'output'
    
    # Create test mesh
    mesh = create_test_mesh('cube', size=2.0)
    mesh = mesh.subdivide_loop(number_of_iterations=3)
    
    input_path = output_dir / 'test_cube_lod.ply'
    o3d.io.write_triangle_mesh(str(input_path), mesh)
    
    print(f"\nInput mesh: {len(mesh.triangles)} faces")
    print("\nAvailable presets:")
    for name in LOD_PRESETS.keys():
        print(f"  - {name}")
    
    # Generate with different presets
    for preset in ['game_prop', 'mobile']:
        print(f"\n--- Using preset: {preset} ---")
        preset_output_dir = output_dir / f'lod_{preset}'
        try:
            results = generate_preset_lod(input_path, preset_output_dir, preset=preset)
        except Exception as e:
            print(f"Error: {e}")


def example_custom_lod():
    """自定义 LOD 生成"""
    print("\n" + "=" * 60)
    print("Example: Custom LOD Configuration")
    print("=" * 60)
    
    output_dir = Path(__file__).parent.parent / 'output' / 'lod_custom'
    
    # Create test mesh
    mesh = create_test_mesh('torus', size=2.0)
    mesh = mesh.subdivide_loop(number_of_iterations=2)
    
    input_path = output_dir / 'input.ply'
    o3d.io.write_triangle_mesh(str(input_path), mesh)
    
    print(f"\nInput mesh: {len(mesh.triangles)} faces")
    
    # Custom configs for VR application
    vr_configs = [
        ('VR_Close', 20000, 0),     # Close up
        ('VR_Mid', 8000, 0),        # Mid distance
        ('VR_Far', 2000, 0),        # Far away
    ]
    
    print("\nGenerating VR-optimized LODs...")
    results = generate_lod_chain(input_path, output_dir, vr_configs)
    
    print(f"\nGenerated {len(results)} LOD levels")


if __name__ == '__main__':
    example_basic_lod()
    example_preset_lod()
    example_custom_lod()
    
    print("\n" + "=" * 60)
    print("LOD examples completed!")
    print("=" * 60)
