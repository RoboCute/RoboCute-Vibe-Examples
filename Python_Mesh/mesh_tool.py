#!/usr/bin/env python3
"""
Mesh Processing CLI Tool

A command-line interface for the Python Mesh Processing toolkit.
提供减面、细分、修复和 LOD 生成功能的命令行工具。

Usage:
    python mesh_tool.py decimate <input> <output> --target-faces 1000
    python mesh_tool.py subdivide <input> <output> --iterations 2
    python mesh_tool.py repair <input> <output>
    python mesh_tool.py diagnose <input>
    python mesh_tool.py lod <input> --output-dir ./lods
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.decimation import decimate_mesh_open3d, decimate_mesh_pymeshlab
from core.subdivision import loop_subdivide, midpoint_subdivide
from core.repair import repair_topology_pymeshlab, diagnose_mesh
from core.lod import LODGenerator, LODConfig


def cmd_decimate(args):
    """减面命令"""
    print(f"Decimating: {args.input} -> {args.output}")
    print(f"Target faces: {args.target_faces}")
    
    try:
        if args.method == 'pymeshlab':
            result = decimate_mesh_pymeshlab(
                args.input,
                args.output,
                target_faces=args.target_faces,
                preservenormal=not args.no_preserve_normal,
                preservetopology=not args.no_preserve_topology
            )
        else:
            result = decimate_mesh_open3d(
                args.input,
                args.output,
                target_faces=args.target_faces
            )
        print(f"✓ Success! Output saved to: {args.output}")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def cmd_subdivide(args):
    """细分命令"""
    print(f"Subdividing: {args.input} -> {args.output}")
    print(f"Algorithm: {args.algorithm}, Iterations: {args.iterations}")
    
    try:
        if args.algorithm == 'loop':
            result = loop_subdivide(args.input, args.output, iterations=args.iterations)
        else:
            result = midpoint_subdivide(args.input, args.output, iterations=args.iterations)
        print(f"✓ Success! Output saved to: {args.output}")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def cmd_repair(args):
    """修复命令"""
    print(f"Repairing: {args.input} -> {args.output}")
    
    try:
        result = repair_topology_pymeshlab(
            args.input,
            args.output,
            remove_duplicates=not args.keep_duplicates,
            remove_unreferenced=not args.keep_unreferenced,
            close_holes=not args.no_close_holes,
            max_hole_size=args.max_hole_size
        )
        print(f"✓ Success! Output saved to: {args.output}")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def cmd_diagnose(args):
    """诊断命令"""
    print(f"Diagnosing: {args.input}\n")
    
    try:
        result = diagnose_mesh(args.input)
        return 0 if result['healthy'] else 1
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def cmd_lod(args):
    """LOD 生成命令"""
    print(f"Generating LOD chain from: {args.input}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        configs = []
        if args.preset:
            from core.lod import LOD_PRESETS
            if args.preset not in LOD_PRESETS:
                print(f"✗ Unknown preset: {args.preset}")
                print(f"Available presets: {', '.join(LOD_PRESETS.keys())}")
                return 1
            configs = LOD_PRESETS[args.preset]
        else:
            # Default LOD configs
            configs = [
                LODConfig('LOD0', 10000, 0),
                LODConfig('LOD1', 3000, 0),
                LODConfig('LOD2', 1000, 0),
            ]
        
        generator = LODGenerator(args.output_dir)
        results = generator.generate(args.input, configs)
        generator.print_report()
        
        print(f"\n✓ Success! Generated {len(results)} LOD levels")
        return 0
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Python Mesh Processing Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decimate mesh to 1000 faces
  python mesh_tool.py decimate input.ply output.ply -t 1000
  
  # Subdivide mesh using Loop algorithm
  python mesh_tool.py subdivide input.ply output.ply -a loop -i 2
  
  # Repair mesh topology
  python mesh_tool.py repair input.ply output.ply
  
  # Diagnose mesh issues
  python mesh_tool.py diagnose input.ply
  
  # Generate LOD chain
  python mesh_tool.py lod input.ply -o ./lods --preset game_character
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Decimate command
    decimate_parser = subparsers.add_parser('decimate', help='Decimate mesh (reduce polygons)')
    decimate_parser.add_argument('input', help='Input mesh file')
    decimate_parser.add_argument('output', help='Output mesh file')
    decimate_parser.add_argument('-t', '--target-faces', type=int, default=5000,
                                help='Target number of faces (default: 5000)')
    decimate_parser.add_argument('-m', '--method', choices=['open3d', 'pymeshlab'], 
                                default='open3d',
                                help='Decimation method (default: open3d)')
    decimate_parser.add_argument('--no-preserve-normal', action='store_true',
                                help='Do not preserve normals')
    decimate_parser.add_argument('--no-preserve-topology', action='store_true',
                                help='Do not preserve topology')
    decimate_parser.set_defaults(func=cmd_decimate)
    
    # Subdivide command
    subdivide_parser = subparsers.add_parser('subdivide', help='Subdivide mesh (increase polygons)')
    subdivide_parser.add_argument('input', help='Input mesh file')
    subdivide_parser.add_argument('output', help='Output mesh file')
    subdivide_parser.add_argument('-a', '--algorithm', choices=['loop', 'midpoint'],
                                 default='loop', help='Subdivision algorithm (default: loop)')
    subdivide_parser.add_argument('-i', '--iterations', type=int, default=1,
                                 help='Number of iterations (default: 1)')
    subdivide_parser.set_defaults(func=cmd_subdivide)
    
    # Repair command
    repair_parser = subparsers.add_parser('repair', help='Repair mesh topology')
    repair_parser.add_argument('input', help='Input mesh file')
    repair_parser.add_argument('output', help='Output mesh file')
    repair_parser.add_argument('--keep-duplicates', action='store_true',
                              help='Keep duplicate vertices')
    repair_parser.add_argument('--keep-unreferenced', action='store_true',
                              help='Keep unreferenced vertices')
    repair_parser.add_argument('--no-close-holes', action='store_true',
                              help='Do not close holes')
    repair_parser.add_argument('--max-hole-size', type=int, default=100,
                              help='Maximum hole size to close (default: 100)')
    repair_parser.set_defaults(func=cmd_repair)
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Diagnose mesh issues')
    diagnose_parser.add_argument('input', help='Input mesh file')
    diagnose_parser.set_defaults(func=cmd_diagnose)
    
    # LOD command
    lod_parser = subparsers.add_parser('lod', help='Generate LOD chain')
    lod_parser.add_argument('input', help='Input mesh file')
    lod_parser.add_argument('-o', '--output-dir', default='./lod_output',
                           help='Output directory (default: ./lod_output)')
    lod_parser.add_argument('--preset', choices=['game_character', 'game_prop', 
                                                 'archviz', 'mobile'],
                           help='Use preset LOD configuration')
    lod_parser.set_defaults(func=cmd_lod)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
