"""
LOD (Level of Detail) Generation Workflow
LOD 生成工作流 - 批量生成不同细节层次的模型
"""

import os
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any
from dataclasses import dataclass

try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


@dataclass
class LODConfig:
    """LOD 配置类"""
    name: str           # LOD 名称
    target_faces: int   # 目标面数
    subdiv_iterations: int = 0  # 细分迭代次数（通常用于高模）
    
    def __repr__(self):
        return f"LODConfig(name='{self.name}', faces={self.target_faces}, subdiv={self.subdiv_iterations})"


class LODGenerator:
    """
    LOD 生成器
    
    生成 LOD 链的完整流程：
    1. 初始清理
    2. 按配置生成各级 LOD
    3. 可选的细分处理
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Tuple[str, Any]] = []
    
    def generate(
        self,
        input_path: Union[str, Path],
        lod_configs: List[LODConfig],
        cleanup_first: bool = True
    ) -> List[Tuple[str, Any]]:
        """
        生成 LOD 链
        
        Args:
            input_path: 高模路径
            lod_configs: LOD 配置列表
            cleanup_first: 是否先进行初始清理
        
        Returns:
            List[(lod_name, mesh)]: 生成的 LOD 列表
        """
        if not PYMESHLAB_AVAILABLE:
            raise ImportError("PyMeshLab is required.")
        
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        print(f"\n{'='*60}")
        print(f"LOD Generation Started")
        print(f"Input: {input_path}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Step 1: 初始清理 (使用 Open3D 更稳定的 API)
        cleaned_path = self.output_dir / 'cleaned_highpoly.ply'
        
        if cleanup_first and OPEN3D_AVAILABLE:
            print("[Step 1] Initial Cleanup...")
            mesh = o3d.io.read_triangle_mesh(str(input_path))
            
            # 使用 Open3D 进行清理
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()
            mesh.remove_degenerate_triangles()
            mesh.compute_vertex_normals()
            
            o3d.io.write_triangle_mesh(str(cleaned_path), mesh)
            print(f"  Cleaned: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces\n")
        elif cleanup_first:
            # Fallback to PyMeshLab with available filters
            print("[Step 1] Initial Cleanup (PyMeshLab)...")
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(input_path))
            # Only apply filters that are known to exist
            ms.apply_filter('simplification_quadric_edge_collapse_decimation', 
                           targetfacenum=ms.current_mesh().face_number(),
                           preservenormal=True)
            ms.save_current_mesh(str(cleaned_path))
            mesh = ms.current_mesh()
            print(f"  Cleaned: {mesh.vertex_number()} vertices, {mesh.face_number()} faces\n")
        else:
            cleaned_path = input_path
        
        # Step 2: 生成各级 LOD
        self.results = []
        
        for i, config in enumerate(lod_configs):
            print(f"[{i+2}] Processing {config.name} (target: {config.target_faces} faces)...")
            
            try:
                result = self._process_lod(cleaned_path, config)
                self.results.append((config.name, result))
            except Exception as e:
                print(f"  ERROR: {e}")
        
        print(f"\n{'='*60}")
        print(f"LOD Generation Complete! Generated {len(self.results)} levels")
        print(f"{'='*60}\n")
        
        return self.results
    
    def _process_lod(self, input_path: Path, config: LODConfig):
        """处理单个 LOD - 返回 Open3D 网格以确保对象有效性"""
        output_path = self.output_dir / f'{config.name}.ply'
        
        # 使用 Open3D 进行减面（更稳定的 API）
        if OPEN3D_AVAILABLE:
            o3d_mesh = o3d.io.read_triangle_mesh(str(input_path))
            current_faces = len(o3d_mesh.triangles)
            
            # 减面
            if current_faces > config.target_faces:
                o3d_mesh = o3d_mesh.simplify_quadric_decimation(
                    target_number_of_triangles=config.target_faces
                )
            
            # 如果需要细分（通常用于高模）
            if config.subdiv_iterations > 0:
                if o3d_mesh.is_edge_manifold():
                    o3d_mesh = o3d_mesh.subdivide_loop(number_of_iterations=config.subdiv_iterations)
                    o3d_mesh.compute_vertex_normals()
                    print(f"  [OK] {config.name}: {len(o3d_mesh.triangles)} faces (with {config.subdiv_iterations}x subdivision)")
                else:
                    print(f"  [WARN] Skipping subdivision (non-manifold)")
            
            o3d.io.write_triangle_mesh(str(output_path), o3d_mesh)
            print(f"  [OK] {config.name}: {len(o3d_mesh.triangles)} faces")
            return o3d_mesh
        else:
            # Fallback to PyMeshLab
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(input_path))
            ms.apply_filter(
                'meshing_decimation_quadric_edge_collapse',
                targetfacenum=config.target_faces,
                preservenormal=True,
                preservetopology=True
            )
            ms.save_current_mesh(str(output_path))
            mesh = ms.current_mesh()
            print(f"  [OK] {config.name}: {mesh.face_number()} faces")
            # 重新加载为 Open3D 网格返回以确保对象有效性
            return o3d.io.read_triangle_mesh(str(output_path))
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """
        获取 LOD 统计信息
        
        Returns:
            Dict: 各级 LOD 的统计信息
        """
        stats = {}
        for name, mesh in self.results:
            if hasattr(mesh, 'vertex_number'):
                # PyMeshLab mesh
                stats[name] = {
                    'vertices': mesh.vertex_number(),
                    'faces': mesh.face_number()
                }
            else:
                # Open3D mesh
                stats[name] = {
                    'vertices': len(mesh.vertices),
                    'faces': len(mesh.triangles)
                }
        return stats
    
    def print_report(self):
        """打印生成报告"""
        stats = self.get_stats()
        
        print("\nLOD Generation Report:")
        print("-" * 50)
        print(f"{'Level':<15} {'Vertices':<12} {'Faces':<12}")
        print("-" * 50)
        
        for name, s in stats.items():
            print(f"{name:<15} {s['vertices']:<12} {s['faces']:<12}")
        
        print("-" * 50)


def generate_lod_chain(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    lod_configs: List[Tuple[str, int, int]]
) -> List[Tuple[str, Any]]:
    """
    快速生成 LOD 链的便捷函数
    
    Args:
        input_path: 高模路径
        output_dir: 输出目录
        lod_configs: [(name, target_faces, subdiv_iterations), ...]
                    例如: [('LOD0', 50000, 0), ('LOD1', 10000, 0)]
    
    Returns:
        List[(lod_name, mesh)]: 生成的 LOD 列表
    """
    configs = [LODConfig(name, faces, subdiv) for name, faces, subdiv in lod_configs]
    generator = LODGenerator(output_dir)
    return generator.generate(input_path, configs)


# 预设配置
LOD_PRESETS = {
    'game_character': [
        LODConfig('LOD0_High', 50000, 0),
        LODConfig('LOD1_Medium', 15000, 0),
        LODConfig('LOD2_Low', 5000, 0),
        LODConfig('LOD3_UltraLow', 1000, 0),
    ],
    'game_prop': [
        LODConfig('LOD0', 10000, 0),
        LODConfig('LOD1', 3000, 0),
        LODConfig('LOD2', 500, 0),
    ],
    'archviz': [
        LODConfig('LOD0_High', 100000, 0),
        LODConfig('LOD1_Medium', 30000, 0),
        LODConfig('LOD2_Low', 10000, 0),
    ],
    'mobile': [
        LODConfig('LOD0', 10000, 0),
        LODConfig('LOD1', 2500, 0),
        LODConfig('LOD2', 600, 0),
        LODConfig('LOD3', 150, 0),
    ]
}


def generate_preset_lod(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    preset: str = 'game_character'
):
    """
    使用预设配置生成 LOD
    
    Args:
        input_path: 高模路径
        output_dir: 输出目录
        preset: 预设名称 ('game_character', 'game_prop', 'archviz', 'mobile')
    """
    if preset not in LOD_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(LOD_PRESETS.keys())}")
    
    generator = LODGenerator(output_dir)
    results = generator.generate(input_path, LOD_PRESETS[preset])
    generator.print_report()
    return results
