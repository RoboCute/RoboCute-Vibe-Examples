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
        
        # Step 1: 初始清理
        cleaned_path = self.output_dir / 'cleaned_highpoly.ply'
        
        if cleanup_first:
            print("[Step 1] Initial Cleanup...")
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(input_path))
            
            ms.apply_filter('remove_duplicate_vertices')
            ms.apply_filter('remove_unreferenced_vertices')
            ms.apply_filter('close_holes', maxholesize=50)
            ms.apply_filter('re_orient_all_faces_coherentely')
            
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
        """处理单个 LOD"""
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(input_path))
        
        # 减面
        ms.apply_filter(
            'simplification_quadric_edge_collapse_decimation',
            targetfacenum=config.target_faces,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True
        )
        
        mesh = ms.current_mesh()
        
        # 如果需要细分（通常用于高模）
        if config.subdiv_iterations > 0 and OPEN3D_AVAILABLE:
            temp_path = self.output_dir / f'temp_{config.name}.ply'
            ms.save_current_mesh(str(temp_path))
            
            o3d_mesh = o3d.io.read_triangle_mesh(str(temp_path))
            
            if o3d_mesh.is_edge_manifold():
                o3d_mesh = o3d_mesh.subdivide_loop(number_of_iterations=config.subdiv_iterations)
                o3d_mesh.compute_vertex_normals()
                
                output_path = self.output_dir / f'{config.name}.ply'
                o3d.io.write_triangle_mesh(str(output_path), o3d_mesh)
                print(f"  ✓ {config.name}: {len(o3d_mesh.triangles)} faces (with {config.subdiv_iterations}x subdivision)")
                
                # 清理临时文件
                temp_path.unlink(missing_ok=True)
                return o3d_mesh
            else:
                print(f"  ⚠ Skipping subdivision (non-manifold)")
        
        # 保存结果
        output_path = self.output_dir / f'{config.name}.ply'
        ms.save_current_mesh(str(output_path))
        print(f"  ✓ {config.name}: {mesh.face_number()} faces")
        
        return mesh
    
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
