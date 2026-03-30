"""
Mesh Decimation (QEM Algorithm) Implementation
减面模块 - 使用 QEM 算法减少网格面数
"""

import os
from pathlib import Path
from typing import Union, Optional

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


def decimate_mesh_pymeshlab(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_faces: int = 10000,
    preservenormal: bool = True,
    preservetopology: bool = True,
    optimalplacement: bool = True
):
    """
    使用 PyMeshLab 进行 QEM 减面（推荐用于生产环境）
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        target_faces: 目标面数
        preservenormal: 保持法线方向
        preservetopology: 保持拓扑结构（避免面片翻转）
        optimalplacement: 优化顶点位置
    
    Returns:
        pymeshlab.Mesh: 简化后的网格
    
    Raises:
        ImportError: 如果 PyMeshLab 未安装
        FileNotFoundError: 如果输入文件不存在
    """
    if not PYMESHLAB_AVAILABLE:
        raise ImportError("PyMeshLab is required. Install with: pip install pymeshlab")
    
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))
    
    original_faces = ms.current_mesh().face_number()
    
    # 核心减面操作: QEM 算法
    ms.apply_filter(
        'simplification_quadric_edge_collapse_decimation',
        targetfacenum=target_faces,
        preservenormal=preservenormal,
        preservetopology=preservetopology,
        optimalplacement=optimalplacement
    )
    
    result = ms.current_mesh()
    print(f"[PyMeshLab] Decimated: {original_faces} -> {result.face_number()} faces")
    
    ms.save_current_mesh(str(output_path))
    return result


def decimate_mesh_open3d(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_faces: int = 5000,
    remove_non_manifold: bool = True
):
    """
    使用 Open3D 快速减面（轻量级）
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        target_faces: 目标三角形数量
        remove_non_manifold: 是否先移除非流形边
    
    Returns:
        o3d.geometry.TriangleMesh: 简化后的网格
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required. Install with: pip install open3d")
    
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    
    original_faces = len(mesh.triangles)
    
    # 修复非流形边（重要！）
    if remove_non_manifold and not mesh.is_edge_manifold():
        mesh = mesh.remove_non_manifold_edges()
    
    simplified = mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )
    
    simplified.compute_vertex_normals()
    
    print(f"[Open3D] Decimated: {original_faces} -> {len(simplified.triangles)} faces")
    
    o3d.io.write_triangle_mesh(str(output_path), simplified)
    return simplified


def decimate_mesh_percentage(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_percentage: float = 0.5,
    method: str = "pymeshlab"
):
    """
    按比例减面
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        target_percentage: 目标百分比 (0-1)
        method: 使用的方法 ("pymeshlab" 或 "open3d")
    """
    if target_percentage <= 0 or target_percentage > 1:
        raise ValueError("target_percentage must be in (0, 1]")
    
    if method == "pymeshlab":
        if not PYMESHLAB_AVAILABLE:
            raise ImportError("PyMeshLab is required.")
        
        input_path = Path(input_path)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(input_path))
        
        original_faces = ms.current_mesh().face_number()
        target_faces = int(original_faces * target_percentage)
        
        ms.apply_filter(
            'simplification_quadric_edge_collapse_decimation',
            targetfacenum=target_faces,
            preservenormal=True,
            preservetopology=True
        )
        
        os.makedirs(Path(output_path).parent, exist_ok=True)
        ms.save_current_mesh(str(output_path))
        return ms.current_mesh()
    
    elif method == "open3d":
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is required.")
        
        mesh = o3d.io.read_triangle_mesh(str(input_path))
        original_faces = len(mesh.triangles)
        target_faces = int(original_faces * target_percentage)
        
        simplified = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )
        
        os.makedirs(Path(output_path).parent, exist_ok=True)
        o3d.io.write_triangle_mesh(str(output_path), simplified)
        return simplified
    
    else:
        raise ValueError(f"Unknown method: {method}")


# 便捷函数
decimate = decimate_mesh_pymeshlab
quick_decimate = decimate_mesh_open3d
