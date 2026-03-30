"""
Utility Functions for Mesh Processing
网格处理工具函数
"""

import os
import time
from pathlib import Path
from typing import Union, Tuple, Optional
from functools import wraps

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


def timer(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  [Timer] {func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper


def get_mesh_info(mesh_path: Union[str, Path]) -> dict:
    """
    获取网格基本信息
    
    Args:
        mesh_path: 网格文件路径
    
    Returns:
        dict: 包含顶点数、面数、边界框等信息
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required.")
    
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    
    # 计算边界框
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    
    info = {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.triangles),
        'edges': mesh.get_edge_statistics()[0] if hasattr(mesh, 'get_edge_statistics') else None,
        'is_watertight': mesh.is_watertight(),
        'is_edge_manifold': mesh.is_edge_manifold(),
        'is_vertex_manifold': mesh.is_vertex_manifold(),
        'bounds': {
            'min': bbox.min_bound.tolist(),
            'max': bbox.max_bound.tolist(),
            'extent': extent.tolist(),
            'center': bbox.get_center().tolist()
        },
        'surface_area': mesh.get_surface_area(),
        'volume': mesh.get_volume() if mesh.is_watertight() else None
    }
    
    return info


def compare_meshes(mesh_path1: Union[str, Path], mesh_path2: Union[str, Path]) -> dict:
    """
    比较两个网格
    
    Args:
        mesh_path1: 第一个网格路径
        mesh_path2: 第二个网格路径
    
    Returns:
        dict: 比较结果
    """
    info1 = get_mesh_info(mesh_path1)
    info2 = get_mesh_info(mesh_path2)
    
    comparison = {
        'mesh1': info1,
        'mesh2': info2,
        'vertex_ratio': info2['vertices'] / info1['vertices'] if info1['vertices'] > 0 else 0,
        'face_ratio': info2['faces'] / info1['faces'] if info1['faces'] > 0 else 0,
        'size_reduction': 1 - (info2['faces'] / info1['faces']) if info1['faces'] > 0 else 0
    }
    
    print(f"\nMesh Comparison:")
    print(f"  Mesh 1: {info1['vertices']} vertices, {info1['faces']} faces")
    print(f"  Mesh 2: {info2['vertices']} vertices, {info2['faces']} faces")
    print(f"  Face ratio: {comparison['face_ratio']:.2%}")
    print(f"  Size reduction: {comparison['size_reduction']:.2%}")
    
    return comparison


def estimate_file_size(mesh_path: Union[str, Path], format: str = 'obj') -> int:
    """
    估算不同格式的文件大小
    
    Args:
        mesh_path: 网格文件路径
        format: 目标格式 ('obj', 'ply', 'stl')
    
    Returns:
        int: 估算的字节数
    """
    info = get_mesh_info(mesh_path)
    vertices = info['vertices']
    faces = info['faces']
    
    # 粗略估算
    if format == 'obj':
        # v x y z\n + f v1 v2 v3\n
        vertex_line = 30  # "v 0.123456 0.123456 0.123456\n"
        face_line = 20    # "f 12345 12345 12345\n"
        return int(vertices * vertex_line + faces * face_line)
    
    elif format == 'ply':
        # PLY 格式头 + 二进制数据
        header = 500
        vertex_data = vertices * 12  # 3 floats
        face_data = faces * 13       # 1 byte count + 3 ints
        return header + vertex_data + face_data
    
    elif format == 'stl':
        # STL 二进制格式
        header = 80
        triangle_data = faces * 50  # 12 floats per triangle
        return header + 4 + triangle_data
    
    else:
        raise ValueError(f"Unknown format: {format}")


def batch_process(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    process_func,
    pattern: str = "*.obj"
):
    """
    批量处理网格文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        process_func: 处理函数，接收 (input_path, output_path)
        pattern: 文件匹配模式
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob(pattern))
    print(f"Found {len(files)} files matching '{pattern}'")
    
    success_count = 0
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing {file_path.name}...")
        try:
            output_path = output_dir / file_path.name
            process_func(file_path, output_path)
            success_count += 1
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nCompleted: {success_count}/{len(files)} files processed successfully")


def visualize_mesh(mesh_path: Union[str, Path], window_name: str = "Mesh"):
    """
    可视化网格
    
    Args:
        mesh_path: 网格文件路径
        window_name: 窗口标题
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required for visualization.")
    
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    
    o3d.visualization.draw_geometries(
        [mesh],
        window_name=window_name,
        width=1024,
        height=768
    )


def create_test_mesh(mesh_type: str = 'cube', size: float = 1.0) -> 'o3d.geometry.TriangleMesh':
    """
    创建测试网格
    
    Args:
        mesh_type: 'cube', 'sphere', 'torus', 'cylinder'
        size: 网格大小
    
    Returns:
        Open3D TriangleMesh
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required.")
    
    if mesh_type == 'cube':
        mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
    elif mesh_type == 'sphere':
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=size/2)
    elif mesh_type == 'torus':
        mesh = o3d.geometry.TriangleMesh.create_torus(tube_radius=size/4, torus_radius=size/2)
    elif mesh_type == 'cylinder':
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=size/3, height=size)
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")
    
    mesh.compute_vertex_normals()
    return mesh


def convert_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    write_triangle_normals: bool = True,
    write_vertex_normals: bool = True,
    write_vertex_colors: bool = False
):
    """
    转换网格格式
    
    Args:
        input_path: 输入路径
        output_path: 输出路径
        write_triangle_normals: 写入面法线
        write_vertex_normals: 写入顶点法线
        write_vertex_colors: 写入顶点颜色
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required.")
    
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    o3d.io.write_triangle_mesh(
        str(output_path),
        mesh,
        write_triangle_normals=write_triangle_normals,
        write_vertex_normals=write_vertex_normals,
        write_vertex_colors=write_vertex_colors
    )
    
    print(f"Converted: {input_path} -> {output_path}")


def get_supported_formats() -> dict:
    """
    获取支持的文件格式
    
    Returns:
        dict: 格式信息
    """
    return {
        'read': ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx'],
        'write': ['.obj', '.ply', '.stl', '.off']
    }
